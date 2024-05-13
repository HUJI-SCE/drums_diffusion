import torch
from diffusers import DiffusionPipeline, Mel, AudioPipelineOutput
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from configs import TrainingConfig
import numpy as np
import torch
from transformers import (
    ClapFeatureExtractor,
    SpeechT5HifiGan,
)
import torchaudio
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
# from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
# from diffusers.pipelines.deprecated.audio_diffusion import mel
# # from diffusers.pipelines.musicldm import MusicLDMPipeline
if is_librosa_available():
    import librosa

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import MusicLDMPipeline
        >>> import torch
        >>> import scipy

        >>> repo_id = "ucsd-reach/musicldm"
        >>> pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> audio_prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        >>> audio = pipe(audio_prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

        >>> # save the audio sample as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```
"""
SAMPLE_RATE = TrainingConfig.SAMPLE_RATE
N_FFT = TrainingConfig.N_FFT
HOP_LENGTH = TrainingConfig.HOP_LENGTH
WIN_LENGTH = TrainingConfig.WIN_LENGTH
N_MELS = TrainingConfig.N_MELS
FMAX = TrainingConfig.FMAX
FMIN = TrainingConfig.FMIN
AUDIO_LEN_SEC = TrainingConfig.AUDIO_LEN_SEC
TARGET_LENGTH = TrainingConfig.TARGET_MEL_LENGTH
NUM_SAMPLES = TrainingConfig.NUM_SAMPLES
TARGET_LENGTH_SEC = TrainingConfig.TARGET_LENGTH_SEC

class MyPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            feature_extractor: Optional[ClapFeatureExtractor],
            mel: Mel,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            # feature_extractor=feature_extractor,
            feature_extractor=None,
            # mel=mel,
            mel = None,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = [
            self.feature_extractor,
            self.unet,
            self.vae,
            self.vocoder,
        ]

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            print(f"mel_spectrogram shape: {mel_spectrogram.shape}")
            mel_spectrogram = mel_spectrogram.squeeze(1)
        mel_spectrogram = torch.reshape(mel_spectrogram, (mel_spectrogram.shape[0], -1,mel_spectrogram.shape[1]))
        print(f"mel_spectrogram shape: {mel_spectrogram.shape}")
        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
            self,
            audio_prompt,
            audio_file_path,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            latents,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if audio_prompt is not None and latents is not None:
            raise ValueError(
                f"Cannot forward both `audio_prompt` and `latents` be supplied . Please make sure to"
                " only forward one of the two."
            )
        elif audio_prompt is None and latents is None and audio_file_path is None:
            raise ValueError(
                # "Provide either `audio_prompt` or `latents`. Cannot leave both `audio_prompt` and `latents` undefined."
                "Provide either `audio_prompt`, `latents` or `audio_file_path`. Cannot leave all three undefined."
            )
        elif audio_prompt is not None and (not isinstance(audio_prompt, np.ndarray) and not isinstance(audio_prompt, list) and not isinstance(audio_prompt, torch.Tensor)):
            raise ValueError(f"`audio_prompt` has to be of type `np.ndarray` or `list` but is {type(audio_prompt)}")


    def extract_log_mel_spectrogram(self, audio,
                                    sr=SAMPLE_RATE,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    n_mels=N_MELS,
                                    f_min=FMIN,
                                    f_max=FMAX,
                                    win_length=WIN_LENGTH
                                    ):
        """
        Extracts the log Mel spectrogram from an audio signal.

        Parameters:
        - audio: numpy array, the audio signal.
        - sr: int, the sampling rate of the audio signal.
        - n_fft: int, the length of the FFT window.
        - hop_length: int, the number of samples between successive frames.
        - n_mels: int, the number of Mel bands.

        Returns:
        - log_mel_spectrogram: numpy array of shape (1, 1, 1024, 64), the log Mel spectrogram.
        """
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                         n_mels=n_mels, fmin=f_min, fmax=f_max,
                                                         win_length=win_length,
                                                         center=True)
        # Convert to log scale (dB)

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram

    #TODO: Implement the extract_mels method
    def extract_mels(self,
                     audio_prompt: Union[np.ndarray, List[np.ndarray]] = None,
                     ):
        arr = np.array(audio_prompt)
        log_mel_spectrogram = self.extract_log_mel_spectrogram(arr)
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return log_mel_spectrogram



    def prepare_latents(self,audio_prompt, audio_file_path ,batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if audio_prompt is None and audio_file_path is None:
                raise ValueError("the model requires an audio audio_prompt or latents to be passed to the model.")
            elif self.mel is not None:
                self.mel.load_audio(audio_file_path, audio_prompt)
                slice = self.mel.get_number_of_slices()
                input_image = self.mel.audio_slice_to_image(slice)
                input_image = np.frombuffer(input_image.tobytes(), dtype="uint8").reshape(
                    (input_image.height, input_image.width)
                )
                input_image = (input_image / 255) * 2 - 1
                input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

                latents = self.vae.encode(input_images).latent_dist.sample().to(device)

                #TODO: check if the return value of the latents is correct
                return latents * self.scheduler.init_noise_sigma
            elif self.feature_extractor is None:
                # if audio_file_path is not None:
                #     audio_prompt,sr = librosa.load(audio_file_path)
                mels = self.extract_mels(audio_prompt)
                print(f"mels shape: {mels.shape}")
                latents = self.vae.encode(mels).latent_dist.sample().to(device)
                print(f"latents shape: {latents.shape}")
                #TODO: check if the return value of the latents is correct
                return latents * self.scheduler.init_noise_sigma
            else:
                mels = self.feature_extractor(audio_prompt,sampling_rate=48000,return_tensors="pt")
                latents = self.vae.encode(mels.data['input_features']).latent_dist.sample().to(device)
                # TODO: check if the return value of the latents is correct
                return latents * self.scheduler.init_noise_sigma



                # if len(audio_prompt) != batch_size:
                #     raise ValueError(
                #         f"the model requires a batch size of {batch_size} audio prompts, but got {len(audio_prompt)}."
                #     )
            # if self.feature_extractor is not None:
            #     mels = self.feature_extractor(audio_prompt,sampling_rate=48000,return_tensors="pt")
            #     latents = self.vae.encode(mels.data['input_features']).latent_dist.sample().to(device)
            #     return latents * self.scheduler.init_noise_sigma
            # else:
            #     mels = self.extract_mels(audio_prompt)
            #     latents = self.vae.encode(mels).latent_dist.sample().to(device)
            #     return latents * self.scheduler.init_noise_sigma

        else:
            latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            #TODO: check if this is correct
            return latents * self.scheduler.init_noise_sigma




    @torch.no_grad()
    def __call__(self,
                 num_inference_steps: int = 50,
                 audio_prompt: Union[np.ndarray, List[np.ndarray]] = None,
                 sr: int = 44100,
                 audio_file_path: Optional[str] = None,
                 audio_length_in_s: float = 5.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 output_type: Optional[str] = "np",
                 eta: float = 0.0,
                 return_dict: bool = True,
                 ):

        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate
        if audio_file_path is not None:
            audio_prompt, sr = librosa.load(audio_file_path,sr=None)
            # if sr != self.vocoder.config.sampling_rate:
            #     raise ValueError(
            #         f"Sampling rate of the audio file is {sr}, but the model requires {self.vocoder.config.sampling_rate}"
            #     )
        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            audio_prompt,
            audio_file_path,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            latents,
        )

        # 2. Define call parameters
        if audio_prompt is not None and isinstance(audio_prompt, np.ndarray):
            batch_size = 1
        elif audio_prompt is not None and isinstance(audio_prompt, list):
            batch_size = len(audio_prompt)
        elif audio_prompt is not None and isinstance(audio_prompt, torch.Tensor):
            batch_size = audio_prompt.shape[0]
        else:
            batch_size = latents.shape[0]

        device = self._execution_device

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            audio_prompt,
            sr,
            audio_file_path,
            batch_size,
            num_channels_latents,
            height,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        self.maybe_free_model_hooks()

        # 8. Post-processing
        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            mel_spectrogram = self.vae.decode(latents).sample
        else:
            return AudioPipelineOutput(audios=latents)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

        audio = audio[:, :original_waveform_length]


        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)


def use_custom_pipe():
    path_to_components = TrainingConfig.path_to_components
    path_to_class_file = TrainingConfig.path_to_class_file
    # path_to_class_file = "/inference/pipline.py"
    my_pipe = DiffusionPipeline.from_pretrained(path_to_components, custom_pipeline=path_to_class_file)
    my_pipe.mel = None
    audio_path = '/Users/mac/Desktop/demucs_out/mdx_extra/134034/no_drums.mp3'
    # audio, sr = librosa.load(audio_path,mono=True,sr=None)
    audio,sr = torchaudio.load(audio_path)
    audio = audio[0:1, 0:NUM_SAMPLES*5]
    audio = torch.mean(audio, dim=0, keepdim=False)
    # audio = audio.unsqueeze(0)
    print(f"audio shape: {audio.shape}")
    lst = [audio,audio]
    audio_out = my_pipe(
        num_inference_steps=50,
        audio_prompt=audio,
        audio_file_path=None,
        audio_length_in_s=5.0,
        generator=None,
        latents=None,
        callback=None,
        callback_steps=1,
        output_type=None,
        eta=0.0,
        return_dict=False,
    )[0]

    torchaudio.save('/Users/mac/pythonProject1/pythonProject/fixing_training/training/utils/techno_diff.wav',
                    src=audio_out
                    , sample_rate=sr,
                    format='wav',
                    )


if __name__ == '__main__':
    # create_custom_pipeline()
    use_custom_pipe()