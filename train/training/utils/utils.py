import diffusers.pipelines.audioldm2.modeling_audioldm2 as A
from diffusers import MusicLDMPipeline, DiffusionPipeline, Mel
import torch
import librosa
import torchaudio


# Usage Example
# create_dataset_csv('/path/to/your/directory')



def create_custom_unet():
    ynet = A.AudioLDM2UNet2DConditionModel(
        sample_size=256,
        in_channels=8,
        out_channels=8,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        only_cross_attention=False,
        block_out_channels=(128, 256, 384, 640),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        # cross_attention_dim = ([None,1024,64],[None,1024,64],[None,1024,64],[None,1024,64]),
        # cross_attention_dim=[[None, None], [None, None], [None, None], [None, None]],
        cross_attention_dim=[[None], [None], [None], [None]],
        # cross_attention_dim = [[None,64],[None,64],[None,64],[None,64]],
        transformer_layers_per_block=1,
        attention_head_dim=8,
        num_attention_heads=None,
        use_linear_projection=False,
        class_embed_type=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        class_embeddings_concat=False,
    )
    print(ynet.num_parameters())
    return ynet


def create_mel_object():
    mel = Mel(
            x_res = 256,
            y_res = 64,
            sample_rate = 44100,
            n_fft = 1024,
            hop_length = 160,
            top_db = 0,
            n_iter = 32,
    )
    return mel


def create_custom_pipeline():
    from inference import pipline
    mel = create_mel_object()
    unet = create_custom_unet()
    repo_id = "ucsd-reach/musicldm"
    pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    my_pipe = pipline.MyPipeline(vae=pipe.vae,
                                 mel=mel,
                                 feature_extractor=pipe.feature_extractor,
                                 unet = unet,
                                 scheduler=pipe.scheduler,
                                 vocoder=pipe.vocoder,
                                 )
    # my_pipe.push_to_hub('drums_diff')
    path = '/models'
    my_pipe.save_pretrained(path)
    return pipe







