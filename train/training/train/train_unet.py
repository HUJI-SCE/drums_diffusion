import sys
import torch
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from diffusers import DDPMPipeline
# from diffusers.utils import make_image_grid
from diffusers import AudioLDM2Pipeline, DDPMPipeline
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

from configs import TrainingConfig
from data.dataset.dataset import CustomAudioDataset
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline, AudioDiffusionPipeline
import wandb

def load_pretrained_models(dir_path):
    """
    Loads pretrained UNet, AutoencoderKL, and Vocoder models from the specified directory path.

    Args:
    - dir_path (str): Path to the directory containing the pretrained models.

    Returns:
    - models (dict): A dictionary containing the loaded models.
    """

    p1 = os.path.join(dir_path, "small_unet_dir")
    p2 = os.path.join(dir_path, "vae")
    p3 = os.path.join(dir_path, "vocoder")
    p4 = os.path.join(dir_path, "scheduler")
    # models = {
    #     "unet": torch.load(f"{dir_path}/try_unet",map_location=torch.device('cpu')),
    #     "vae": torch.load(f"{dir_path}/vae",map_location=torch.device('cpu')).half(),
    #     "vocoder": torch.load(f"{dir_path}/vocoder",map_location=torch.device('cpu')),
    #     "scheduler": torch.load(f"{dir_path}/scheduler",map_location=torch.device('cpu'))
    #     "unet": torch.load(p1).cuda(),
    #     "vae": torch.load(p2).cuda(),
    #     "vocoder": torch.load(p3),
    #     "scheduler": torch.load(p4)
    # }
    unet = DiffusionPipeline.from_pretrained(p1, use_safetensors=True)
    return unet, torch.load(p2, map_location=torch.device('cpu')), \
           torch.load(p3, map_location=torch.device('cpu')), torch.load(p4, map_location=torch.device('cpu'))


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # images = pipeline(
    #     batch_size=config.eval_batch_size,
    #     generator=torch.manual_seed(config.seed),
    # ).images
    #
    # # Make a grid out of the images
    # image_grid = make_image_grid(images, rows=4, cols=4)
    #
    # # Save the images
    # test_dir = os.path.join(config.output_dir, "samples")
    # os.makedirs(test_dir, exist_ok=True)
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    pass


def train_loop(config, unet, vae, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with='wandb',
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    # accelerator.init_trackers("example_project", config=config)
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        # accelerator.init_trackers(
        #     project_name="my_project",
        #     config={"dropout": 0.1, "learning_rate": 1e-2},
        # init_kwargs = {"wandb": {"entity": "my-wandb-team"}}
        # )
        accelerator.init_trackers("example_project", config=config)
        # accelerator.init_trackers("train_example")
    # accelerator.init_trackers("example_project", config=config)
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    unet, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, train_dataloader, lr_scheduler
    )
    unet.train()

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            # TODO: GET ITEM RETURN (NO_DRUM_SPEC,DRUM_SPEC)
            batch_no_drum_spec, batch_drum_spec = batch


            latents_drums = (vae.encode(batch_drum_spec.unsqueeze(1)).latent_dist.sample() * 0.18215)
            latents_no_drums = (vae.encode(batch_no_drum_spec.unsqueeze(1)).latent_dist.sample() * 0.18215)

            # noise_minus_drums = noise + latents_drums
            batch_size = batch_drum_spec.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=batch_drum_spec.device,
                dtype=torch.int64
            )

            latent_drums_as_noise = latents_no_drums - latents_drums
            noise = latent_drums_as_noise
            noisy_latents = noise_scheduler.add_noise(latents_drums, latent_drums_as_noise, timesteps)


            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=None, return_dict=False)[0]

                    # Calculate the loss
                    # with accelerator.autocast():
                    loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            # logger.info(logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(unet), scheduler=noise_scheduler)
            if (epoch + 1) % config.eval_epoch == 0 or epoch == config.num_epochs - 1:
                pass
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=config.repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == '__main__':
    config = TrainingConfig()
    wandb.login(key=config.wandb)
    csv_file_path = config.annotation_file_path
    dataset = CustomAudioDataset(csv_file_path)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    my_pipe = DiffusionPipeline.from_pretrained(config.path_to_components, custom_pipeline=config.path_to_class_file)
    unet, vae, noise_scheduler = my_pipe.unet, my_pipe.vae, my_pipe.scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataset) * config.num_epochs),
    )
    train_loop(config, unet, vae, noise_scheduler, optimizer, dataloader, lr_scheduler)
