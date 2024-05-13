import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# from torchvision.datasets import MNIST
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from mpl_toolkits.axes_grid1 import ImageGrid
# from torchvision.utils import save_image, make_grid

from torch.optim import Adam
from diffusers import AutoencoderKL
from tqdm import tqdm
# create a transofrm to apply to each datapoint
# transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
# path = '~/datasets'
# train_dataset = MNIST(path, transform=transform, download=True)
# test_dataset  = MNIST(path, transform=transform, download=True)
#
# # create train and test dataloaders
batch_size = 100
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
import os
from data.dataset.dataset import CustomAudioDataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass
class TrainingConfig:


    root_dir = "/Users/mac/pythonProject1/pythonProject/train_file"
    output_dir = os.path.join(root_dir, "out_dir")
    models_dir = os.path.join(root_dir, "../../models")
    csv_file_path = os.path.join(root_dir, "anno.csv")


    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 2000
    eval_epoch = 10
    save_model_epochs = 5
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
        block_out_channels=(128,256,512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=1,
        norm_num_groups=32,
        sample_size=1024,
        scaling_factor=0.18215,
        force_upcast=True,
    )
model = vae.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat):
    KLD = nn.functional.kl_div(x_hat, x)
    return KLD


def train(model,config ,optimizer, device, data_loader):
    model.train()
    for epoch in tqdm(range(config.num_epochs)):
        overall_loss = 0
        b=0
        for batch_idx, (x, _) in enumerate(data_loader):
            b=batch_idx
            x = x.view(batch_size, x.shape).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (b * config.train_batch_size))
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            torch.save(model.state_dict(), config.models_dir + f"/vae_state_dict.pt")
    return

if __name__ == '__main__':
    vae_path = "vae"
    config = TrainingConfig()
    dataset = CustomAudioDataset(config.csv_file_path)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=1,
        norm_num_groups=32,
        sample_size=32,
        scaling_factor=0.18215,
        force_upcast=True,
    )
    train(vae, config, optimizer, device, dataloader)

    # ).half()
    # print(vae.num_parameters())
    # vae.eval()
    # x = torch.randn((1, 1, 1024, 64), dtype=torch.float16)
    # latents_drums = vae.encode(x).latent_dist.sample() * 0.18215
    # print(latents_drums.shape)

    torch.save(vae, vae_path)
    # x_hat, mean, log_var = vae(x)

