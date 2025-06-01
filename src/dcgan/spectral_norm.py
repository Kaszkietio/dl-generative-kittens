import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from torchvision import utils as vutils
from tqdm import tqdm

from dcgan import Generator, Discriminator
from utils import set_seed, get_device
from data_utils import DATASET_PATH, CHECKPOINT_PATH, CatDataset, MEAN, STD

MEAN_TENSOR = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).cuda()
STD_TENSOR = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1).cuda()

OPTIMIZERS = {
    "SGD": SGD,
    "AdamW": AdamW
}

SCHEDULERS = {
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ReduceLROnPlateau": ReduceLROnPlateau
}

def train(
    d: Discriminator,
    g: Generator,
    ds: DataLoader,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    criterion: nn.BCELoss,
    noise_factor: float = 0.05,
    epoch: int = 0,
    num_epochs: int = 25
):
    arr_D_loss = [float("inf")]
    arr_G_loss = [float("inf")]
    arr_D_x = [float("inf")]
    arr_D_G_z1 = [float("inf")]
    arr_D_G_z2 = [float("inf")]
    batch_sizes = []

    real_label = 0.9
    fake_label = 0.0

    for i, input in enumerate(ds):
        opt_d.zero_grad()

        # input += torch.randn_like(input) * noise_factor  # Add noise to the input
        input = input.cuda()
        b_size = len(input)
        labels = torch.full((b_size,), real_label, dtype=torch.float, device=input.device)
        output = d(input).view(-1)
        err_d_real = criterion(output, labels)
        err_d_real.backward()
        D_x = output.mean().item()

        # Generate fake images
        noise = torch.randn(b_size, g.nz, 1, 1, device=input.device)
        fake_images = g(noise)
        labels.fill_(fake_label)
        output = d(torch.detach(fake_images)).view(-1)
        err_f_fake = criterion(output, labels)
        err_f_fake.backward()
        err_d = err_d_real + err_f_fake
        opt_d.step()
        D_G_z1 = output.mean().item()

        # Update generator
        opt_g.zero_grad()
        noise = torch.randn(b_size, g.nz, 1, 1, device=input.device)
        fake_images = g(noise)
        labels.fill_(real_label)  # Trick D
        output = d(fake_images).view(-1)
        err_g = criterion(output, labels)
        err_g.backward()
        D_G_z2 = output.mean().item()
        opt_g.step()

        arr_G_loss.append(err_g.item())
        arr_D_loss.append(err_d.item())
        arr_D_x.append(D_x)
        arr_D_G_z1.append(D_G_z1)
        arr_D_G_z2.append(D_G_z2)
        batch_sizes.append(b_size)

        if i % 50 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(ds)}]", end=' ')
            print(f"G_loss: {arr_G_loss[-1]:.4f}({(arr_G_loss[-1] - arr_G_loss[-2]):.2e})", end=' ')
            print(f"D_loss: {arr_D_loss[-1]:.4f}({(arr_D_loss[-1] - arr_D_loss[-2]):.2e})", end=' ')
            print(f"D_x: {arr_D_x[-1]:.4f}({(arr_D_x[-1] - arr_D_x[-2]):.2e})", end=' ')
            print(f"D_G_z1: {arr_D_G_z1[-1]:.4f}({(arr_D_G_z1[-1] - arr_D_G_z1[-2]):.2e})", end=' ')
            print(f"D_G_z2: {arr_D_G_z2[-1]:.4f}({(arr_D_G_z2[-1] - arr_D_G_z2[-2]):.2e})", end=' ')
            print()


            mlflow.log_metric("G_loss", arr_G_loss[-1], step=len(ds)*epoch + i)
            mlflow.log_metric("D_loss", arr_D_loss[-1], step=len(ds)*epoch + i)
            mlflow.log_metric("D_x", arr_D_x[-1], step=len(ds)*epoch + i)
            mlflow.log_metric("D_G_z1", arr_D_G_z1[-1], step=len(ds)*epoch + i)
            mlflow.log_metric("D_G_z2", arr_D_G_z2[-1], step=len(ds)*epoch + i)

    return torch.mean(torch.tensor(arr_G_loss)), \
            torch.mean(torch.tensor(arr_D_loss)), \
            torch.mean(torch.tensor(arr_D_x)),    \
            torch.mean(torch.tensor(arr_D_G_z1)), \
            torch.mean(torch.tensor(arr_D_G_z2))



def evaluate(
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    checkpoint_dir: str,
    epoch: int,
):
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        # fake_images = fake_images * STD_TENSOR + MEAN_TENSOR # Unnormalize
        vutils.save_image(fake_images.detach(),
                          os.path.join(checkpoint_dir, f"fake_samples_epoch_{epoch}.png"), normalize=True)


def main(config: dict):
    print("Starting training:")

    experiment_name = config["experiment_name"]
    print("Experiment name:", experiment_name)

    # Set seed for reproducibility
    seed = int(config["seed"]) if "seed" in config else 0
    set_seed(seed)
    print("Setting seed:", seed)

    device = get_device()
    print("Device: ", device)

    checkpoint = config.get("checkpoint_folder", CHECKPOINT_PATH)
    checkpoint = os.path.abspath(checkpoint)
    os.makedirs(checkpoint, exist_ok=True)
    print("Checkpoint folder:", checkpoint)

    # Save config to checkpoint folder
    with open(os.path.join(checkpoint, "config.json"), "w") as f:
        f.write(json.dumps(config))

    image_size = int(config["image_size"]) if "image_size" in config else 64
    batch_size = int(config["batch_size"]) if "batch_size" in config else 256
    print("Batch size:", batch_size)
    data_path = os.path.abspath(config["data_path"]) if "data_path" in config else DATASET_PATH
    print("Data path:", data_path)
    dataset = CatDataset(data_path, transform=T.Compose([
        # T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
        # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomHorizontalFlip(),
        T.Resize(int(image_size * 1.15)),
        T.RandomCrop(image_size),
        T.ConvertImageDtype(torch.float),
        # T.Normalize(mean=MEAN, std=STD),
    ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model_params = config["model_params"]
    discriminator = Discriminator(**model_params["discriminator"]).to(device)
    generator = Generator(**model_params["generator"]).to(device)
    print("Model params:", config["model_params"])

    optimizer_params = config["optimizer_params"]
    d_optimizer_params = optimizer_params["discriminator"]
    opt_d: torch.optim.Optimizer = OPTIMIZERS[d_optimizer_params["name"]](params=discriminator.parameters(),
                                                **d_optimizer_params["params"])
    g_optimizer_params = optimizer_params["generator"]
    opt_g: torch.optim.Optimizer = OPTIMIZERS[g_optimizer_params["name"]](params=generator.parameters(),
                                                **g_optimizer_params["params"])
    print("Optimizer for discriminator:", opt_d)
    print("Optimizer for generator:", opt_g)

    scheduler_params = config["scheduler_params"]
    d_scheduler_params = scheduler_params.get("discriminator", {})
    scheduler_d = SCHEDULERS[d_scheduler_params["name"]](opt_d, **d_scheduler_params["params"]) \
        if "discriminator" in scheduler_params else None
    g_scheduler_params = scheduler_params.get("generator", {})
    scheduler_g = SCHEDULERS[g_scheduler_params["name"]](opt_g, **g_scheduler_params["params"]) \
        if "generator" in scheduler_params else None
    print("Discriminator scheduler:", scheduler_d)
    print("Generator scheduler:", scheduler_g)

    criterion = nn.BCELoss()
    print("Loss function:", criterion)

    epochs = int(config["epochs"])
    print("Number of epochs: ", epochs)
    warmup_epochs = int(config["warmup_epochs"])
    print("Warmup epochs:", warmup_epochs)

    min_delta = float(config["early_stopping"]["min_delta"]) if "early_stopping" in config else 0.0
    print("Min delta:", min_delta)
    patience = int(config["early_stopping"]["patience"]) if "early_stopping" in config else 0
    print("Patience:", patience)

    best_models = {
        "discriminator": discriminator.state_dict(),
        "generator": generator.state_dict()
    }
    best_loss = {
        "discriminator": float("inf"),
        "generator": float("inf")
    }
    best_loss_epoch = 0
    previous = {
        "mean_G_loss": float("inf"),
        "mean_D_loss": float("inf"),
        "mean_D_x": float("inf"),
        "mean_D_G_z1": float("inf"),
        "mean_D_G_z2": float("inf")
    }

    fixed_noise = torch.randn(25, generator.nz, 1, 1).cuda()

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:3113")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_params", config["model_params"])
        mlflow.log_param("optimizer_discriminator", d_optimizer_params)
        mlflow.log_param("optimizer_generator", g_optimizer_params)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("scheduler_discriminator", d_scheduler_params)
        mlflow.log_param("scheduler_generator", g_scheduler_params)
        mlflow.log_param("seed", seed)
        mlflow.log_params(config["model_params"])

        for epoch in range(epochs):
            print("Epoch", epoch)

            print("Processing training")
            discriminator.train()
            generator.train()
            mean_G_loss, mean_D_loss, mean_D_x, mean_D_G_z1, mean_D_G_z2 = train(discriminator,
                                                        generator,
                                                        loader,
                                                        opt_d,
                                                        opt_g,
                                                        criterion,
                                                        epoch=epoch,
                                                        num_epochs=epochs)

            print("Processing validation")
            # discriminator.eval()
            generator.eval()
            evaluate(generator, fixed_noise, checkpoint, epoch)

            mlflow.log_metric("mean_G_loss", mean_G_loss, step=epoch)
            mlflow.log_metric("mean_D_loss", mean_D_loss, step=epoch)
            mlflow.log_metric("mean_D_x", mean_D_x, step=epoch)
            mlflow.log_metric("mean_D_G_z1", mean_D_G_z1, step=epoch)
            mlflow.log_metric("mean_D_G_z2", mean_D_G_z2, step=epoch)

            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("discriminator_learning_rate", scheduler_d.get_last_lr()[0])
            mlflow.log_metric("generator_learning_rate", scheduler_g.get_last_lr()[0])

            scheduler_d.step()
            scheduler_g.step()


            # # Saving checkpoint
            # if epoch >= warmup_epochs and min_delta < best_loss["discriminator"] - mean_D_loss:
            #     print("Saving best models")
            #     best_models["discriminator"] = discriminator.state_dict()
            #     artifact_path = f"discriminator_checkpoint_epoch_{epoch}"
            #     mlflow.pytorch.log_model(discriminator, artifact_path=artifact_path)

            #     best_models["generator"] = generator.state_dict()
            #     artifact_path = f"generator_checkpoint_epoch_{epoch}"
            #     mlflow.pytorch.log_model(generator, artifact_path=artifact_path)

            # # Early stopping
            # if min_delta < best_loss["discriminator"] - mean_D_loss:
            #     best_loss_epoch = epoch
            #     best_loss["discriminator"] = mean_D_loss
            #     best_loss["generator"] = mean_G_loss
            # elif epoch - best_loss_epoch >= patience:
            #     print("Early stopping!")
            #     break


        X = next(iter(loader))
        X = X.cuda()
        signature_d = mlflow.models.infer_signature(X.detach().cpu().numpy(), discriminator(X).detach().cpu().numpy())
        artifact_path_d = f"d_final_epoch_{epoch}"
        # discriminator.load_state_dict(best_models["discriminator"])
        mlflow.pytorch.log_model(discriminator, artifact_path_d, signature=signature_d)

        X = torch.randn(batch_size, generator.nz, 1, 1).cuda()
        artifact_path_g = f"g_final_epoch_{epoch}"
        # generator.load_state_dict(best_models["generator"])
        signature_g = mlflow.models.infer_signature(X.detach().cpu().numpy(), generator(X).detach().cpu().numpy())
        mlflow.pytorch.log_model(generator, artifact_path_g, signature=signature_g)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train a DCGAN model")
    parser.add_argument('--config', type=str,
                        default=os.path.abspath(os.path.join(__file__, "..", "config.json")),
                        help='Path to the configuration file')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    main(config)