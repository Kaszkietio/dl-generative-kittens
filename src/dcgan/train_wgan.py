import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
import torchvision.models
from torchvision.transforms import v2 as T
from torchvision import utils as vutils

from wgan import Generator, Critic, compute_gradient_penalty, calculate_fid
from utils import set_seed, get_device
from data_utils import DATASET_PATH, CHECKPOINT_PATH, CatDataset, MEAN, STD

MEAN_TENSOR = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1).cuda()
STD_TENSOR = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1).cuda()

OPTIMIZERS = {
    "AdamW": AdamW
}

SCHEDULERS = {
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ReduceLROnPlateau": ReduceLROnPlateau
}

def train(
    C: Critic,
    G: Generator,
    ds: DataLoader,
    optimizer_C: torch.optim.Optimizer,
    optimizer_G: torch.optim.Optimizer,
    n_critic: int,
    lambda_gp: float,
    epoch: int = 0,
    num_epochs: int = 25
):
    arr_C_loss = [float("inf")]
    arr_G_loss = [float("inf")]

    for i, real_imgs in enumerate(ds):
        real_imgs = real_imgs.cuda()
        batch_size_curr = real_imgs.size(0)

        # Train Critic
        for _ in range(n_critic):
            z = torch.randn(batch_size_curr, G.nz, 1, 1).cuda()
            fake_imgs = G(z).detach()
            real_validity = C(real_imgs)
            fake_validity = C(fake_imgs)
            gp = compute_gradient_penalty(C, real_imgs.data, fake_imgs.data)
            loss_C = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

        # Train Generator
        z = torch.randn(batch_size_curr, G.nz, 1, 1).cuda()
        gen_imgs = G(z)
        loss_G = -torch.mean(C(gen_imgs))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        arr_G_loss.append(loss_G.item())
        arr_C_loss.append(loss_C.item())

        if i % 50 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(ds)}]", end=' ')
            print(f"G_loss: {arr_G_loss[-1]:.4f}({(arr_G_loss[-1] - arr_G_loss[-2]):.2e})", end=' ')
            print(f"C_loss: {arr_C_loss[-1]:.4f}({(arr_C_loss[-1] - arr_C_loss[-2]):.2e})", end=' ')
            print()


            mlflow.log_metric("G_loss", arr_G_loss[-1], step=len(ds)*epoch + i)
            mlflow.log_metric("C_loss", arr_C_loss[-1], step=len(ds)*epoch + i)

    return torch.mean(torch.tensor(arr_G_loss)), \
            torch.mean(torch.tensor(arr_C_loss))



def evaluate(
    G: Generator,
    resize: nn.Module,
    feature_extractor: nn.Module,
    fixed_noise: torch.Tensor,
    real_images: torch.Tensor,
    checkpoint_dir: str,
    epoch: int,
):
    with torch.no_grad():
        fake_images = G(fixed_noise)
        vutils.save_image(fake_images.detach(),
                          os.path.join(checkpoint_dir, f"fake_samples_epoch_{epoch}.png"), normalize=True)

        # Calculate FID
        fake_images = resize(fake_images)
        real_images = resize(real_images).cuda()
        fid = calculate_fid(real_images, fake_images, feature_extractor)
        print(f"FID: {fid:.4f}")
        mlflow.log_metric("FID", fid)



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
        T.Resize(image_size),
        T.ConvertImageDtype(torch.float),
    ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model_params = config["model_params"]
    C = Critic(**model_params["critic"]).to(device)
    C.train()

    G = Generator(**model_params["generator"]).to(device)
    print("Model params:", config["model_params"])

    optimizer_params = config["optimizer_params"]
    c_optimizer_params = optimizer_params["critic"]
    optimizer_C: torch.optim.Optimizer = OPTIMIZERS[c_optimizer_params["name"]](params=C.parameters(),
                                                **c_optimizer_params["params"])
    g_optimizer_params = optimizer_params["generator"]
    optimizer_G: torch.optim.Optimizer = OPTIMIZERS[g_optimizer_params["name"]](params=G.parameters(),
                                                **g_optimizer_params["params"])
    print("Optimizer for critic:", optimizer_C)
    print("Optimizer for generator:", optimizer_G)

    scheduler_params = config["scheduler_params"]
    c_scheduler_params = scheduler_params.get("critic", {})
    scheduler_c = SCHEDULERS[c_scheduler_params["name"]](optimizer_C, **c_scheduler_params["params"]) \
        if "critic" in scheduler_params else None
    g_scheduler_params = scheduler_params.get("generator", {})
    scheduler_g = SCHEDULERS[g_scheduler_params["name"]](optimizer_G, **g_scheduler_params["params"]) \
        if "generator" in scheduler_params else None
    print("critic scheduler:", scheduler_c)
    print("Generator scheduler:", scheduler_g)

    epochs = int(config["epochs"])
    print("Number of epochs: ", epochs)
    warmup_epochs = int(config["warmup_epochs"])
    print("Warmup epochs:", warmup_epochs)

    min_delta = float(config["early_stopping"]["min_delta"]) if "early_stopping" in config else 0.0
    print("Min delta:", min_delta)
    patience = int(config["early_stopping"]["patience"]) if "early_stopping" in config else 0
    print("Patience:", patience)

    n_critic = int(config["n_critic"]) if "n_critic" in config else 5
    print("Number of critic updates per generator update:", n_critic)
    lambda_gp = float(config["lambda_gp"]) if "lambda_gp" in config else 10.0
    print("Lambda for gradient penalty:", lambda_gp)

    feature_extractor = torchvision.models.inception_v3(pretrained=True).cuda()
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval()
    resize = T.Resize((299, 299))
    fixed_noise = torch.randn(25, G.nz, 1, 1).cuda()

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:3113")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("model_params", config["model_params"])
        mlflow.log_param("optimizer_critic", c_optimizer_params)
        mlflow.log_param("optimizer_generator", g_optimizer_params)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("scheduler_critic", c_scheduler_params)
        mlflow.log_param("scheduler_generator", g_scheduler_params)
        mlflow.log_param("seed", seed)
        mlflow.log_params(config["model_params"])

        for epoch in range(epochs):
            print("Epoch", epoch)

            print("Processing training")
            G.train()
            mean_G_loss, mean_C_loss = train(C,
                                            G,
                                            loader,
                                            optimizer_C,
                                            optimizer_G,
                                            n_critic,
                                            lambda_gp,
                                            epoch=epoch,
                                            num_epochs=epochs)

            print("Processing validation")
            G.eval()
            evaluate(G,
                     resize,
                     feature_extractor,
                     fixed_noise,
                     next(iter(loader)),
                     checkpoint,
                     epoch)

            mlflow.log_metric("mean_G_loss", mean_G_loss, step=epoch)
            mlflow.log_metric("mean_C_loss", mean_C_loss, step=epoch)

            mlflow.log_metric("epoch", epoch)
            mlflow.log_metric("critic_learning_rate", scheduler_c.get_last_lr()[0])
            mlflow.log_metric("generator_learning_rate", scheduler_g.get_last_lr()[0])

            scheduler_c.step()
            scheduler_g.step()


        X = next(iter(loader))
        X = X.cuda()
        signature_c = mlflow.models.infer_signature(X.detach().cpu().numpy(), C(X).detach().cpu().numpy())
        artifact_path_c = f"c_final_epoch_{epoch}"
        mlflow.pytorch.log_model(C, artifact_path_c, signature=signature_c)

        X = torch.randn(batch_size, G.nz, 1, 1).cuda()
        artifact_path_g = f"g_final_epoch_{epoch}"
        signature_g = mlflow.models.infer_signature(X.detach().cpu().numpy(), G(X).detach().cpu().numpy())
        mlflow.pytorch.log_model(G, artifact_path_g, signature=signature_g)


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