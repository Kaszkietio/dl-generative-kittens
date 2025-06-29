import os
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import numpy as np
from scipy import linalg
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from data_utils import CatDataset, DATASET_PATH

@torch.no_grad()
def calculate_fid(real_images: torch.Tensor, fake_images: torch.Tensor, feature_extractor: nn.Module):
    """
    Calculate the FID score between real and fake images.
    Args:
        real_images (Tensor): Real images from the dataset.
        fake_images (Tensor): Fake images generated by the generator.
        feature_extractor (nn.Module): Pre-trained model to extract features.
    Returns:
        float: The FID score.
    """
    def get_activations(images):
        if images.shape[1] != 3:
            images = images.repeat(1, 3, 1, 1)
        images = images.cuda()
        features: torch.Tensor = feature_extractor(images)
        return features.view(images.size(0), -1)

    real_acts = get_activations(real_images).cpu().numpy()
    gen_acts = get_activations(fake_images).cpu().numpy()

    mu_real, sigma_real = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_acts, axis=0), np.cov(gen_acts, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


@torch.no_grad()
def calculate_fid_full_dataset(
        loader: torch.utils.data.DataLoader,
        generator: nn.Module,
        resize: T.Resize,
        feature_extractor: nn.Module):
    real_features = []
    fake_features = []
    print("Calculating FID for the full dataset...")
    feature_extractor = feature_extractor.eval().cuda()
    for images in tqdm(loader):
        if isinstance(images, tuple):
            images = images[0]
        if images.shape[1] != 3:
            images = images.repeat(1, 3, 1, 1)
        images = images.cuda()
        features_real: torch.Tensor = feature_extractor(images).cpu().numpy()
        real_features.append(features_real)

        z = torch.rand((len(images), generator.nz, 1, 1)).cuda()
        fake_images = generator(z)  # Placeholder for generated images
        fake_images = resize(fake_images)
        features_fake = feature_extractor(fake_images).cpu().numpy()
        fake_features.append(features_fake)

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    print("Calculating FID score...")
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def main(config: dict):
    mlflow.set_tracking_uri('http://localhost:3113')

    feature_extractor = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT).cuda()
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval()
    resize = T.Resize((299, 299))

    dataset_path = config.get("dataset_path", DATASET_PATH)

    # Load the dataset
    dataset = CatDataset(dataset_path, transform=T.Compose([
        resize,
        T.ConvertImageDtype(torch.float32),
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    metrics = {
        "model_name": [],
        "fid": [],
    }

    for model_cfg in config["models"]:
        model_name = model_cfg["model_name"]
        model_path = model_cfg["model_path"]
        print("Evaluation of model:", model_name)
        metrics["model_name"].append(model_name)

        # Load the model
        if model_path.startswith("mlflow"):
            model = mlflow.pytorch.load_model(model_path).cuda()
        else:
            model_state = torch.load(model_path)
            from dcgan_init import Generator
            model = Generator(nz=100, ngf=64).cuda()
            model.load_state_dict(model_state)


        fid = calculate_fid_full_dataset(
            loader=loader,
            generator=model,
            resize=resize,
            feature_extractor=feature_extractor
        )
        metrics["fid"].append(fid)
        # Save metrics to CSV
        df = pd.DataFrame(metrics)
        output_path = config.get("output_path", ".")
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(os.path.join(config["output_path"], "metrics.csv"), index=False)
        print("FID scores:")
        print(df)



if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")

    config_path = parser.parse_args().config
    with open(config_path, "r") as f:
        config = f.read()

    config = json.loads(config)
    main(config)