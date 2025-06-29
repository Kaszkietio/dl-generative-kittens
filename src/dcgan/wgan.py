import numpy as np
from scipy import linalg
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int):
        """
        Generator for WGAN.
        Args:
            nz (int): Size of the latent vector z.
            ngf (int): Number of generator feature maps.
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 2, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

class Critic(nn.Module):
    def __init__(self, ndf: int):
        """
        Critic for WGAN.
        Args:
            ndf (int): Number of discriminator feature maps.
        """
        super(Critic, self).__init__()
        self.ndf = ndf
        self.main = self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )

        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(C: Critic, real_samples: torch.Tensor, fake_samples: torch.Tensor):
    """
    Computes the gradient penalty for WGAN.
    Args:
        C (Critic): The critic model.
        real_samples (Tensor): Real samples from the dataset.
        fake_samples (Tensor): Fake samples generated by the generator.
    Returns:
        Tensor: The computed gradient penalty.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = C(interpolates)
    fake = torch.ones_like(d_interpolates).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gradients_penalty

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
