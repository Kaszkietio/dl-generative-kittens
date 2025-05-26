import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from data_utils import DATASET_PATH, CatDataset, MEAN, STD

def train():
    # Parameters
    image_size = 64
    batch_size = 128
    nz = 100  # Size of z latent vector
    ngf = 64  # Generator feature maps
    ndf = 64  # Discriminator feature maps
    num_epochs = 25
    lr = 0.0002
    beta1 = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = DATASET_PATH  # Change to your dataset path

    # Data loader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    dataset = ImageFolder(data_root+"/..", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)

    # Discriminator
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # Initialize models
    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            ## Update D
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, labels)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            labels.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            optimizerD.step()

            ## Update G
            netG.zero_grad()
            labels.fill_(real_label)  # Trick D
            output = netD(fake_images).view(-1)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD_real + errD_fake:.4f} Loss_G: {errG:.4f}")

        # Save progress
        vutils.save_image(fake_images.detach(), f"fake_samples_epoch_{epoch}.png", normalize=True)

    # Save models
    torch.save(netG.state_dict(), "dcgan_generator.pth")
    torch.save(netD.state_dict(), "dcgan_discriminator.pth")

if __name__ == "__main__":
    train()
    print("Training complete. Models saved as 'dcgan_generator.pth' and 'dcgan_discriminator.pth'.")
