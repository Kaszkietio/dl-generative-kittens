import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io.image import read_image
from torchvision import transforms as T

DATASET_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data", "cats", "Data"))
CHECKPOINT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "checkpoints"))

MEAN =  (0.4837435185909271, 0.4360336363315582, 0.38709205389022827)
STD =   (0.15184095, 0.14043211, 0.1453041)

class CatDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.transform = transform if transform is not None else T.Compose([
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=MEAN, std=STD)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = read_image(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

class MeanStdAggregator:
    def __init__(self):
        self.count = 0
        self.mean = torch.zeros(3)
        self.M2 = torch.zeros(3)

    def update(self, new_value: torch.Tensor):
        self.count += 1
        new_value = new_value.mean((1, 2))
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.count < 2:
            return float("nan")
        else:
            mean = self.mean
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return (mean, variance, sample_variance)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Calculate dataset mean and std")
    parser.add_argument('--data_dir', required=True, type=str, help='Path to the dataset directory')
    args = parser.parse_args()

    # Load the dataset
    data_pir = args.data_dir
    dataset = CatDataset(data_pir, transform=T.Compose([
        T.ConvertImageDtype(torch.float)
    ]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    agg = MeanStdAggregator()

    print("Calculating mean and std...")
    for img in tqdm(dataset):
        agg.update(img)
    mean, variance, sample_variance = agg.finalize()

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {sample_variance.tolist()}")