import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Custom loader that skips unreadable/corrupt images
class CleanImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.samples = self._filter_valid_images(self.samples)

    def _filter_valid_images(self, samples):
        valid_samples = []
        for path, label in samples:
            try:
                with Image.open(path) as img:
                    img.verify()  # Check image validity
                valid_samples.append((path, label))
            except Exception as e:
                print(f"🛑 Skipping corrupted file: {path}")
        return valid_samples

# Your data loader
def get_data_loaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CleanImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes
