
import os
from PIL import Image
from torch.utils.data import Dataset

class CleanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            for img in os.listdir(class_path):
                if img.lower().endswith(('.jpg', '.png')) and 'distortion' not in img.lower():
                    self.image_paths.append(os.path.join(class_path, img))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class DistortedDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for class_name in os.listdir(root_dir):
            distorted_path = os.path.join(root_dir, class_name, "distortion")
            if not os.path.isdir(distorted_path):
                continue
            for img in os.listdir(distorted_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(distorted_path, img))
                    self.labels.append(class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
