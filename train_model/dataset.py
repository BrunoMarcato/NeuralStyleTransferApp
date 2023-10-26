import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform != None:
            augs = self.transform(image = image)
            image = augs['image']

        return image

def train_dataloader(train_image_dir, batch_size = 1, num_workers = 1, pin_memory = True, train_transforms = None):
    train_dataset = TrainDataset(
      img_dir = train_image_dir,
      transform = train_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    return train_loader