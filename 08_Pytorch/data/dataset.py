import torch
import random
import numpy as np

from torch.utils.data import Subset
from torch.utils.data import Dataset

class AlbumentationsDataset(Dataset):
    def __init__(self, image_dataset, labeled_indices, unlabeled_indices, transform=None):
        self.image_dataset = image_dataset
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.transform = transform
        self.all_indices = np.concatenate((labeled_indices, unlabeled_indices))

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        real_idx = self.all_indices[idx]
        image, label = self.image_dataset[real_idx]
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        is_labeled = real_idx in self.labeled_indices

        return image, label if is_labeled else -1, is_labeled