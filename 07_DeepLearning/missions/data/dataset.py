import numpy as np
from torch.utils.data import Dataset

class AlbumentationsDataset(Dataset):
    def __init__(self, image_dataset, transform=None):
        self.image_dataset = image_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label