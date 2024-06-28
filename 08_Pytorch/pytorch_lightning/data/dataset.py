import torch
from torchvision import datasets
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

class DataModule(LightningDataModule):
    def __init__(self, 
                 data_dir="path/to/dir", 
                 batch_size=32, 
                 num_workers=4,
                 transform=transforms.ToTensor()):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def prepare_data(self):
        ## 데이터 다운로드(준비)
        datasets.MNIST(root=self.data_dir, download=True, train=True, transform=None)
        datasets.MNIST(root=self.data_dir, download=True, train=False, transform=None)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_dataset = datasets.MNIST(root=self.data_dir, download=True, train=True, transform=self.transform)
            self.train_dataset, self.valid_dataset = random_split(full_train_dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage == 'predict':
            self.test_dataset = datasets.MNIST(root=self.data_dir, download=True, train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
