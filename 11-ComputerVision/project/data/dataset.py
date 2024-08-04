import cv2
import torch
import random
import numpy as np
import pandas as pd

from torch.nn import functional as F
from torch.utils.data import Dataset
from data.augmentation import mixup, cutout

class DocTypeDataset(Dataset):
    def __init__(self, img_path, csv_path, meta_path, transform, one_hot=False):
        self.one_hot = one_hot
        self.img_path = img_path
        self.transform = transform
        
        self.df = pd.read_csv(csv_path)
        meta_df = pd.read_csv(meta_path)
        
        self.classes = list(meta_df['class_name'].unique())
        self.num_classes = len(self.classes)

        # ## 자주 틀리는 클래스들만 선별해서 학습
        # target_classes = [3, 4, 7, 10, 11, 12, 13, 14]
        # self.df = self.df[self.df['target'].isin(target_classes)] ## 리스트에 해당하는 target인 행들만 선별
        # target_map_dict = {(idx, label) for idx, label in enumerate(target_classes)}
        # self.df['target'] = self.df['target'].map(target_map_dict)

        # meta_df = meta_df[meta_df['target'].isin(target_classes)]
        # self.classes = list(meta_df['class_name'].unique())
        # self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        id = self.df.iloc[idx, 0]
        target = self.df.iloc[idx, 1]

        img_path = f"{self.img_path}/{id}"
        image = cv2.imread(img_path)

        if self.one_hot:
            y = F.one_hot(torch.tensor(target), num_classes=self.num_classes).float()
            if random.random() > 0.5:
                rand_idx = random.randint(0, len(self.df)-1)
                bg_file_name = self.df.iloc[rand_idx, 0]
                bg_target = self.df.iloc[rand_idx, 1]
                bg_target = F.one_hot(torch.tensor(bg_target), num_classes=self.num_classes).float()

                bg_img = cv2.imread(f"{self.img_path}/{bg_file_name}")
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

                img_h, img_w = image.shape[:2]
                bg_img = cv2.resize(bg_img, (img_w, img_h))

                image, target = mixup(image, bg_img, target, bg_target, alpha=0.5)
        else:
            y = target

        x = self.transform(image=image)['image']

        return id, x, y