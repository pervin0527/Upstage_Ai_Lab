import cv2
import torch
import random
import pandas as pd

from torch.nn import functional as F
from torch.utils.data import Dataset
from data.augmentation import mixup, albumentation_transform, augraphy_transform, batch_transform

class DocTypeDataset(Dataset):
    def __init__(self, img_path, csv_path, meta_path, img_h, img_w, one_hot=False):
        self.one_hot = one_hot
        self.img_path = img_path
        
        if "train" in csv_path.split('/')[-1]:
            print("Train dataset Loaded")
            self.is_train = True
        else:
            print("Valid dataset Loaded")
            self.is_train = False

        self.df = pd.read_csv(csv_path)
        meta_df = pd.read_csv(meta_path)
        
        self.classes = list(meta_df['class_name'].unique())
        self.num_classes = len(self.classes)

        self.alb_transform = albumentation_transform(img_h, img_w)
        self.aup_transform = augraphy_transform()
        self.transform = batch_transform(img_h, img_w)

        ## 자주 틀리는 클래스들만 선별해서 학습
        # target_classes = [3, 4, 7, 10, 11, 12, 13, 14]
        # self.df = self.df[self.df['target'].isin(target_classes)] ## 리스트에 해당하는 target인 행들만 선별
        # target_map_dict = dict()
        # for i, c in enumerate(target_classes):
        #     target_map_dict[c] = i
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

        if self.is_train:
            prob = random.random()
            if prob < 0.4:
                image = self.aup_transform(image)
                image = self.alb_transform(image=image)['image']
            elif 0.4 <= prob < 0.6:
                image = self.aup_transform(image)
            elif 0.6 <= prob < 0.8:
                image = self.alb_transform(image=image)['image']
            
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
    

class TransformerDataset(Dataset):
    def __init__(self, img_path, csv_path, meta_path, img_h, img_w, one_hot=False, processor=None):
        self.one_hot = one_hot
        self.img_path = img_path
        self.processor = processor
        
        if "train" in csv_path.split('/')[-1]:
            print("Train dataset Loaded")
            self.is_train = True
        else:
            print("Valid dataset Loaded")
            self.is_train = False

        self.df = pd.read_csv(csv_path)
        meta_df = pd.read_csv(meta_path)
        
        self.classes = list(meta_df['class_name'].unique())
        self.num_classes = len(self.classes)

        self.alb_transform = albumentation_transform(img_h, img_w)
        self.aup_transform = augraphy_transform()
        self.transform = batch_transform(img_h, img_w)

        ## 자주 틀리는 클래스들만 선별해서 학습
        # target_classes = [3, 4, 7, 10, 11, 12, 13, 14]
        # self.df = self.df[self.df['target'].isin(target_classes)] ## 리스트에 해당하는 target인 행들만 선별
        # target_map_dict = dict()
        # for i, c in enumerate(target_classes):
        #     target_map_dict[c] = i
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

        if self.is_train:
            prob = random.random()
            if prob < 0.4:
                image = self.aup_transform(image)
                image = self.alb_transform(image=image)['image']
            elif 0.4 <= prob < 0.6:
                image = self.aup_transform(image)
            elif 0.6 <= prob < 0.8:
                image = self.alb_transform(image=image)['image']
            
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

        x = self.processor(images=image, return_tensors="pt")['pixel_values']

        return id, x.squeeze(0), y