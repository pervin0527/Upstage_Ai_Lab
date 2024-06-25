import torch
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
    

def ssl_preprocessing(train_dataset, test_dataset, num_labeled=1, num_unlabeled=0):
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    labels = list(set(y_train))
    print(f"Labels: {labels}")

    indexes_labeled = []
    indexes_unlabaled = []
    for label in labels: ## [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        label_indexes = np.where(y_train == label)[0] ## label인 데이터를 모두 찾음.

        ## labeled data만 샘플링.
        if num_unlabeled == 0:
            ## 중복을 허용하지 않으면서 num_labeled만큼 샘플링.
            sampled_indexes = np.random.choice(label_indexes, size=num_labeled, replace=False)
            indexes_labeled.extend(sampled_indexes)

            ## 전체 인덱스에서 샘플링된 인덱스를 제거한 후 나머지 인덱스를 indexes_unlabaled에 추가
            indexes_unlabaled.extend(np.setdiff1d(label_indexes, sampled_indexes))

        ## 라벨된 데이터와 라벨되지 않은 데이터를 모두 샘플링 (N_unlabeled가 0이 아닌 경우)
        else:
            sampled_indexes = np.random.choice(np.where(y_train == label)[0], size=num_labeled + num_unlabeled, replace=False)
            indexes_labeled.extend(sampled_indexes[:num_labeled])
            indexes_unlabaled.extend(sampled_indexes[num_labeled:])

    data_dict = {}
    data_dict["x_train_labeled"] = torch.tensor(x_train[indexes_labeled, :], dtype=torch.float32).unsqueeze(1) / 255.0
    data_dict["x_train_unlabeled"] = torch.tensor(x_train[indexes_unlabaled, :], dtype=torch.float32).unsqueeze(1) / 255.0

    data_dict["y_train_labeled"] = torch.tensor(y_train[indexes_labeled], dtype=torch.long)
    data_dict["y_train_unlabeled"] = torch.tensor(y_train[indexes_unlabaled], dtype=torch.long)
    
    data_dict["x_test"] = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
    data_dict["y_test"] = torch.tensor(y_test, dtype=torch.long)

    print(f"Labeled Train Shape   : {data_dict['x_train_labeled'].shape}")
    print(f"UnLabeled Train Shape : {data_dict['x_train_unlabeled'].shape}")

    return data_dict