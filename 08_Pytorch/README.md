# 1.Installation

[https://pytorch.kr/get-started/locally/](https://pytorch.kr/get-started/locally/)

    ## 아나콘다 가상환경, 파이토치 설치.
    conda create --name DL python=3.8
    conda install pytorch==2.0.1 torchvision==0.15.2 torchtext==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

    conda create --name DL python=3.9
    conda install pytorch torchvision torchaudio torchtext=0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia

    conda install pytorch-lightning==2.0.0 hydra-core==1.3.2 -c conda-forge
    conda install tensorboard tensorboardX

    ## 주피터 노트북 커널
    conda install pexpect jupyter
    pip install pexpect jupyter ipykernel
    pip uninstall pyzmq
    pip install pyzmq

    ## 라이브러리 설치
    pip install timm transformers

설치가 정상적으로 되었는지 검사

```python
import torch

torch.__version__ ## '2.3.1'
torch.cuda.is_available() ## True
```

# 2.Mission

## 2-1.Pytorch의 Dataset, DataLoader 클래스에 대해 정리하기.

[https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

- Dataset 클래스는 사용자의 데이터셋에 대한 custom class를 정의할 때 상속받아야 하는 부모 클래스.
- 상속 받는 사용자 정의 클래스는 반드시 `__init__`, `__len__`, `__getitem__` 3가지 메서드를 정의해야한다.
- `__init__` : 데이터가 존재하는 경로를 명시 또는 데이터 파일을 로드하며 데이터에 적용될 transform(augmentation) 기능도 클래스 초기화시 명시해야한다.
- `__len__` : 데이터셋에 포함된 데이터의 총량을 반환하도록 정의.
- `__getitem__` : 데이터셋에서 `idx`번째 데이터 샘플 하나를 메모리에 로드하고, 관련 label을 반환하도록 하는 로직을 정의하고 학습에 반영될 데이터를 반환한다.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

- DataLoader 클래스는 객체를 생성할 때 사용자가 정의한 데이터셋의 객체를 입력으로 받는다.
- 앞서 Dataset 클래스의 ``___getitem__` 메서드가 하나씩 데이터를 반환하는데, DataLoader는 batch_size만큼을 하나의 batch로 묶에서 공급하는 역할.

## 2.Dataset, Dataloader 클래스의 내부를 읽고 정리하기.

[https://pytorch.org/docs/stable/data.html](https://pytorch.org/docs/stable/data.html)

### 2-1.Map Style Datasets

`Dataset`을 상속 받는 사용자 정의 클래스는 크게 두 가지 방식으로 구현이 가능하다. 앞서 봤던 방식은 **Map-Style Datasets** 로 index에 맵핑되는 데이터를 가져오는 방식을 말한다.

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 데이터셋 인스턴스 생성
data = torch.randn(100, 3, 224, 224)  # 예시 데이터 (이미지 100개)
labels = torch.randint(0, 10, (100,))  # 예시 레이블
dataset = CustomDataset(data, labels)

# DataLoader 생성
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

### 2-2.Iterable Style Datasets

또 다른 방식에는 **Iterable-style datasets** 가 있는데, 이는 `__iter__` 메서드를 구현하고 데이터 샘플에 대한 반복자(iterator)를 나타내는 데이터셋이다.

- 데이터베이스나 원격 서버로부터 데이터를 순차적으로 읽어야 하는 경우나 실시간으로 생성되는 로그 데이터나 센서 데이터 등을 활용할 때 유용하다.
- 또한 매우 큰 데이터셋을 다룰 때, 전체 데이터를 메모리에 로드하지 않고 순차적으로 처리할 수 있다.
- 이 방식은 청크 읽기(chunk-reading)와 동적 배치 크기(dynamic batch size)를 구현하는 데 용이.
- `__iter__` 메서드를 구현하여 데이터 샘플을 순차적으로 반환할 수 있는 반복자를 생성한다.
- 이를 통해 데이터셋을 반복(iterate)할 때마다 새로운 데이터를 읽어올 수 있다.

```python
import time
import torch
from torch.utils.data import IterableDataset

class RealTimeLogsDataset(IterableDataset):
    def __init__(self, log_source):
        self.log_source = log_source

    def __iter__(self):
        while True:
            log = self.log_source.get_new_log()
            if log is None:
                time.sleep(1)
                continue
            yield log

class LogSource:
    def __init__(self):
        self.logs = []

    def add_log(self, log):
        self.logs.append(log)

    def get_new_log(self):
        if self.logs:
            return self.logs.pop(0)
        else:
            return None

log_source = LogSource()
dataset = RealTimeLogsDataset(log_source)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

주의할 점은 IterableDataset을 멀티프로세스 데이터 로딩과 함께 사용할 때, 각 워커 프로세스는 동일한 데이터셋 객체의 복제본을 갖게 된다는 것이다. 따라서, 중복 데이터를 피하기 위해 각 복제본을 다르게 구성해야한다.

```python
import torch
from torch.utils.data import IterableDataset, DataLoader
import itertools

class MultiProcessIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 단일 워커
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.data)
        # 다중 워커
        else:
            per_worker = int(math.ceil(len(self.data) / float(worker_info.num_workers))) # 각 워커가 처리할 데이터의 크기를 계산.
            worker_id = worker_info.id # 현재 워커의 ID

            # 워커가 처리할 데이터의 시작 인덱스(iter_start)와 종료 인덱스(iter_end)를 계산
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data))

        return iter(self.data[iter_start:iter_end])

data = range(100)
dataset = MultiProcessIterableDataset(data)
dataloader = DataLoader(dataset, batch_size=10, num_workers=4)
```

### 2-3.Sampler

Sampler는 사용자 Dataset 객체가 데이터를 반환(`___getitem__`)하는 idx를 어떻게 설정할 것인가를 정의하는 것으로, DataLoader 객체에 적용된다.

```python
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

# 예제 데이터셋
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = [i for i in range(100)]
dataset = ExampleDataset(data)
custom_sampler = CustomSampler(dataset)
dataloader = DataLoader(dataset, sampler=custom_sampler, batch_size=10)

for batch in dataloader:
    print(batch)


class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

# 예제 데이터셋 및 배치 샘플러 사용
batch_sampler = CustomBatchSampler(custom_sampler, batch_size=10)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

for batch in dataloader:
    print(batch)
```

### 2-4.자동 배치

일반적으로 DataLoader 클래스를 선언할 때 `batch_size`가 None이 아니므로 자동적으로 미니 배치를 생성하게 된다.  
반면에 데이터베이스에서 대량으로 읽어들이거나 연속된 메모리 청크를 읽는 것이 더 저렴할 수 있는데 이러한 경우 자동 배치를 비활성화하고 데이터 로더가 데이터셋의 각 멤버를 직접 반환하도록 할 수 있다.  
batch_size와 batch_sampler가 모두 None일 때, 자동 배치가 비활성화된다.

```python
dataloader = DataLoader(dataset, batch_size=None)

for sample in dataloader:
    print(sample)
```

### 2-5.collate_fn

collate_fn은 자동 배치가 활성화된 경우와 비활성화된 경우에 사용 방식이 다르다.

- 자동 배치가 비활성화된 경우: collate_fn은 각 개별 데이터 샘플에 대해 호출되며, 출력은 데이터 로더 반복자에서 반환된다. 기본 collate_fn은 NumPy 배열을 PyTorch 텐서로 변환한다.
- 자동 배치가 활성화된 경우: collate_fn은 각 시간에 데이터 샘플 목록과 함께 호출된다. 입력 샘플을 배치로 묶어서 데이터 로더 반복자에서 반환.

### 2-6.num_workers

파이토치의 DataLoader는 기본적으로 단일 프로세스 데이터를 로딩하지만 num_workers 인자를 양수로 설정하면 멀티 프로세스 데이터 로딩을 수행할 수 있다.

```python
import torch
from torch.utils.data import DataLoader, Dataset

class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    return torch.stack(batch, dim=0)

if __name__ == '__main__':
    data = [torch.tensor([i]) for i in range(100)]
    dataset = ExampleDataset(data)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)

```

멀티 프로세스 모드에서는 DataLoader의 반복자(iterator)가 생성될 때마다 워커 프로세스가 생성된다.  
각 워커는 데이터셋, collate_fn, worker_init_fn을 사용하여 데이터를 초기화하고 fetching하며 데이터셋 접근, 내부 I/O, 변환(transform) 작업(collate_fn 포함)이 워커 프로세스에서 실행된다.

### 2-7.memory pinning

DataLoader에 `pin_memory=True`를 전달하면 가져온 데이터 텐서를 자동으로 고정된 메모리에 배치한다.
