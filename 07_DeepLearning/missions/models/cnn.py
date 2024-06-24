import torch

from torch import nn
from torch.nn import functional as F

class CNN_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

class CNN_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

class CNN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool1 = nn.AdaptiveMaxPool2d(output_size=(14, 14))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.adaptive_pool2 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.adaptive_pool2 = nn.AdaptiveMaxPool2d(output_size=(7, 7))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.adaptive_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.adaptive_pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    
class CNN_V4(nn.Module):
    def __init__(self, drop_prob=0.3, weight_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.relu = nn.ReLU()

        if weight_init:
            self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class CNN_V5(nn.Module):
    def __init__(self, drop_prob=0.3, weight_init=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # (N, 1, 28, 28) -> (N, 64, 28, 28)
        self.bn1 = nn.BatchNorm2d(64)  # (N, 64, 28, 28)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # (N, 64, 28, 28) -> (N, 128, 28, 28)
        self.bn2 = nn.BatchNorm2d(128)  # (N, 128, 28, 28)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # (N, 128, 28, 28) -> (N, 256, 28, 28)
        self.bn3 = nn.BatchNorm2d(256)  # (N, 256, 28, 28)
        
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))  # (N, 256, 28, 28) -> (N, 256, 7, 7)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (N, 256, 7, 7) -> (N, 256, 1, 1)

        self.dropout = nn.Dropout(p=drop_prob)  # (N, 256, 1, 1)
        
        self.conv_out = nn.Conv2d(256, 10, kernel_size=1, stride=1)  # (N, 256, 1, 1) -> (N, 10, 1, 1)

        self.relu = nn.ReLU()  # (N, 10, 1, 1)

        if weight_init:
            self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (N, 1, 28, 28) -> (N, 64, 28, 28)
        x = self.relu(self.bn2(self.conv2(x)))  # (N, 64, 28, 28) -> (N, 128, 28, 28)
        x = self.relu(self.bn3(self.conv3(x)))  # (N, 128, 28, 28) -> (N, 256, 28, 28)
        x = self.adaptive_pool(x)  # (N, 256, 28, 28) -> (N, 256, 7, 7)
        
        x = self.conv_out(x)  # (N, 256, 7, 7) -> (N, 10, 7, 7)
        x = self.global_avg_pool(x)  # (N, 10, 7, 7) -> (N, 10, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 10, 1, 1) -> (N, 10)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
