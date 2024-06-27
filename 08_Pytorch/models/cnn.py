from torch import nn

class CNN_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 625, bias=True)
        self.fc2 = nn.Linear(625, 10, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool1(x)

        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = self.relu(self.conv3(x))
        x = self.max_pool3(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x