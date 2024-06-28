import torch

from torch import nn
from torchmetrics import Accuracy
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

class CNN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.model.num_classes
        self.learning_rate = cfg.model.learning_rate
        self.dropout_ratio = cfg.model.dropout_ratio
        self.use_scheduler = cfg.model.use_scheduler
        self.criterion = instantiate(cfg.criterion)
        self.optimizer = cfg.optimizer
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)  # [BATCH_SIZE, 1, 28, 28] -> [BATCH_SIZE, 16, 24, 24]
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5) # [BATCH_SIZE, 16, 24, 24] -> [BATCH_SIZE, 32, 20, 20]
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # [BATCH_SIZE, 32, 20, 20] -> [BATCH_SIZE, 32, 10, 10]
        self.dropout2 = nn.Dropout(self.dropout_ratio)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5) # [BATCH_SIZE, 32, 10, 10] -> [BATCH_SIZE, 64, 6, 6]
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2) # 크기를 1/2로 줄입니다. [BATCH_SIZE, 64, 6, 6] -> [BATCH_SIZE, 64, 3, 3]
        self.dropout3 = nn.Dropout(self.dropout_ratio)

        self.output = nn.Linear(64 * 3 * 3, self.num_classes)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x
    

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters())

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1, verbose=True)
            return [optimizer], [scheduler]
        else:
            return optimizer
        

    def training_step(self, batch, batch_idx):
        ## model.train()을 생략.
        x, y = batch ## to(device)를 생략.
        y_pred = self(x) ## outputs = model(images)

        loss = self.criterion(y_pred, y)
        acc = self.accuracy(y_pred, y)
        ## loss.backward(), optimizer.step()은 생략한다. 라이트닝이 자동으로 수행.

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        ## valid, test에서 사용하던 model.eval()과 torch.no_grad()를 생략한다. 자동으로 수행함.
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        _, preds = torch.max(y_pred, dim=1) ## [batch_size, num_classes]. num_classes 중 최고 확률 하나 선택.
        acc = self.accuracy(preds, y)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("valid_acc", acc, on_step=False, on_epoch=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        _, preds = torch.max(y_pred, dim=1) ## [batch_size, num_classes]. num_classes 중 최고 확률 하나 선택.
        acc = self.accuracy(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True)


    def predict_step(self, batch, batch_idx):
        x, _ = batch
        predictions = self(x)
        _, preds = torch.max(predictions, dim=1)

        return preds


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)