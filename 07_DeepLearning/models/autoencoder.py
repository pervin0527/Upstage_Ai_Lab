import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.output_activation = torch.sigmoid
        
        # Encoder layers
        self.encoder1 = nn.Linear(784, 512)
        self.encoder2 = nn.Linear(512, 256)
        self.encoder3 = nn.Linear(256, 128)
        
        # Decoder layers
        self.decoder1 = nn.Linear(128, 256)
        self.decoder2 = nn.Linear(256, 512)
        self.decoder3 = nn.Linear(512, 784)
        
    def forward(self, x):
        x = x.view(-1, 784)
        
        # Encoding
        x = self.activation(self.encoder1(x))
        x = self.activation(self.encoder2(x))
        x = self.activation(self.encoder3(x))
        
        # Decoding
        x = self.activation(self.decoder1(x))
        x = self.activation(self.decoder2(x))
        x = self.output_activation(self.decoder3(x))
        
        # Reshape back to image format
        x = x.view(-1, 1, 28, 28)

        return x


class EncoderClassifier(nn.Module):
    def __init__(self, autoencoder, num_classes=10):
        super().__init__()
        self.encoder1 = autoencoder.encoder1
        self.encoder2 = autoencoder.encoder2
        self.encoder3 = autoencoder.encoder3
        self.activation = autoencoder.activation
        
        # Classification layer
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 784)
        
        # Encoding
        x = self.activation(self.encoder1(x))
        x = self.activation(self.encoder2(x))
        x = self.activation(self.encoder3(x))
        
        # Classification
        x = self.classifier(x)
        
        return x
