from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):
        z = self.encoder(x)

        ## Decoder 부분과 classifier 부분으로 분기.
        x_reconstructed = self.decoder(z)
        class_logits = self.classifier(z)

        return x_reconstructed, class_logits

    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        z = self.encode(x)

        return self.classifier(z)
