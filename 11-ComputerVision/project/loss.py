import torch

from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, is_onehot, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.is_onehot = is_onehot

    def forward(self, inputs, targets):
        if self.is_onehot:
            targets = targets.argmax(dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss