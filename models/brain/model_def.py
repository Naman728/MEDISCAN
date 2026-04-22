import torch
from torch import nn
from torchvision.models import vgg16


class BrainMultiHeadModel(nn.Module):
    def __init__(self, num_heads: int = 6):
        super().__init__()
        self.features = vgg16(weights=None).features

        self.shared_fc = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        logits = [head(x) for head in self.heads]
        return torch.cat(logits, dim=1)


def build_model():
    return BrainMultiHeadModel(num_heads=6)