import torch
import torch.nn as nn
import torchvision.models as models


class CTFeatureModel(nn.Module):
    def __init__(self, n: int = 7):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for _ in range(n)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        s = self.shared_fc(x)
        return torch.cat([h(s) for h in self.heads], dim=1)


def build_model(num_heads: int = 7):
    return CTFeatureModel(n=num_heads)
