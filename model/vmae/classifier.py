import torch
import torch.nn as nn
import torch.nn.functional as F


class MILClassifier(nn.Module):
    def __init__(self, input_dim=710, drop_p=0.0):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.drop_p = drop_p
        self.weight_init()

    def weight_init(self):
        # for layer in self.classifier:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.classifier(x)

        return x
