import torch
import torch.nn as nn

from .conv_2plus1d import Conv2Plus1D


class Simple2Plus1DConvNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 2,
            dropout: float = 0
    ):
        super().__init__()
        self.conv_net = nn.Sequential(
            Conv2Plus1D(in_channels, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            Conv2Plus1D(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            #nn.Dropout(dropout),


            Conv2Plus1D(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            Conv2Plus1D(128, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            #nn.Dropout(dropout),


            Conv2Plus1D(128, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            Conv2Plus1D(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            #nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(16, num_classes)
        )


    def forward(self, x):
        x = self.conv_net(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        return self.fc(x)