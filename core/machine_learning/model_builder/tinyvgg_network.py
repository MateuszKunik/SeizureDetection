import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 2,
            dropout: float = 0
    ):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # nn.Conv2d(128, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes))
        

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x