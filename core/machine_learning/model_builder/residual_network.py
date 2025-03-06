import torch
import torch.nn as nn


class AlaResNet18(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            IdentityBlock(64, dropout),
            IdentityBlock(64, dropout)
        )

        self.conv3 = ResidualBlock(64, 128, dropout)

        self.conv4 = ResidualBlock(128, 256, dropout)

        self.conv5 = ResidualBlock(256, 512, dropout)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)
        )


    def forward(self, x):
        # print(f"input shape: {x.shape}")

        x = self.conv1(x)
        # print(f"conv1 output shape: {x.shape}")

        x = self.conv2(x)
        # print(f"conv2 output shape: {x.shape}")

        x = self.conv3(x)
        # print(f"conv3 output shape: {x.shape}")

        x = self.conv4(x)
        # print(f"conv4 output shape: {x.shape}")

        x = self.conv5(x)
        # print(f"conv5 output shape: {x.shape}")

        x = self.avgpool(x)
        # print(f"avgpool output shape: {x.shape}")

        x = torch.flatten(x, start_dim=1)
        # print(f"flatten output shape: {x.shape}")

        x = self.fc(x)
        # print(f"fc output shape: {x.shape}")

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.conv_block = ConvolutionalBlock(in_channels, out_channels, dropout)
        self.identity_block = IdentityBlock(out_channels, dropout)


    def forward(self, x):
        return self.identity_block(self.conv_block(x))



class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout))
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.convolution(x) + self.shortcut(x))
    


class IdentityBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout))
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.convolution(x) + x)