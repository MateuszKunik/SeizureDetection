import torch
import torch.nn as nn

from .conv_2plus1d import Conv2Plus1D


class R2Plus1DConvNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 2,
            dropout: float = 0
    ):
        """
        Pytorch implementation of the R(2+1)D convolutional model for spatiotemporal data
        based on A Closer Look at Spatiotemporal Convolutions for Action Recognition

        This model is designed for video classification and utilizes (2+1)D
        convolutions, which decompose 3D convolutions into separate spatial and
        temporal operations.  The network consists of multiple convolutional blocks
        followed by adaptive pooling  and a fully connected layer for classification.

        Args:
            in_channels (int): Number of channels in the input image.
            num_class (int): Number of output classes for classification.
            dropout (float): : The probability of dropping units during training.
                                                                                
        Attributes:
            net (nn.Sequential): The main body of the network, consisting of:
                - Multiple Conv2Plus1D blocks with LeakyReLU activations.
                - Adaptive average pooling to reduce feature dimensions.
                - Flattening operation to convert tensors to vectors.
                - Fully connected (fc) layer for classification.
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv2Plus1D(in_channels, 16, (3, 7, 7), (1, 2, 2), (1, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        
        self.conv2 = ResidualBlock(16, 16, dropout)
        self.conv3 = ResidualBlock(16, 32, dropout)
        self.conv4 = ResidualBlock(32, 64, dropout)
        self.conv5 = ResidualBlock(64, 128, dropout)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(100, num_classes)
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
            Conv2Plus1D(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            Conv2Plus1D(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.Dropout(dropout))
        
        self.shortcut = Conv2Plus1D(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(2, 2, 2), padding=(0, 0, 0))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.convolution(x) + self.shortcut(x))
    


class IdentityBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()

        self.convolution = nn.Sequential(
            Conv2Plus1D(channels, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),

            Conv2Plus1D(channels, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.Dropout(dropout))
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.convolution(x) + x)