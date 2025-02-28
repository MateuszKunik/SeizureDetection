import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # print(f"Input shape: {x.shape}")

        x = self.conv_block1(x)
        # print(f"First block output shape: {x.shape}")

        x = self.conv_block2(x)
        # print(f"Second block output shape: {x.shape}")    

        x = self.conv_block3(x)
        # print(f"Third block output shape: {x.shape}")     

        x = torch.flatten(x, start_dim=1)
        # print(f"Flatten output shape {x.shape}")

        x = self.fc(x)
        # print(f"Fully conencted output shape: {x.shape}")
        # print(f"Example of output: {x[0]}")

        return x
    

        
class SimpleNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 2,
            dropout: float = 0
    ):
        super().__init__()
        self.conv_net = nn.Sequential(
            # 18, 64, 64
            Conv2Plus1D(in_channels, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2)),
            # 18, 64, 64
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            # 18, 32, 32
            #nn.Dropout(dropout),

            Conv2Plus1D(64, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            # 18, 32, 32
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            # 9, 16, 16
            #nn.Dropout(dropout),

            # Conv2Plus1D(256, 256, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            # nn.BatchNorm3d(64),
            # nn.ReLU(inplace=True),
            # #nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            # #nn.Dropout(dropout),

            Conv2Plus1D(256, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            # 9, 16, 16
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            # 5, 8, 8
            #nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes)
        )


    def forward(self, x):
        x = self.conv_net(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        
        return self.fc(x)
    

    

class STConvNet(nn.Module):
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
        self.convolutional_net = nn.Sequential(
            # conv1
            Conv2Plus1D(in_channels, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # conv2_x
            Conv2Plus1D(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # conv3_x
            Conv2Plus1D(64, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # conv4_x
            Conv2Plus1D(128, 256, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            # conv5_x
            Conv2Plus1D(256, 512, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.convolutional_net(x)
        return self.classifier(x)


class Conv2Plus1D(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            kernel_size: tuple = (3, 3, 3),
            stride: tuple = (1, 1, 1),
            padding: tuple = (0, 0, 0),
            dilation: tuple = (1, 1, 1),
            bias: bool = True
    ):
        """
        Pytorch implementation of (2+1)D spatiotemporal convolution block.

        This model decomposes a 3D convolution into two separate operations:
        - A 2D spatial convolution (applied per frame) to capture spatial features.
        - A 1D temporal convolution (applied across frames) to model temporal relationships.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): (Temporal, Spatial, Spatial) kernel sizes.
            stride (tuple): (Length, Height, Width) strides.
            padding (tuple): (Length, Height, Width) paddings.
            dilation (tuple): (Length, Height, Width) dilation rates.
            bias (bool): Whether to include a bias term in convolutions.

        Attributes:
            spatial_convolution (nn.Conv2d): 2D convolution for spatial feature extraction.
            temporal_convolution (nn.Conv1d): 1D convolution for temporal feature extraction.
            relu (nn.ReLU): Activation function.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.calculate_hidden_channels(kernel_size)
        
        self.spatial_convolution = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size[1:],
            stride=stride[1:],
            dilation=dilation[1:],
            padding=padding[1:],
            bias=bias)

        self.temporal_convolution = nn.Conv1d(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
            bias=bias)
    

    def forward(self, x):
        # 2D spatial convolution
        batch, channels, frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch * frames, channels, height, width)
        x = F.relu(self.spatial_convolution(x))

        # 1D temporal convolution
        _, channels, height, width = x.size()
        x = x.view(batch, frames, channels, height, width)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(batch * height * width, channels, frames)
        x = self.temporal_convolution(x)

        # Final output
        channels, frames = x.size(1), x.size(2)
        x = x.view(batch, height, width, channels, frames)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x


    def calculate_hidden_channels(self, kernel_size):
        """
        Computes the number of intermediate channels (projection dimension) 
        used between spatial and temporal convolutions in the (2+1)D block.

        The goal is to approximate the parameter count of a full 3D convolution 
        while maintaining computational efficiency.

        Args:
            kernel_size (tuple): (Temporal, Spatial, Spatial) kernel sizes.

        Returns:
            int: Number of intermediate channels.
        """
        temporal = kernel_size[0]
        spatial = (kernel_size[1] + kernel_size[2]) // 2
                                                                                
        return int(
            (temporal * spatial ** 2 * self.in_channels * self.out_channels) / (
                spatial ** 2 * self.in_channels + temporal * self.out_channels
            )
        )