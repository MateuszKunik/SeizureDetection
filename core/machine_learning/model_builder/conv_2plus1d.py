import torch
import torch.nn as nn
import torch.nn.functional as F


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