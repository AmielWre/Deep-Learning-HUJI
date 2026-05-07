import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    Q1: Encoder model that compresses a 28x28 MNIST image into a latent space.
    The architecture uses strided convolutions to reduce spatial dimensions
    instead of pooling layers for a more compact representation.
    """
    def __init__(self, in_channels: int, latent_dim: int, first_layer_channels: int, kernel_size: int = 3, 
                 stride: int = 2, padding: int = 1, input_size: int = 28):
        """
        Args:
            in_channels (int): Number of input channels.
            latent_dim (int): The size of the bottleneck vector (d).
            first_layer_channels (int): Number of filters in the first conv layer.
            kernel_size (int): Size of the convolutional kernels.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            input_size (int): The height and width of the input images.
        """
        super(Encoder, self).__init__()
        cur_spatial_size = input_size  # For example, 28 for MNIST
        
        print(f"  Encoder Layer 1 - Input: {in_channels}x{cur_spatial_size}x{cur_spatial_size}")
        
        # --- Layer 1 ---
        # Example inputs: input size = 28, kernel_size = 3, stride = 2, padding = 1, 
        # first_layer_channels = 15
        # -> input: (N, 1 (grey, not RGB), 28, 28) -> output: (N, 15, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=first_layer_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        cur_spatial_size = self._get_conv_output_size(cur_spatial_size, kernel_size, stride, padding)
        print(f"    Output: {first_layer_channels}x{cur_spatial_size}x{cur_spatial_size}")
        
        # --- Layer 2 ---
        # N = batch size
        # With the same parameters as above: input: (N, 15, 14, 14) -> output: (N, 30, 7, 7)
        self.conv2 = nn.Conv2d(first_layer_channels, first_layer_channels * 2, kernel_size, stride, padding)
        print(f"  Encoder Layer 2 - Input: {first_layer_channels}x{cur_spatial_size}x{cur_spatial_size}")
        cur_spatial_size = self._get_conv_output_size(cur_spatial_size, kernel_size, stride, padding)
        print(f"    Output: {first_layer_channels * 2}x{cur_spatial_size}x{cur_spatial_size}")
        
        self.relu = nn.ReLU()
        
        # Save the final dimensions for the bottleneck
        self.final_channels = first_layer_channels * 2
        self.final_spatial = cur_spatial_size
        self.flatten_dim = self.final_channels * (self.final_spatial ** 2)
        # With the example parameters, flatten_dim = 30 * 7 * 7 = 1470, 
        # which is the size of the vector before the bottleneck.
        
        # 1a: The bottleneck - mapping the 3D volume to a 1D vector.
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)
        print(f"  Encoder Bottleneck - Flatten: {self.flatten_dim}, Output (Latent): {latent_dim}")

    def _get_conv_output_size(self, n: int, kernel_size: int, stride: int, padding: int) -> int:
        """
        Utility function to calculate the output spatial size after a convolutional layer.
        
        Args:
            n (int): The input spatial size (height/width).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding applied to the input.
        
        Returns:
            int: The output spatial size after the convolution.
        """
        return ((n + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of images and returns their latent representations.
        
        Args:
            x (torch.Tensor): Batch of images of shape (N, 1, n, n).
            
        Returns:
            torch.Tensor: Latent vectors of shape (N, latent_dim).
        """
        # Step 1: Pass through convolutions to get feature maps (N, C*2, 7, 7)
        # Layer 1
        x = self.relu(self.conv1(x))
        # Layer 2
        x = self.relu(self.conv2(x))
        
        # Flatten and project to latent space
        x = torch.flatten(x, start_dim=1)
        x = self.fc_latent(x)
        return x

class Decoder(nn.Module):
    """
    Decoder model that reconstructs an image from a latent vector d.
    It performs the inverse operations of the Encoder using Transposed Convolutions.
    """
    def __init__(self, 
                 in_channels: int, latent_dim: int, last_layer_channels: int,  kernel_size: int = 3, 
                 stride: int = 2,  padding: int = 1, output_padding: int = 1,
                 bottleneck_spatial: int = 7):
        """
        Args:
            in_channels (int): Number of input channels.
            latent_dim (int): The size of the input latent vector (d).
            last_layer_channels (int): Channels in the layer before the final output.
            kernel_size (int): Size of the convolutional kernels.
            stride (int): Stride for the transposed convolutional layers.
            padding (int): Padding for the transposed convolutional layers.
            output_padding (int): Additional size added to one side of the output shape.
            bottleneck_spatial (int): The starting spatial size (e.g., 7 for MNIST).
        """
        super(Decoder, self).__init__()
        
        # Track the spatial size dynamically as we "un-shrink" the image
        self.cur_spatial = bottleneck_spatial
        self.last_layer_channels = last_layer_channels
        
        # Calculate the size needed to reconstruct the 3D volume from the 1D latent vector
        self.flatten_dim = (last_layer_channels * 2) * (self.cur_spatial ** 2)
        
        # Step 1: Mapping the tiny vector back up to the flattened size
        self.fc_upscale = nn.Linear(latent_dim, self.flatten_dim)
        
        print(f"  Decoder Upscale - Input (Latent): {latent_dim}, Output (Flatten): {self.flatten_dim}")
        
        # Step 1: Mapping the tiny vector back up to the flattened size
        self.fc_upscale = nn.Linear(latent_dim, self.flatten_dim)
        
        # --- Layer 1: 7x7 -> 14x14 ---
        # Formula: (n-1)*s - 2*p + k + output_padding
        print(f"  Decoder Layer 1 - Input: {last_layer_channels * 2}x{self.cur_spatial}x{self.cur_spatial}")
        self.deconv1 = nn.ConvTranspose2d(last_layer_channels * 2, last_layer_channels, 
                                          kernel_size=kernel_size, stride=stride, 
                                          padding=padding, output_padding=output_padding)
        self.cur_spatial = self._get_deconv_output_size(self.cur_spatial, kernel_size, stride, padding, output_padding)
        print(f"    Output: {last_layer_channels}x{self.cur_spatial}x{self.cur_spatial}")
        
        # --- Layer 2: 14x14 -> 28x28 ---
        print(f"  Decoder Layer 2 - Input: {last_layer_channels}x{self.cur_spatial}x{self.cur_spatial}")
        self.deconv2 = nn.ConvTranspose2d(last_layer_channels, in_channels, 
                                          kernel_size=kernel_size, stride=stride, 
                                          padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # Final pixels must be [0, 1]
        self.cur_spatial = self._get_deconv_output_size(self.cur_spatial, kernel_size, stride, padding, output_padding)
        print(f"    Output: {in_channels}x{self.cur_spatial}x{self.cur_spatial}")

    def _get_deconv_output_size(self, n: int, kernel_size: int, stride: int, padding: int, output_padding: int) -> int:
        """
        Calculates the output spatial size after a Transposed Convolutional layer.
        Formula: (n - 1) * stride - 2 * padding + kernel_size + output_padding
        """
        return (n - 1) * stride - 2 * padding + kernel_size + output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs images from latent vectors.
        
        Args:
            x (torch.Tensor): Latent vectors of shape (N, latent_dim).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (N, 1, 28, 28).
        """
        # Step 1: Project d-vector back to high-dimensional vector
        x = self.fc_upscale(x)
        
        # Step 2: Reshape back into a "cube" for deconvolution
        # Example shape: (N, last_layer_channels * 2, 7, 7)
        x = x.view(-1, self.last_layer_channels * 2, 7, 7)
        
        # Step 3: Deconvolve and Upscale
        x = self.relu(self.deconv1(x))
        
        # Step 4: Final reconstruction with Sigmoid
        x = self.sigmoid(self.deconv2(x))
        return x

class ClassifierHead(nn.Module):
    """
    Simple MLP to map the latent space to 10 digit classes (0-9).
    """
    def __init__(self, latent_dim: int):
        """
        Args:
            latent_dim (int): The size of the input latent vector (d).
        """
        super(ClassifierHead, self).__init__()
        # A single linear layer mapping d to 10 class scores (logits).
        self.fc = nn.Linear(latent_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts class scores for a latent representation.
        
        Args:
            x (torch.Tensor): Latent vectors of shape (N, latent_dim).
            
        Returns:
            torch.Tensor: Unnormalized class scores (logits) of shape (N, 10).
        """
        return self.fc(x)