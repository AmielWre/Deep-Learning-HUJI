import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    Q1: Encoder model that compresses a 28x28 MNIST image into a latent space.
    The architecture uses strided convolutions to reduce spatial dimensions
    instead of pooling layers for a more compact representation.
    """
    def __init__(self, latent_dim: int, first_layer_channels: int, kernel_size: int = 3, 
                 stride: int = 2, padding: int = 1, input_size: int = 28):
        """
        Args:
            latent_dim (int): The size of the bottleneck vector (d).
            first_layer_channels (int): Number of filters in the first conv layer.
            kernel_size (int): Size of the convolutional kernels.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            input_size (int): The height and width of the input images.
        """
        super(Encoder, self).__init__()
        self.cur_spatial_size = input_size  # For example, 28 for MNIST
        
        # --- Layer 1 ---
        # Example inputs: input size = 28, kernel_size = 3, stride = 2, padding = 1, 
        # first_layer_channels = 15
        # -> input: (N, 1 (grey, not RGB), 28, 28) -> output: (N, 15, 14, 14)
        self.conv1 = nn.Conv2d(1, first_layer_channels, kernel_size, stride, padding)
        self.cur_spatial = self._get_conv_output_size(self.cur_spatial_size, kernel_size, stride, padding)
        
        # --- Layer 2 ---
        # With the same parameters as above: input: (N, 15, 14, 14) -> output: (N, 30, 7, 7)
        self.conv2 = nn.Conv2d(first_layer_channels, first_layer_channels * 2, kernel_size, stride, padding)
        self.cur_spatial = self._get_conv_output_size(self.cur_spatial, kernel_size, stride, padding)
        
        self.relu = nn.ReLU()
        
        # Save the final dimensions for the bottleneck
        self.final_channels = first_layer_channels * 2
        self.final_spatial = self.cur_spatial
        self.flatten_dim = self.final_channels * (self.final_spatial ** 2)
        # With the example parameters, flatten_dim = 30 * 7 * 7 = 1470, 
        # which is the size of the vector before the bottleneck.
        
        # 1a: The bottleneck - mapping the 3D volume to a 1D vector.
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)

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
        x = self.conv_layers(x)
        
        # Step 2: Flatten the 3D cube into a 1D vector per image
        x = torch.flatten(x, start_dim=1)
        
        # Step 3: Project to the latent space d
        latent = self.fc_latent(x)
        return latent

class Decoder(nn.Module):
    """
    Q1: Decoder model that reconstructs an image from a latent vector d.
    It performs the inverse operations of the Encoder[cite: 1].
    """
    def __init__(self, latent_dim: int, last_layer_channels: int, 
                 kernel_size: int = 3, stride: int = 2, padding: int = 1):
        """
        Args:
            latent_dim (int): The size of the input latent vector (d)[cite: 1].
            last_layer_channels (int): Channels in the layer before the final output[cite: 1].
            kernel_size (int): Size of the convolutional kernels.
            stride (int): Stride for the transposed convolutional layers.
            padding (int): Padding for the transposed convolutional layers.
        """
        super(Decoder, self).__init__()
        
        self.channels = last_layer_channels
        self.spatial = 7
        self.flatten_dim = (last_layer_channels * 2) * self.spatial * self.spatial
        
        # Mapping the tiny vector back up to the size needed for 3D convolutions[cite: 1].
        self.fc_upscale = nn.Linear(latent_dim, self.flatten_dim)
        
        self.deconv_layers = nn.Sequential(
            # 7x7 -> 14x14 (upsampling via Transposed Convolution)[cite: 1]
            nn.ConvTranspose2d(last_layer_channels * 2, last_layer_channels, 
                               kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(last_layer_channels, 1, 
                               kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            # Final activation to keep pixels in [0, 1] range[cite: 1].
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs images from latent vectors.
        
        Args:
            x (torch.Tensor): Latent vectors of shape (N, latent_dim).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (N, 1, 28, 28).
        """
        # Step 1: Project d-vector back to high-dimensional vector[cite: 1]
        x = self.fc_upscale(x)
        
        # Step 2: Reshape back into a "cube" (N, Channels, 7, 7) for deconvolution[cite: 1]
        x = x.view(-1, self.channels * 2, self.spatial, self.spatial)
        
        # Step 3: Upscale to final 28x28 image
        reconstruction = self.deconv_layers(x)
        return reconstruction

class ClassifierHead(nn.Module):
    """
    Q2: Simple MLP to map the latent space to 10 digit classes (0-9)[cite: 1].
    """
    def __init__(self, latent_dim: int):
        """
        Args:
            latent_dim (int): The size of the input latent vector (d)[cite: 1].
        """
        super(ClassifierHead, self).__init__()
        # A single linear layer mapping d to 10 class scores (logits)[cite: 1].
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