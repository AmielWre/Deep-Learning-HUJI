"""
Neural network models for HLA antigen discovery.

This module provides multi-layer perceptron architectures for classifying
9-mer peptides into HLA allele categories.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class HLAMLP(nn.Module):
    """
    Multi-Layer Perceptron for HLA peptide classification.
    
    Provides flexible architecture options:
    - Standard MLP with specified hidden layers
    - Linear layers preserving the 180-dimensional input
    - Variant with ReLU activations for improved non-linearity
    """
    
    def __init__(self, input_dim: int = 180, hidden_dims: Optional[List[int]] = None,
                 num_classes: int = 7, use_relu: bool = False, dropout_rate: float = 0.5):
        """
        Initialize the HLA MLP model.
        
        Description:
            Creates a multi-layer perceptron with configurable architecture.
            Can be set up as a simple linear model or with ReLU activations.
        
        Args:
            input_dim (int): Dimension of input (default: 180 for one-hot encoded 9-mers).
            hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
                If None, uses [180, 180] as in the exercise specification.
            num_classes (int): Number of output classes (default: 7 for 6 alleles + negative).
            use_relu (bool): Whether to use ReLU activations (default: False for linear model).
            dropout_rate (float): Dropout rate for regularization (default: 0.5).
        
        Returns:
            None
        
        Side Effects:
            Initializes the neural network layers.
        
        Raises:
            ValueError: If any dimension is non-positive.
        """
        super(HLAMLP, self).__init__()
        
        if hidden_dims is None:
            # Exercise specification: keep 180 dimensions for 2 inner layers
            hidden_dims = [180, 180]
        
        # Validate dimensions
        if input_dim <= 0 or num_classes <= 0 or dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("Invalid dimension or dropout rate")
        
        self.use_relu = use_relu
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
            
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_relu:
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Description:
            Processes input through all network layers and outputs
            logits for each class.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 180).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        
        Side Effects:
            None
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions (class indices) from input.
        
        Description:
            Performs forward pass and returns the predicted class
            for each sample (argmax of logits).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 180).
        
        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,).
        
        Side Effects:
            None
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax) from input.
        
        Description:
            Performs forward pass and applies softmax to get normalized
            class probabilities.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 180).
        
        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        
        Side Effects:
            None
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class HLAMLPLinear(HLAMLP):
    """
    Linear variant of HLAMLP without ReLU activations.
    
    Useful for comparison with the ReLU-based model as specified
    in the exercise requirements.
    """
    
    def __init__(self, input_dim: int = 180, hidden_dims: Optional[List[int]] = None,
                 num_classes: int = 7):
        """
        Initialize the linear HLA MLP model.
        
        Description:
            Creates a multi-layer perceptron with only linear transformations,
            no ReLU activations or dropout.
        
        Args:
            input_dim (int): Dimension of input (default: 180).
            hidden_dims (Optional[List[int]]): List of hidden layer dimensions
                (default: [180, 180] as per exercise).
            num_classes (int): Number of output classes (default: 7).
        
        Returns:
            None
        
        Side Effects:
            Initializes the neural network layers.
        """
        super(HLAMLPLinear, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            use_relu=False,
            dropout_rate=0.0
        )


class HLAMLPReLU(HLAMLP):
    """
    ReLU variant of HLAMLP with activation functions.
    
    Provides improved non-linearity and learning capacity
    compared to the linear variant.
    """
    
    def __init__(self, input_dim: int = 180, hidden_dims: Optional[List[int]] = None,
                 num_classes: int = 7, dropout_rate: float = 0.5):
        """
        Initialize the ReLU HLA MLP model.
        
        Description:
            Creates a multi-layer perceptron with ReLU activations
            and optional dropout for regularization.
        
        Args:
            input_dim (int): Dimension of input (default: 180).
            hidden_dims (Optional[List[int]]): List of hidden layer dimensions
                (default: [180, 180] as per exercise).
            num_classes (int): Number of output classes (default: 7).
            dropout_rate (float): Dropout rate (default: 0.5).
        
        Returns:
            None
        
        Side Effects:
            Initializes the neural network layers.
        """
        if hidden_dims is None:
            hidden_dims = [180, 180]
        
        super(HLAMLPReLU, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            use_relu=True,
            dropout_rate=dropout_rate
        )
