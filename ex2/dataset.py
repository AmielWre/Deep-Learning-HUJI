import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

def get_mnist_loaders(batch_size: int = 64, use_subset: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads and prepares the MNIST dataset loaders.
    
    Args:
        batch_size (int): Number of images per training batch.
        use_subset (bool): If True, returns a loader with only 100 random training examples.
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Standard normalization for MNIST: images come in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download datasets
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if use_subset:
        # Create a subset of exactly 100 samples as requested in the exercise
        indices = torch.randperm(len(train_set))[:100]
        train_set = Subset(train_set, indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader