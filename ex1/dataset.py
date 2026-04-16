"""
Data loading and preprocessing for HLA antigen discovery.

This module provides utilities for loading 9-mer amino acid sequences,
encoding them as one-hot vectors, and splitting data into train/test sets.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


# Define the 20 amino acids
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Map amino acid to index
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}


def one_hot_encode_peptide(peptide: str) -> np.ndarray:
    """
    Convert a 9-mer peptide to a one-hot encoded vector.
    
    Description:
        Takes a sequence of 9 amino acids and creates a 180-dimensional
        one-hot encoded vector (9 positions × 20 amino acids).
    
    Args:
        peptide (str): A 9-character string representing amino acids.
    
    Returns:
        np.ndarray: A 180-dimensional numpy array with values in {0, 1}.
    
    Raises:
        ValueError: If peptide is not exactly 9 characters.
        KeyError: If peptide contains invalid amino acids.
    """
    if len(peptide) != 9:
        raise ValueError(f"Peptide must be 9 characters long, got {len(peptide)}")
    
    encoding = np.zeros(180, dtype=np.float32)
    
    for pos, aa in enumerate(peptide):
        if aa not in AA_TO_INDEX:
            raise KeyError(f"Invalid amino acid: {aa}")
        aa_idx = AA_TO_INDEX[aa]
        encoding[pos * 20 + aa_idx] = 1.0
    
    return encoding


def load_sequences_from_file(file_path: str) -> List[str]:
    """
    Load peptide sequences from a text file.
    
    Description:
        Reads sequences from a file, one sequence per line.
        Strips whitespace and skips empty lines.
    
    Args:
        file_path (str): Path to the file containing sequences.
    
    Returns:
        List[str]: List of peptide sequences.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
    return sequences


class PeptideDataset(Dataset):
    """
    PyTorch Dataset for HLA peptide classification.
    
    Loads 9-mer sequences from positive and negative files, encodes them
    as one-hot vectors, and creates train/test splits.
    """
    
    def __init__(self, data_dir: str, test_ratio: float = 0.1, random_seed: int = 42):
        """
        Initialize the PeptideDataset and create train/test splits.
        
        Description:
            Loads all positive and negative peptide sequences from the specified
            directory, performs one-hot encoding, and creates a 90:10 train/test split.
            Both splits are accessible via the train_split and test_split properties.
        
        Args:
            data_dir (str): Path to the directory containing the data files.
            test_ratio (float): Fraction of data to use for testing (default: 0.1 for 90:10 split).
            random_seed (int): Random seed for reproducibility (default: 42).
                
        Side Effects:
            - Reads files from disk during initialization.
            - Prints diagnostic information about loaded data.
        
        Raises:
            FileNotFoundError: If required data files are not found.
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Load data
        self.sequences = []
        self.labels = []
        
        # HLA allele files and their labels
        allele_files = {
            'A0101_pos.txt': 0,
            'A0201_pos.txt': 1,
            'A0203_pos.txt': 2,
            'A0207_pos.txt': 3,
            'A0301_pos.txt': 4,
            'A2402_pos.txt': 5,
        }

        neg_label = len(allele_files)
        
        # Load positive examples
        for filename, label in allele_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Expected file not found: {file_path}")
            
            sequences = load_sequences_from_file(str(file_path))
            self.sequences.extend(sequences)
            self.labels.extend([label] * len(sequences))
            print(f"Loaded {len(sequences)} sequences from {filename} (label: {label})")
        
        # Load negative examples
        neg_file = self.data_dir / 'negs.txt'
        if not neg_file.exists():
            raise FileNotFoundError(f"Expected file not found: {neg_file}")
        
        neg_sequences = load_sequences_from_file(str(neg_file))
        self.sequences.extend(neg_sequences)
        self.labels.extend([neg_label] * len(neg_sequences))
        print(f"Loaded {len(neg_sequences)} sequences from negs.txt (label: {neg_label})")
        
        # Convert to numpy arrays for easier manipulation
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
        # Create train/test split (once, with single shuffle)
        total_samples = len(self.sequences)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        split_idx = int(total_samples * (1 - test_ratio))
        
        self.train_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        
        print(f"\nDataset initialized:")
        print(f"  Total samples: {total_samples}")
        print(f"  Train samples: {len(self.train_indices)}")
        print(f"  Test samples: {len(self.test_indices)}")
        
        # Create split datasets
        self.train_split = _PeptideSplit(self, self.train_indices, 'train')
        self.test_split = _PeptideSplit(self, self.test_indices, 'test')
    
    def __len__(self) -> int:
        """Return total number of samples (use len(dataset.train_split) or len(dataset.test_split))."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Direct access (use dataset.train_split[idx] or dataset.test_split[idx])."""
        raise NotImplementedError("Use dataset.train_split or dataset.test_split")
    
    def get_class_weights(self, split='train') -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.
        
        Description:
            Calculates inverse frequency weights for each class in the specified split.
            Rare classes receive higher weights to improve model learning.
        
        Args:
            split (str): Either 'train' or 'test' (default: 'train').
        
        Returns:
            torch.Tensor: Weight tensor of shape (7,), one weight per class.
        
        Side Effects:
            None
        """
        indices = self.train_indices if split == 'train' else self.test_indices
        unique_labels, counts = np.unique(self.labels[indices], return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique_labels)
        
        # Create weight tensor for all 7 classes
        class_weights = torch.ones(7)
        for label, weight in zip(unique_labels, weights):
            class_weights[label] = weight
        
        return class_weights
    
    def get_label_distribution(self, split='train') -> Dict[int, int]:
        """
        Get the distribution of labels in the specified split.
        
        Description:
            Counts the number of samples for each class in the split.
        
        Args:
            split (str): Either 'train' or 'test' (default: 'train').
        
        Returns:
            Dict[int, int]: Dictionary mapping class labels to sample counts.
        
        Side Effects:
            None
        """
        indices = self.train_indices if split == 'train' else self.test_indices
        unique_labels, counts = np.unique(self.labels[indices], return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}


class _PeptideSplit(Dataset):
    """
    Internal helper class representing a single split (train or test) of PeptideDataset.
    
    Users should not instantiate this directly; access via dataset.train_split or dataset.test_split.
    """
    
    def __init__(self, parent_dataset: PeptideDataset, indices: np.ndarray, split_name: str):
        """
        Initialize a split view of the parent dataset.
        
        Args:
            parent_dataset (PeptideDataset): The parent dataset.
            indices (np.ndarray): Indices for this split.
            split_name (str): Name of split ('train' or 'test').
        """
        self.parent = parent_dataset
        self.indices = indices
        self.split_name = split_name
    
    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from this split.
        
        Description:
            Retrieves a peptide sequence, encodes it as a one-hot vector,
            and returns it with its label.
        
        Args:
            idx (int): Index within this split (0 to len(split)-1).
        
        Returns:
            Tuple[torch.Tensor, int]: A tuple of (encoded_peptide, label)
                where encoded_peptide is a 180-dimensional tensor and label
                is an integer in the range [0, 6].
        
        Side Effects:
            None
        """
        actual_idx = self.indices[idx]
        peptide = self.parent.sequences[actual_idx]
        label = self.parent.labels[actual_idx]
        
        # One-hot encode the peptide
        encoded = one_hot_encode_peptide(peptide)
        encoded_tensor = torch.from_numpy(encoded).float()
        
        return encoded_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for this split.
        
        Description:
            Delegates to parent dataset with the correct split name.
        
        Returns:
            torch.Tensor: Weight tensor of shape (7,), one weight per class.
        """
        return self.parent.get_class_weights(split=self.split_name)
    
    def get_label_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of labels in this split.
        
        Description:
            Delegates to parent dataset with the correct split name.
        
        Returns:
            Dict[int, int]: Dictionary mapping class labels to sample counts.
        """
        return self.parent.get_label_distribution(split=self.split_name)
