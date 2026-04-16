"""
Training pipeline and spike protein analysis for HLA antigen discovery.

This module handles model training, loss visualization, and scanning
the SARS-CoV-2 Spike protein for peptides detectable by different HLA alleles.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PeptideDataset, one_hot_encode_peptide
from models import HLAMLPLinear, HLAMLPReLU


# SARS-CoV-2 Spike protein sequence (1273 amino acids)
SPIKE_PROTEIN = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV"
    "NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT"
    "LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN"
    "CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC"
    "NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP"
    "GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI"
    "SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC"
    "LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN"
    "TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRAANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA"
    "ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL"
    "QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
)


def calculate_class_weights(dataset: PeptideDataset, device: torch.device) -> torch.Tensor:
    """
    Calculate weighted CrossEntropyLoss weights for imbalanced classes.
    
    Description:
        Computes inverse frequency weights to handle class imbalance.
        Classes with fewer samples receive higher weights.
    
    Args:
        dataset (PeptideDataset): The dataset to analyze for class distribution.
        device (torch.device): Device to place the tensor on (CPU or GPU).
    
    Returns:
        torch.Tensor: Weight tensor of shape (7,) on the specified device.
    
    Side Effects:
        None
    
    Raises:
        ValueError: If the dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot calculate weights for empty dataset")
    
    weights = dataset.get_class_weights()
    return weights.to(device)


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> float:
    """
    Train the model for one epoch.
    
    Description:
        Performs one complete pass through the training data,
        updating model weights and computing average loss.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimizer for updating weights.
        device (torch.device): Device to run training on (CPU or GPU).
    
    Returns:
        float: Average training loss for the epoch.
    
    Side Effects:
        - Updates model parameters via backward pass and optimizer step.
        - Prints progress information.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on test data.
    
    Description:
        Computes loss and accuracy on the test set without updating weights.
    
    Args:
        model (nn.Module): The neural network model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy) where accuracy is in [0, 1].
    
    Side Effects:
        None
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: torch.device = None, use_class_weights: bool = True) -> Dict[str, List[float]]:
    """
    Complete training pipeline for the model.
    
    Description:
        Trains the model for multiple epochs, evaluates on test set,
        and tracks losses for visualization.
    
    Args:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        num_epochs (int): Number of training epochs (default: 50).
        learning_rate (float): Learning rate for optimizer (default: 0.001).
        device (torch.device): Device to train on (default: CUDA if available, else CPU).
        use_class_weights (bool): Whether to use weighted loss for class imbalance (default: True).
    
    Returns:
        Dict[str, List[float]]: Dictionary with keys 'train_loss', 'test_loss', 'test_accuracy'
            containing lists of values for each epoch.
    
    Side Effects:
        - Modifies model parameters.
        - Prints training progress to console.
    
    Raises:
        ValueError: If loaders are empty or num_epochs is non-positive.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    
    print(f"Training on device: {device}")
    print(f"Using class weights: {use_class_weights}")
    
    model.to(device)
    
    # Setup loss function
    if use_class_weights:
        # Get train dataset from dataloader
        train_dataset = train_loader.dataset
        class_weights = calculate_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(accuracy)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return history


def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """
    Plot training and test losses over epochs.
    
    Description:
        Creates a visualization of loss curves during training.
        Optionally saves the figure to disk.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing 'train_loss' and 'test_loss'.
        save_path (str): Optional path to save the figure (default: None, display only).
    
    Returns:
        None
    
    Side Effects:
        - Creates and displays a matplotlib figure.
        - Saves to disk if save_path is provided.
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Loss plot saved to {save_path}")
    
    plt.show()


def plot_accuracy(history: Dict[str, List[float]], save_path: str = None) -> None:
    """
    Plot test accuracy over epochs.
    
    Description:
        Creates a visualization of test accuracy during training.
    
    Args:
        history (Dict[str, List[float]]): Dictionary containing 'test_accuracy'.
        save_path (str): Optional path to save the figure (default: None).
    
    Returns:
        None
    
    Side Effects:
        - Creates and displays a matplotlib figure.
        - Saves to disk if save_path is provided.
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(history['test_accuracy']) + 1)
    
    plt.plot(epochs, history['test_accuracy'], 'g-', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Test Accuracy Over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Accuracy plot saved to {save_path}")
    
    plt.show()


def scan_spike_protein(model: nn.Module, device: torch.device,
                       spike_sequence: str = SPIKE_PROTEIN,
                       window_size: int = 9) -> List[Tuple[int, str, List[float]]]:
    """
    Perform sliding window scan on Spike protein to find detectable peptides.
    
    Description:
        Uses a sliding window to extract all 9-mers from the Spike protein,
        predicts their detectability (probability of being positive), and
        returns the top 3 most likely detectable peptides.
    
    Args:
        model (nn.Module): Trained model for prediction.
        device (torch.device): Device to run prediction on.
        spike_sequence (str): Spike protein sequence (default: SARS-CoV-2 Spike).
        window_size (int): Size of peptide window (default: 9).
    
    Returns:
        List[Tuple[int, str, List[float]]]: List of (position, peptide, probabilities)
            tuples for the top 3 peptides. Position is 0-indexed.
    
    Side Effects:
        - Runs inference on GPU/CPU.
        - Prints diagnostic information.
    
    Raises:
        ValueError: If spike sequence is shorter than window_size.
    """
    if len(spike_sequence) < window_size:
        raise ValueError(f"Spike sequence too short: {len(spike_sequence)} < {window_size}")
    
    model.eval()
    results = []
    
    print(f"\nScanning Spike protein ({len(spike_sequence)} AA)...")
    print(f"Extracting all {window_size}-mers (total: {len(spike_sequence) - window_size + 1})")
    
    with torch.no_grad():
        for i in range(len(spike_sequence) - window_size + 1):
            peptide = spike_sequence[i:i + window_size]
            
            try:
                # Encode peptide
                encoded = one_hot_encode_peptide(peptide)
                x = torch.from_numpy(encoded).float().unsqueeze(0).to(device)
                
                # Get probabilities
                probs = model.get_probabilities(x)
                probs_list = probs[0].cpu().numpy().tolist()
                
                # Calculate "positive" score (average of allele classes 0-5)
                positive_score = np.mean(probs_list[:6])
                results.append((i, peptide, probs_list, positive_score))
                
            except (ValueError, KeyError):
                # Skip peptides with invalid amino acids
                continue
    
    # Sort by positive score and get top 3
    results.sort(key=lambda x: x[3], reverse=True)
    top_3 = results[:3]
    
    print(f"\nTop 3 most detectable peptides:")
    for rank, (pos, peptide, probs, score) in enumerate(top_3, 1):
        print(f"\n{rank}. Position {pos}: {peptide}")
        print(f"   Overall positive score: {score:.4f}")
        print(f"   Class probabilities:")
        alleles = ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402', 'Negative']
        for allele, prob in zip(alleles, probs):
            print(f"     {allele}: {prob:.4f}")
    
    return [(pos, peptide, probs) for pos, peptide, probs, _ in top_3]


def main():
    """
    Main training script for HLA antigen discovery.
    
    Description:
        Complete pipeline: loads data, creates models, trains them,
        plots results, and analyzes the Spike protein.
    
    Args:
        None
    
    Returns:
        None
    
    Side Effects:
        - Loads data from disk.
        - Trains and modifies model parameters.
        - Creates and saves plots.
        - Prints extensive diagnostic information.
    """
    # Setup paths
    data_dir = Path(__file__).parent / 'ex1 data' / 'ex1 data'
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset once (contains both train and test splits)
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    dataset = PeptideDataset(str(data_dir), test_ratio=0.1)
    train_split = dataset.train_split
    test_split = dataset.test_split
    
    # Print class distribution
    print("\nTrain set class distribution:")
    train_dist = dataset.get_label_distribution(split='train')
    for label, count in sorted(train_dist.items()):
        alleles = ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402', 'Negative']
        print(f"  Class {label} ({alleles[label]}): {count}")
    
    print("\nTest set class distribution:")
    test_dist = dataset.get_label_distribution(split='test')
    for label, count in sorted(test_dist.items()):
        alleles = ['A0101', 'A0201', 'A0203', 'A0207', 'A0301', 'A2402', 'Negative']
        print(f"  Class {label} ({alleles[label]}): {count}")
    
    # Create dataloaders with weighted sampling
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    
    # Use WeightedRandomSampler to handle class imbalance
    class_weights = dataset.get_class_weights(split='train')
    sample_weights = []
    
    for label in dataset.labels[dataset.train_indices]:
        sample_weights.append(class_weights[label].item())
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_split, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_split, batch_size=32, shuffle=False)
    
    print(f"Train DataLoader: {len(train_loader)} batches of size 32")
    print(f"Test DataLoader: {len(test_loader)} batches of size 32")
    
    # ========== Train Linear Model ==========
    print("\n" + "="*60)
    print("TRAINING LINEAR MODEL (No ReLU)")
    print("="*60)
    
    linear_model = HLAMLPLinear()
    linear_history = train_model(
        linear_model,
        train_loader,
        test_loader,
        num_epochs=30,
        learning_rate=0.001,
        device=device,
        use_class_weights=True
    )
    
    # ========== Train ReLU Model ==========
    print("\n" + "="*60)
    print("TRAINING RELU MODEL")
    print("="*60)
    
    relu_model = HLAMLPReLU(dropout_rate=0.3)
    relu_history = train_model(
        relu_model,
        train_loader,
        test_loader,
        num_epochs=30,
        learning_rate=0.001,
        device=device,
        use_class_weights=True
    )
    
    # ========== Visualizations ==========
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    base_save_dir = Path(__file__).parent / 'results'
    base_save_dir.mkdir(exist_ok=True)
    
    plot_training_history(linear_history, save_path=str(base_save_dir / 'linear_loss.png'))
    plot_accuracy(linear_history, save_path=str(base_save_dir / 'linear_accuracy.png'))
    
    plot_training_history(relu_history, save_path=str(base_save_dir / 'relu_loss.png'))
    plot_accuracy(relu_history, save_path=str(base_save_dir / 'relu_accuracy.png'))
    
    # ========== Spike Protein Analysis ==========
    print("\n" + "="*60)
    print("SPIKE PROTEIN ANALYSIS")
    print("="*60)
    
    spike_results_linear = scan_spike_protein(linear_model, device)
    spike_results_relu = scan_spike_protein(relu_model, device)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
