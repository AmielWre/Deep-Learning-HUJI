import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Encoder, Decoder, ClassifierHead
from dataset import get_mnist_loaders
import numpy as np
import matplotlib
matplotlib.use('Agg')  # This tells matplotlib to run without a GUI window
import matplotlib.pyplot as plt
import os
from datetime import datetime


def load_pretrained_encoder(encoder_path: str, latent_dim: int, first_layer_channels: int, device: torch.device) -> Encoder:
    """
    Loads a pre-trained encoder from the specified path.
    
    We retrieve the saved weights from disk and load them into a new encoder instance
    for use as a fixed feature extractor.
    
    Args:
        encoder_path (str): Full path to the encoder weights file (.pth).
        latent_dim (int): The latent dimension of the encoder.
        first_layer_channels (int): The first layer channels of the encoder.
        device (torch.device): Device to load the model on (CPU or GPU).
        
    Returns:
        Encoder: Pre-trained encoder model with loaded weights.
    """
    in_channels = 1
    encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, 
                     first_layer_channels=first_layer_channels).to(device)
    
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Pre-trained encoder loaded from: {encoder_path}")
    else:
        print(f"Warning: Pre-trained encoder not found at {encoder_path}")
        print(f"Using randomly initialized encoder instead.")
    
    return encoder


def run_classifier_training(use_pretrained: bool = False,
                           freeze_encoder: bool = False,
                           use_subset: bool = False,
                           scenario_label: str = "experiment",
                           batch_size: int = 16,
                           learning_rate: float = 1e-3,
                           epochs: int = 10,
                           latent_dim: int = 16,
                           first_layer_channels: int = 15) -> dict:
    """
    Unified function for training a classifier (Q2 and Q3).
    
    We train a classification network combining an encoder with an MLP classifier.
    For Q2, we train both encoder and classifier from scratch.
    For Q3, we load pre-trained encoder weights and freeze them, training only the MLP.
    
    Args:
        use_pretrained (bool): Whether to load pre-trained encoder from Q1.
        freeze_encoder (bool): Whether to freeze encoder weights during training.
        use_subset (bool): If True, use only 100 random training examples.
        scenario_label (str): Label for this scenario (used in output filenames and logs).
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for Adam optimizer.
        epochs (int): Number of training epochs.
        latent_dim (int): Latent dimension of the encoder.
        first_layer_channels (int): First layer channels of the encoder.
        
    Returns:
        dict: Dictionary containing 'train_losses', 'test_losses', 'train_accuracies', 'test_accuracies'
    """
    in_channels = 1
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_label.upper()}")
    print(f"Pre-trained: {use_pretrained}, Freeze Encoder: {freeze_encoder}")
    print(f"Training on {'100 random examples' if use_subset else 'full MNIST dataset (60,000 examples)'}")
    print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading MNIST dataset (use_subset={use_subset})...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, use_subset=use_subset)
    
    # Initialize encoder
    if use_pretrained:
        print(f"Loading Pre-trained Encoder (latent_dim={latent_dim}, first_layer_channels={first_layer_channels}):")
        models_dir = os.path.join(os.path.dirname(__file__), "ex2 results", "models")
        encoder_path = os.path.join(models_dir, f"encoder_d{latent_dim}_c{first_layer_channels}.pth")
        encoder = load_pretrained_encoder(encoder_path, latent_dim, first_layer_channels, device)
    else:
        print(f"Initializing Encoder from scratch (latent_dim={latent_dim}, first_layer_channels={first_layer_channels}):")
        encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, 
                         first_layer_channels=first_layer_channels).to(device)
    
    # Initialize classifier head (MLP)
    print(f"Initializing ClassifierHead (latent_dim={latent_dim} -> {num_classes} classes):")
    classifier = ClassifierHead(latent_dim=latent_dim).to(device)
    
    # Freeze encoder if requested
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen (not trainable)")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if freeze_encoder:
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    # Tracking lists
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        # --- Training Phase ---
        encoder.train()
        classifier.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            latents = encoder(images)
            logits = classifier(latents)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_running_loss += loss.item()
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_running_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # --- Testing Phase ---
        encoder.eval()
        classifier.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                latents = encoder(images)
                logits = classifier(latents)
                loss = criterion(logits, labels)
                
                # Track metrics
                test_running_loss += loss.item()
                test_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                test_total += labels.size(0)
        
        avg_test_loss = test_running_loss / len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    # Print summary
    print(f"\nFinal Results ({scenario_label}):")
    print(f"  Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"  Training Loss: {train_losses[-1]:.4f}")
    print(f"  Test Loss: {test_losses[-1]:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'encoder': encoder,
        'classifier': classifier
    }


def save_classifier_encoder(encoder: nn.Module, latent_dim: int, first_layer_channels: int, prefix: str = "classification") -> None:
    """
    Saves a classification-trained encoder to disk.
    
    This saves encoders trained for classification tasks (e.g., Q2)
    so they can be used later for Q4 (task-specific reconstruction).
    
    Args:
        encoder (nn.Module): Trained encoder model.
        latent_dim (int): The latent dimension used.
        first_layer_channels (int): The first layer channels used.
        prefix (str): Prefix to distinguish encoder type (e.g., "classification").
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results", "models")
    os.makedirs(results_dir, exist_ok=True)
    
    encoder_path = os.path.join(results_dir, f"encoder_{prefix}_d{latent_dim}_c{first_layer_channels}.pth")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Classification encoder saved: {encoder_path}")

def load_classifier_encoder(latent_dim: int, first_layer_channels: int, device: torch.device) -> Encoder:
    """
    Loads a pre-trained classification encoder (e.g., from Q2).
    
    Used by Q4 to load the encoder trained for classification tasks.
    
    Args:
        latent_dim (int): The latent dimension of the encoder.
        first_layer_channels (int): The first layer channels of the encoder.
        device (torch.device): Device to load the model on (CPU or GPU).
        
    Returns:
        Encoder: Classification-trained encoder model with loaded weights.
    """
    in_channels = 1
    encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, 
                     first_layer_channels=first_layer_channels).to(device)
    
    models_dir = os.path.join(os.path.dirname(__file__), "ex2 results", "models")
    encoder_path = os.path.join(models_dir, f"encoder_classification_d{latent_dim}_c{first_layer_channels}.pth")
    
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"Classification encoder loaded from: {encoder_path}")
    else:
        print(f"Warning: Classification encoder not found at {encoder_path}")
        print(f"Please ensure Q2 has been run to generate the encoder.")
    
    return encoder

def plot_reconstruction_results(losses: list[float], 
                               original_images: torch.Tensor, 
                               reconstructed_images: torch.Tensor, 
                               latent_dim: int,
                               first_layer_channels: int,
                               num_samples: int = 5,
                               filename_prefix: str = "experiment") -> None:
    """
    Creates diagnostic plots for the autoencoder performance.
    
    Args:
        losses (list[float]): A list of loss values per epoch.
        original_images (torch.Tensor): A batch of original images from the test set.
        reconstructed_images (torch.Tensor): The output from the decoder for those images.
        latent_dim (int): The latent dimension used.
        first_layer_channels (int): The first layer channels used.
        num_samples (int): Number of image pairs to display.
        filename_prefix (str): Prefix for the saved filenames.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Save Original vs Reconstructed comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    plt.suptitle(f"Original (Top) vs. Reconstructed (Bottom)\nLatent Dim={latent_dim}, First Layer Channels={first_layer_channels}")

    originals = original_images.cpu().detach().numpy()
    reconstructions = reconstructed_images.cpu().detach().numpy()

    for i in range(num_samples):
        axes[0, i].imshow(originals[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    filepath = os.path.join(results_dir, f"{filename_prefix}_comparison.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Comparison plot saved: {filepath}")

def plot_combined_loss_curves(all_losses_data: dict, filename_prefix: str = "all_experiments") -> None:
    """
    Plots all loss curves on a single graph with different colors and labels.
    
    Args:
        all_losses_data (dict): Dictionary mapping config names to loss lists.
                               Example: {"d=4, c=3": [losses...], "d=4, c=15": [losses...], ...}
        filename_prefix (str): Prefix for the saved filename.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Save the Loss curve with all 4 configurations
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors
    
    for (config_name, losses), color in zip(all_losses_data.items(), colors):
        plt.plot(losses, label=config_name, color=color, linewidth=2)
    
    plt.title('Autoencoder Training: Comparison of Configurations')
    plt.xlabel('Epoch')
    plt.ylabel('Mean L1 Reconstruction Loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(results_dir, f"{filename_prefix}_loss_curve.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Combined loss curve saved: {filepath}")

def save_models(encoder: nn.Module, decoder: nn.Module, latent_dim: int, first_layer_channels: int) -> None:
    """
    Saves encoder and decoder model weights to disk.
    
    Args:
        encoder (nn.Module): Trained encoder model.
        decoder (nn.Module): Trained decoder model.
        latent_dim (int): The latent dimension used.
        first_layer_channels (int): The first layer channels used.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results", "models")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save encoder weights
    encoder_path = os.path.join(results_dir, f"encoder_d{latent_dim}_c{first_layer_channels}.pth")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved: {encoder_path}")
    
    # Save decoder weights
    decoder_path = os.path.join(results_dir, f"decoder_d{latent_dim}_c{first_layer_channels}.pth")
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Decoder weights saved: {decoder_path}")

def run_reconstruction_experiment() -> None:
    """
    Implements the first stage of the exercise: Training a convolutional autoencoder.
    Runs 4 different configurations and compares their results.
    
    This function initializes the datasets, builds the encoder/decoder pairs, 
    and executes the training loop using Mean L1 Loss for image reconstruction.
    Configurations tested:
    - latent_dim: 4, 16
    - first_layer_channels: 3, 15
    """
    # Hyperparameters - Fixed across all experiments
    in_channels = 1  # MNIST images are grayscale
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # All 4 configurations
    configs = [
        (4, 3),    # latent_dim=4, first_layer_channels=3
        (4, 15),   # latent_dim=4, first_layer_channels=15
        (16, 3),   # latent_dim=16, first_layer_channels=3
        (16, 15),  # latent_dim=16, first_layer_channels=15
    ]
    
    # 1. Prepare Data (shared for all configs)
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    
    # Dictionary to store loss curves for all configs
    all_losses_data = {}
    test_images_dict = {}
    reconstructed_dict = {}
    
    # 2. Run training for each configuration
    for latent_dim, first_layer_channels in configs:
        config_name = f"d={latent_dim}, c={first_layer_channels}"
        print(f"\n{'='*70}")
        print(f"Starting Training: Latent Dim={latent_dim}, Channels={first_layer_channels}")
        print(f"{'='*70}")
        
        # Initialize Models
        print(f"Initializing Encoder with latent_dim={latent_dim}, first_layer_channels={first_layer_channels}:")
        encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, 
                         first_layer_channels=first_layer_channels).to(device)
        
        print(f"Initializing Decoder with latent_dim={latent_dim}, first_layer_channels={first_layer_channels}:")
        decoder = Decoder(in_channels=in_channels, latent_dim=latent_dim, 
                         last_layer_channels=first_layer_channels).to(device)
        
        # Define Loss and Optimizer
        criterion = nn.L1Loss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        
        # Training Loop
        epoch_losses = []
        
        for epoch in range(epochs):
            encoder.train()
            decoder.train()
            running_loss = 0.0
            
            for images, _ in train_loader:
                images = images.to(device)
                # Forward pass
                latent = encoder(images)
                reconstructed = decoder(latent)
                loss = criterion(reconstructed, images)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Store loss data for later plotting
        all_losses_data[config_name] = epoch_losses
        
        # Save trained models
        save_models(encoder, decoder, latent_dim, first_layer_channels)
        
        # --- Final Visualization for this config ---
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # Get a small batch from the test loader
            test_images, _ = next(iter(test_loader))
            test_images = test_images.to(device)
            
            # Reconstruct
            latents = encoder(test_images)
            reconstructed = decoder(latents)
            
            # Save for later
            test_images_dict[config_name] = test_images
            reconstructed_dict[config_name] = reconstructed
            
            # Call the plotting function for individual config
            prefix_filename = f"autoencoder_d{latent_dim}_c{first_layer_channels}"
            plot_reconstruction_results(
                epoch_losses, 
                test_images, 
                reconstructed, 
                latent_dim=latent_dim,
                first_layer_channels=first_layer_channels,
                num_samples=5, 
                filename_prefix=prefix_filename
            )
    
    # 3. Plot combined loss curves for all 4 configurations
    print(f"\n{'='*70}")
    print("Creating combined loss curve plot...")
    print(f"{'='*70}")
    plot_combined_loss_curves(all_losses_data, filename_prefix="all_configs")

def plot_classifier_results(train_losses: list[float],
                           test_losses: list[float],
                           train_accuracies: list[float],
                           test_accuracies: list[float],
                           batch_size: int,
                           scenario_name: str = "classifier") -> None:
    """
    Creates diagnostic plots for the classifier performance.
    
    We generate two plots: one showing loss evolution and one showing accuracy evolution
    across training and test sets throughout the training epochs.
    
    Args:
        train_losses (list[float]): Training loss values per epoch.
        test_losses (list[float]): Test loss values per epoch.
        train_accuracies (list[float]): Training accuracy values per epoch.
        test_accuracies (list[float]): Test accuracy values per epoch.
        scenario_name (str): Name of the scenario (e.g., "full", "subset") for filename.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', marker='s', linewidth=2)
    plt.title(f'Classifier Training: Loss ({scenario_name.capitalize()}), Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_filepath = os.path.join(results_dir, f"classifier_{scenario_name}_loss.png")
    plt.savefig(loss_filepath)
    plt.close()
    print(f"Loss plot saved: {loss_filepath}")
    
    # Plot 2: Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s', linewidth=2)
    plt.title(f'Classifier Training: Accuracy ({scenario_name.capitalize()}), Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    accuracy_filepath = os.path.join(results_dir, f"classifier_{scenario_name}_accuracy.png")
    plt.savefig(accuracy_filepath)
    plt.close()
    print(f"Accuracy plot saved: {accuracy_filepath}")

def compute_accuracy(model_output: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes classification accuracy.
    
    We calculate the percentage of correct predictions by comparing the model's
    predicted classes (argmax of logits) with the ground truth labels.
    
    Args:
        model_output (torch.Tensor): Model output logits of shape (N, num_classes).
        labels (torch.Tensor): Ground truth labels of shape (N,).
        
    Returns:
        float: Accuracy as a percentage (0-100).
    """
    predictions = torch.argmax(model_output, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    return accuracy

def plot_q3_results(train_losses: list[float],
                    test_losses: list[float],
                    train_accuracies: list[float],
                    test_accuracies: list[float],
                    batch_size: int,
                    scenario_name: str = "q3") -> None:
    """
    Creates diagnostic plots for Q3 (transfer learning) performance.
    
    We generate separate plots showing loss and accuracy evolution for Q3.
    Each plot includes detailed information about the training configuration.
    
    Args:
        train_losses (list[float]): Training loss values per epoch.
        test_losses (list[float]): Test loss values per epoch.
        train_accuracies (list[float]): Training accuracy values per epoch.
        test_accuracies (list[float]): Test accuracy values per epoch.
        batch_size (int): Batch size used for training.
        scenario_name (str): Name of the scenario (e.g., "q3_full", "q3_subset") for filename.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "ex2 results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', marker='s', linewidth=2)
    plt.title(f'Q3 Transfer Learning: Loss ({scenario_name.replace("q3_", "").capitalize()}), Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_filepath = os.path.join(results_dir, f"{scenario_name}_loss.png")
    plt.savefig(loss_filepath)
    plt.close()
    print(f"Q3 Loss plot saved: {loss_filepath}")
    
    # Plot 2: Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s', linewidth=2)
    plt.title(f'Q3 Transfer Learning: Accuracy ({scenario_name.replace("q3_", "").capitalize()}), Batch Size={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    accuracy_filepath = os.path.join(results_dir, f"{scenario_name}_accuracy.png")
    plt.savefig(accuracy_filepath)
    plt.close()
    print(f"Q3 Accuracy plot saved: {accuracy_filepath}")


def run_classifier_from_scratch() -> dict:
    """
    Question 2: Training classifiers from scratch (Q2).
    
    We train classification networks that combine the encoder architecture with an MLP.
    Both encoder and classifier are trained together from randomly initialized weights.
    Two scenarios are evaluated: full dataset and 100-sample subset.
    
    Returns:
        dict: Dictionary with keys "q2_full" and "q2_subset" containing training results.
    """
    print(f"\n{'='*70}")
    print("EXERCISE 2 - QUESTION 2: Classifier Training from Scratch")
    print(f"{'='*70}")
    
    results = {}
    batch_size = 16
    learning_rate = 1e-3
    epochs = 10
    
    # Q2: Full dataset
    results['q2_full'] = run_classifier_training(
        use_pretrained=False,
        freeze_encoder=False,
        use_subset=False,
        scenario_label="q2_full",
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Q2: Subset (100 examples)
    results['q2_subset'] = run_classifier_training(
        use_pretrained=False,
        freeze_encoder=False,
        use_subset=True,
        scenario_label="q2_subset",
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Generate plots for Q2
    print(f"\nGenerating plots for Q2 scenarios...")
    plot_classifier_results(
        results['q2_full']['train_losses'],
        results['q2_full']['test_losses'],
        results['q2_full']['train_accuracies'],
        results['q2_full']['test_accuracies'],
        batch_size=batch_size,
        scenario_name="q2_full"
    )
    
    plot_classifier_results(
        results['q2_subset']['train_losses'],
        results['q2_subset']['test_losses'],
        results['q2_subset']['train_accuracies'],
        results['q2_subset']['test_accuracies'],
        batch_size=batch_size,
        scenario_name="q2_subset"
    )
    
    # Save Q2 encoder (full dataset version only) for Q4 use
    print(f"\nSaving Q2 encoder for Q4 (Task-Specific Encoding)...")
    latent_dim = 16
    first_layer_channels = 15
    save_classifier_encoder(
        results['q2_full']['encoder'],
        latent_dim=latent_dim,
        first_layer_channels=first_layer_channels,
        prefix="classification"
    )
    
    return results

def run_classifier_pre_trained(q2_results: dict = None) -> dict:
    """
    Question 3: Transfer learning with pre-trained encoder (Q3).
    
    We train classifiers using the unsupervised pre-trained encoder from Q1.
    The encoder is frozen (non-trainable) and only the MLP head is trained.
    Two scenarios are evaluated: full dataset and 100-sample subset.
    
    Can be run independently without Q2 results. If Q2 results are provided,
    they will be used for optional comparison analysis.
    
    Args:
        q2_results (dict, optional): Results from Q2 for comparison. If None, Q3 runs standalone.
        
    Returns:
        dict: Dictionary with keys "q3_full" and "q3_subset" containing training results.
    """
    print(f"\n{'='*70}")
    print("EXERCISE 2 - QUESTION 3: Transfer Learning with Pre-trained Encoder")
    print(f"{'='*70}")
    
    results = {}
    batch_size = 16
    learning_rate = 1e-3
    epochs = 10
    
    # Q3: Full dataset with frozen pre-trained encoder
    results['q3_full'] = run_classifier_training(
        use_pretrained=True,
        freeze_encoder=True,
        use_subset=False,
        scenario_label="q3_full",
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Q3: Subset with frozen pre-trained encoder
    results['q3_subset'] = run_classifier_training(
        use_pretrained=True,
        freeze_encoder=True,
        use_subset=True,
        scenario_label="q3_subset",
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Generate separate Q3 plots
    print(f"\nGenerating Q3 plots...")
    
    # Full dataset Q3 plot
    plot_q3_results(
        results['q3_full']['train_losses'],
        results['q3_full']['test_losses'],
        results['q3_full']['train_accuracies'],
        results['q3_full']['test_accuracies'],
        batch_size=batch_size,
        scenario_name="q3_full"
    )
    
    # Subset Q3 plot
    plot_q3_results(
        results['q3_subset']['train_losses'],
        results['q3_subset']['test_losses'],
        results['q3_subset']['train_accuracies'],
        results['q3_subset']['test_accuracies'],
        batch_size=batch_size,
        scenario_name="q3_subset"
    )
    
    return results

if __name__ == "__main__":
    # Uncomment to run Q1 (Autoencoder reconstruction)
    run_reconstruction_experiment()
    
    # Run Q2 and Q3 with comparison
    # q2_results = run_classifier_from_scratch()  # Q2: Train from scratch
    # q3_results = run_classifier_pre_trained()  # Q3: Transfer learning. Note that you
    # need to have the pre-trained encoder weights from Q1 for this to work properly.
    
    print(f"\n{'='*70}")
    print("All experiments completed. Check 'ex2 results' folder for outputs.")
    print(f"{'='*70}")
