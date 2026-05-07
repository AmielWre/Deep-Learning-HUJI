import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Encoder, Decoder
from dataset import get_mnist_loaders
import numpy as np
import matplotlib
matplotlib.use('Agg')  # This tells matplotlib to run without a GUI window
import matplotlib.pyplot as plt
import os


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

if __name__ == "__main__":
    print("hi")
    #run_reconstruction_experiment()