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


def plot_reconstruction_results(losses: list[float], 
                               original_images: torch.Tensor, 
                               reconstructed_images: torch.Tensor, 
                               num_samples: int = 5,
                               filename_prefix: str = "experiment") -> None:
    """
    Creates diagnostic plots for the autoencoder performance.
    
    Args:
        losses (list[float]): A list of loss values per epoch.
        original_images (torch.Tensor): A batch of original images from the test set.
        reconstructed_images (torch.Tensor): The output from the decoder for those images.
        num_samples (int): Number of image pairs to display.
        filename_prefix (str): Prefix for the saved filenames.
    """
    # 1. Save the Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Mean L1 Reconstruction Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename_prefix}_loss_curve.png") # Save to file
    plt.close() # Close to free up memory

    # 2. Save Original vs Reconstructed comparison
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    plt.suptitle("Original (Top) vs. Reconstructed (Bottom)")

    originals = original_images.cpu().detach().numpy()
    reconstructions = reconstructed_images.cpu().detach().numpy()

    for i in range(num_samples):
        axes[0, i].imshow(originals[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_comparison.png") # Save to file
    plt.close()
    print(f"Plots saved as {filename_prefix}_loss_curve.png and {filename_prefix}_comparison.png")

def run_reconstruction_experiment() -> None:
    """
    Implements the first stage of the exercise: Training a convolutional autoencoder.
    
    This function initializes the datasets, builds the encoder/decoder pair, 
    and executes the training loop using Mean L1 Loss for image reconstruction.
    """
    # Hyperparameters
    latent_dim = 16  # Experiment with d=4 and d=16
    first_layer_channels = 15  # Experiment with 3 and 15
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix_filename = f"autoencoder_d{latent_dim}_c{first_layer_channels}"

    # 1. Prepare Data
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # 2. Initialize Models
    encoder = Encoder(latent_dim=latent_dim, first_layer_channels=first_layer_channels).to(device)
    decoder = Decoder(latent_dim=latent_dim, last_layer_channels=first_layer_channels).to(device)

    # 3. Define Loss and Optimizer
    # This`` exercise requires Mean L1 error for reconstruction, change it as needed.
    criterion = nn.L1Loss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # 4. Training Loop
    epoch_losses = []

    print(f"Starting Training: Latent Dim={latent_dim}, Channels={first_layer_channels}")
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

    # --- Final Visualization ---
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Get a small batch from the test loader
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)
        
        # Reconstruct
        latents = encoder(test_images)
        reconstructed = decoder(latents)
        
        # Call the plotting function
        plot_reconstruction_results(epoch_losses, test_images, reconstructed, prefix_filename)

if __name__ == "__main__":
    run_reconstruction_experiment()