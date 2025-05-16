import torch
import torch.nn as nn
from sparse_autoencoder import SparseAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import wandb

def train_sae(activations_tensor_flat, autoencoder, device):
    # Create dataset and dataloader
    dataset = TensorDataset(activations_tensor_flat)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    sparsity_lambda = 1e-3
    patience = 70
    best_loss = float('inf')
    epochs_without_improvement = 0

    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        total_reconstruction_loss = 0
        total_sparsity_loss = 0
        total_l0_loss = 0

        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)

            # Compute losses
            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = torch.mean(torch.abs(encoded))
            loss = reconstruction_loss + sparsity_lambda * sparsity_loss
            l0_loss = torch.mean((torch.abs(encoded) > 1e-12).float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights after optimization step
            autoencoder.normalize_decoder_weights()

            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_l0_loss += l0_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_reconstruction_loss = total_reconstruction_loss / len(dataloader)
        avg_sparsity_loss = total_sparsity_loss / len(dataloader)
        avg_l0_loss = total_l0_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Reconstruction Loss: {avg_reconstruction_loss:.6f}, Sparsity Loss: {avg_sparsity_loss:.6f}, L0 Loss: {avg_l0_loss:.6f}', flush=True)
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "reconstruction_loss": avg_reconstruction_loss, "sparsity_loss": avg_sparsity_loss, "l0_loss": avg_l0_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(autoencoder.state_dict(), f"best_llm_sae_{wandb.run.name}.pth")  # Save model parameters
            print(f"New best model saved with loss {best_loss:.6f}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Stopping early after {epoch+1} epochs due to no improvement.")
            break

# Initialize wandb
wandb.init(project="interp-for-aion", entity="rczhang")

# Set device (use CUDA if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read in activations
activations_path = '/mnt/home/rzhang/ceph/activations_4992examples.pt'
activations_tensor_flat = torch.load(activations_path, map_location=device, weights_only=True)
print(f"Original activations tensor shape: {activations_tensor_flat.shape}")

# Preprocess activations according to the paper:
# 1. Subtract mean over the dmodel dimension
# 2. Normalize to unit norm
print("\nPreprocessing activations...")
# Subtract mean over dmodel dimension (dim=1 since shape is [batch, dmodel])
activations_tensor_flat = activations_tensor_flat - activations_tensor_flat.mean(dim=1, keepdim=True)
print(f"After mean subtraction - mean: {activations_tensor_flat.mean().item():.6f}, std: {activations_tensor_flat.std().item():.6f}")

# Normalize to unit norm
norms = torch.norm(activations_tensor_flat, p=2, dim=1, keepdim=True)
activations_tensor_flat = activations_tensor_flat / norms
print(f"After normalization - mean: {activations_tensor_flat.mean().item():.6f}, std: {activations_tensor_flat.std().item():.6f}")
print(f"Norm of each vector: {torch.norm(activations_tensor_flat, p=2, dim=1).mean().item():.6f} (should be 1.0)")

# Define the sparse autoencoder
input_size = activations_tensor_flat.shape[1]
hidden_size = input_size*4
autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")

# Train the sparse autoencoder
train_sae(activations_tensor_flat, autoencoder, device)