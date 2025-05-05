import torch
import torch.nn as nn
from sparse_autoencoder import SparseAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import wandb

def train_sae(activations_tensor_flat, autoencoder, device):
    # Create dataset and dataloader
    dataset = TensorDataset(activations_tensor_flat)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    sparsity_lambda = 1e-2
    patience = 300
    best_loss = float('inf')
    epochs_without_improvement = 0

    num_epochs = 10000
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)

            # Compute losses
            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = torch.mean(torch.abs(encoded))
            loss = reconstruction_loss + sparsity_lambda * sparsity_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights after optimization step
            autoencoder.normalize_decoder_weights()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Reconstruction Loss: {reconstruction_loss.item():.6f}, Sparsity Loss: {sparsity_loss.item():.6f}', flush=True)
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "reconstruction_loss": reconstruction_loss.item(), "sparsity_loss": sparsity_loss.item()})

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
print(f"Activations tensor shape: {activations_tensor_flat.shape}")

# Define the sparse autoencoder
input_size = activations_tensor_flat.shape[1]
hidden_size = input_size*4
autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")

# Train the sparse autoencoder
train_sae(activations_tensor_flat, autoencoder, device)