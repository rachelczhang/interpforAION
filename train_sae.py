import torch
import torch.nn as nn
from sparse_autoencoder import SparseAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt

def train_sae(activations_tensor_flat, autoencoder, device):
    # Calculate baseline MSE (predicting mean activations)
    mean_activations = activations_tensor_flat.mean(dim=0, keepdim=True)  # [1, dmodel]
    baseline_mse = torch.mean((activations_tensor_flat - mean_activations) ** 2).item()
    print(f"\nBaseline MSE (predicting mean): {baseline_mse:.6f}")

    # Create dataset and dataloader
    dataset = TensorDataset(activations_tensor_flat)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    # sparsity_lambda = 1e-3
    patience = 70
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    # Parameters for dead latent auxiliary loss
    k_aux = autoencoder.k * 2  # Number of top-k dead latents to use for auxiliary loss
    aux_lambda = 0.1  # Weight for auxiliary loss

    num_epochs = 100
    activation_counts = np.zeros(autoencoder.encoder.out_features)
    total_samples = 0
    for epoch in range(num_epochs):
        total_loss = 0
        total_reconstruction_loss = 0
        # total_sparsity_loss = 0
        total_aux_loss = 0
        total_l0_loss = 0

        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)

            # Compute main reconstruction loss
            reconstruction_loss = criterion(reconstructed, batch)
            
            normalized_reconstruction_loss = reconstruction_loss / baseline_mse
            # sparsity_loss = torch.mean(torch.abs(encoded))
            # loss = normalized_reconstruction_loss + sparsity_lambda * sparsity_loss
            
            # Compute auxiliary loss using top-k dead latents
            with torch.no_grad():
                # Get indices of dead latents
                dead_latents = autoencoder.get_dead_latents()
                if len(dead_latents) > 0:
                    # Get the k_aux largest dead latents by their maximum activation
                    max_acts = torch.max(autoencoder.encoder.weight[dead_latents], dim=1)[0]
                    _, indices = torch.topk(max_acts, min(k_aux, len(dead_latents)))
                    top_k_dead = dead_latents[indices]
                else:
                    top_k_dead = torch.tensor([], device=device, dtype=torch.long)

            # If we have dead latents, compute auxiliary loss
            if len(top_k_dead) > 0:
                # Get activations for dead latents
                dead_acts = encoded[:, top_k_dead]
                # Compute reconstruction using only dead latents
                dead_reconstructed = autoencoder.decoder(dead_acts)
                aux_loss = criterion(dead_reconstructed, batch)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # Total loss is reconstruction loss plus auxiliary loss
            loss = normalized_reconstruction_loss + aux_lambda * aux_loss

            # Compute L0 loss for monitoring (not used in training)
            l0_loss = torch.mean((torch.abs(encoded) > 1e-12).float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Normalize decoder weights after optimization step
            # autoencoder.normalize_decoder_weights()

            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            # total_sparsity_loss += sparsity_loss.item() 
            total_aux_loss += aux_loss.item()
            total_l0_loss += l0_loss.item()

            # Track activation frequency for this batch
            with torch.no_grad():
                # encoded: [batch_size, hidden_size]
                # Count nonzero activations (i.e., in top-k)
                activation_counts += (encoded.cpu().numpy() > 1e-12).sum(axis=0)
                total_samples += encoded.shape[0]

        avg_loss = total_loss / len(dataloader)
        avg_reconstruction_loss = total_reconstruction_loss / len(dataloader)
        avg_normalized_reconstruction_loss = avg_reconstruction_loss / baseline_mse
        # avg_sparsity_loss = total_sparsity_loss / len(dataloader)
        avg_aux_loss = total_aux_loss / len(dataloader)
        avg_l0_loss = total_l0_loss / len(dataloader)
        
        # Get dead latent ratio
        dead_ratio = autoencoder.get_dead_latent_ratio()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, '
              f'Reconstruction Loss: {avg_reconstruction_loss:.6f}, '
              f'Normalized Reconstruction Loss: {avg_normalized_reconstruction_loss:.6f}, '
            #   f'Sparsity Loss: {avg_sparsity_loss:.6f}, '
            #   f'L0 Loss: {avg_l0_loss:.6f}', flush=True)
              f'Auxiliary Loss: {avg_aux_loss:.6f}, '
              f'L0 Loss: {avg_l0_loss:.6f}, '
              f'Dead Latent Ratio: {dead_ratio:.2%}', flush=True)
        wandb.log({
            "epoch": epoch+1, 
            "loss": avg_loss, 
            "reconstruction_loss": avg_reconstruction_loss,
            "normalized_reconstruction_loss": avg_normalized_reconstruction_loss,
            # "sparsity_loss": avg_sparsity_loss, 
            # "l0_loss": avg_l0_loss
            "auxiliary_loss": avg_aux_loss,
            "l0_loss": avg_l0_loss,
            "dead_latent_ratio": dead_ratio
        })

        # Log/plot activation frequency every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            freq = activation_counts / total_samples  # Fraction of samples each latent is active
            # Log histogram to wandb
            wandb.log({
                "latent_activation_frequency": wandb.Histogram(freq),
                "epoch": epoch + 1
            })
            # Optionally, also plot and log as an image
            plt.figure(figsize=(8, 4))
            plt.hist(freq, bins=50, color='blue', alpha=0.7)
            plt.xlabel('Activation Frequency')
            plt.ylabel('Number of Latents')
            plt.title(f'Latent Activation Frequency (Epoch {epoch+1})')
            plt.tight_layout()
            plt.savefig('latent_activation_frequency.png')
            plt.close()
            wandb.log({"latent_activation_frequency_plot": wandb.Image('latent_activation_frequency.png'), "epoch": epoch + 1})
            # Reset counts for next interval
            activation_counts[:] = 0
            total_samples = 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(autoencoder.state_dict(), f"best_llm_sae_{wandb.run.name}.pth")
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
# autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
# print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")
# k = hidden_size // 10  # Use 10% sparsity by default
k = max(1, int(hidden_size * 0.02))  # Use 2% sparsity by default
autoencoder = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}, k: {k}")

# Train the sparse autoencoder
train_sae(activations_tensor_flat, autoencoder, device)