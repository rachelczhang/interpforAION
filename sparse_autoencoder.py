import torch
import torch.nn as nn
import torch.nn.functional as F

class TopK(nn.Module):
    """TopK activation function that keeps only the k largest activations (by value) per sample."""
    def __init__(self, k):
        super().__init__()
        self.k = k
    def forward(self, x):
        # x: [batch, hidden]
        # Get indices of top-k values for each sample
        topk_vals, topk_idx = torch.topk(x, self.k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_idx, 1.0)
        out = x * mask
        # Debug: print mean number of nonzero activations per sample (should be k)
        if self.training:
            nonzero_per_sample = (out.abs() > 1e-6).sum(dim=1).float().mean().item()
            print(f"[TopK] Mean nonzero activations per sample: {nonzero_per_sample:.2f} (target k={self.k})")
        return out

class SparseAutoencoder(nn.Module):
    # def __init__(self, input_size, hidden_size):
    def __init__(self, input_size, hidden_size, k=None):
        super(SparseAutoencoder, self).__init__()
        # self.encoder = nn.Linear(input_size, hidden_size)
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)  # No bias in encoder
        self.decoder = nn.Linear(hidden_size, input_size)
        # self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(input_size))  # Pre-subtraction bias
        self.k = k if k is not None else hidden_size // 10  # Default to 10% sparsity
        self.topk = TopK(self.k)
        
        # Initialize encoder weights to decoder transpose to prevent dead latents
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize encoder weights to decoder transpose to prevent dead latents."""
        with torch.no_grad():
            # Initialize decoder weights with normalized random values
            nn.init.normal_(self.decoder.weight, std=0.02)
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=1)
            # Set encoder weights to decoder transpose
            self.encoder.weight.data = self.decoder.weight.data.t()
            
    def encode(self, x):
        # # Subtract decoder bias before encoding
        # x = x - self.decoder.bias
        # return self.relu(self.encoder(x))
        
        # Subtract bias before encoding (as per paper equation 2)
        x = x - self.bias
        # Apply encoder and TopK activation
        encoded = self.topk(self.encoder(x))
        return encoded
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    # def normalize_decoder_weights(self):
    #     """Normalize decoder weights to prevent degenerate solutions"""

    def get_dead_latents(self, threshold=1e-6, activation_counts=None, total_samples=None):
        """Return indices of dead latents (those that never activate)."""
        if activation_counts is None or total_samples is None:
            return torch.tensor([], dtype=torch.long)
        activation_rates = activation_counts / total_samples
        dead_mask = activation_rates < threshold
        return torch.where(dead_mask)[0]
    
    def get_alive_latents(self, threshold=1e-6):
        """Return indices of alive latents (those that do activate)."""
        with torch.no_grad():
            # self.decoder.weight.data = F.normalize(
            #     self.decoder.weight.data, p=2, dim=1
            # )
            max_acts = torch.max(self.encoder.weight, dim=1)[0]
            alive_mask = max_acts >= threshold
            return torch.where(alive_mask)[0]
    
    def get_dead_latent_ratio(self, threshold=1e-6, activation_counts=None, total_samples=None):
        """Return the ratio of dead latents."""
        dead_latents = self.get_dead_latents(threshold, activation_counts, total_samples)
        return len(dead_latents) / self.encoder.weight.shape[0]