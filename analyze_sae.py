import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sparse_autoencoder import SparseAutoencoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import sys
from save_activations import get_data
import torch.nn.functional as F
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add both directories to sys.path
sys.path.append(os.path.join(current_dir, "AION"))
sys.path.append(os.path.join(current_dir, "4Mclone"))

from aion import AION

class Args:
    """Simple class to hold arguments needed for setup functions."""
    def __init__(self):
        self.data_config = "4Mclone/cfgs/default/mmoma/data/mmu/rusty_legacysurvey_desi_sdss_hsc_eval.yaml"
        self.text_tokenizer_path = "4Mclone/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
        self.input_size = 224
        self.patch_size = 16
        self.num_input_tokens = 128
        self.num_target_tokens = 128
        self.min_input_tokens = None
        self.min_target_tokens = None
        self.batch_size = 64
        self.epoch_size = 128
        self.num_workers = 1
        self.num_tasks = 1
        self.pin_mem = True
        self.dist_eval = False
        self.fixed_eval = False
        self.fixed_eval_input_tokens = None
        self.fixed_eval_target_tokens = None
        self.in_domains = None
        self.out_domains = None
        self.all_domains = None

def select_tokens_for_modality(targets, target_mask, num_decoder_tokens):
    """
    Select tokens using the same process as the model's forward_mask_decoder.
    
    Args:
        targets: Tensor of target tokens [batch_size, seq_length]
        target_mask: Boolean mask indicating which tokens should be predicted [batch_size, seq_length]
        num_decoder_tokens: Maximum number of tokens to select per batch
        
    Returns:
        selected_targets: Selected target tokens, flattened across batch dimension
        selected_mask: Boolean mask indicating which selected tokens are valid, flattened across batch dimension
    """
    batch_size = targets.shape[0]
    
    # Add small constant to mask for deterministic sorting
    mask_arange = torch.arange(target_mask.shape[1], device=target_mask.device).unsqueeze(0) * 1e-6
    ids_shuffle = torch.argsort(target_mask + mask_arange, dim=1)
    ids_keep = ids_shuffle[:, :num_decoder_tokens]
    
    # Gather selected tokens and their masks
    selected_targets = torch.gather(targets, dim=1, index=ids_keep)  # [batch_size, num_decoder_tokens]
    selected_mask = torch.gather(target_mask, dim=1, index=ids_keep)  # [batch_size, num_decoder_tokens]
    
    # Flatten across batch dimension
    selected_targets = selected_targets.reshape(-1)  # [batch_size * num_decoder_tokens]
    selected_mask = selected_mask.reshape(-1)  # [batch_size * num_decoder_tokens]
    
    return selected_targets, selected_mask

def calculate_loss_ratio(model, data_loader, autoencoder, device):
    """
    Calculate the loss ratio to evaluate autoencoder performance.
    
    Loss Ratio = (L_zero_ablated - L_approximation) / (L_zero_ablated - L_original)
    
    For each batch:
    1. Get original activations from 9th decoder block
    2. Calculate original loss (L_original)
    3. Calculate loss with zero-ablated activations (L_zero_ablated)
    4. Calculate loss with autoencoder approximated activations (L_approximation)
    
    The model uses masked multimodal modeling where:
    - Input tokens are selected from various modalities up to a budget (256 tokens)
    - Target tokens are selected from remaining tokens up to a budget (128 tokens)
    - The target mask indicates which tokens should be predicted
    - The model's forward pass returns logits for each modality, filtered to only include tokens that should be predicted
    """
    losses = {'zero_ablated': 0, 'approximation': 0, 'original': 0}
    num_batches = 0
    total_examples = 0

    # Get the 9th decoder block
    layer = model.decoder[8]
    
    # Hook for storing original activations
    original_activations = {}
    def store_hook(module, input, output):
        original_activations['decoder_block_9'] = output.clone()
        print(f"\nOriginal activations stats:")
        print(f"Shape: {output.shape}")
        print(f"Mean: {output.mean().item():.4f}")
        print(f"Std: {output.std().item():.4f}")
        print(f"Min: {output.min().item():.4f}")
        print(f"Max: {output.max().item():.4f}")
    
    # Hook for modifying activations
    def modify_hook(new_activations):
        def hook(module, input, output):
            print(f"\nModified activations stats:")
            print(f"Shape: {new_activations.shape}")
            print(f"Mean: {new_activations.mean().item():.4f}")
            print(f"Std: {new_activations.std().item():.4f}")
            print(f"Min: {new_activations.min().item():.4f}")
            print(f"Max: {new_activations.max().item():.4f}")
            return new_activations
        return hook

    for batch_idx, batch in enumerate(data_loader):
        # Get current batch size
        current_batch_size = next(iter(batch.values()))['tensor'].shape[0]
        total_examples += current_batch_size
        print(f"\nProcessing batch {batch_idx + 1} with {current_batch_size} examples")
        
        # Move batch to device
        processed_data = {}
        for mod, d in batch.items():
            processed_data[mod] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in d.items()}
        
        # Prepare input dictionary and target mask
        input_dict = {}
        target_mask = {}
        for mod, d in processed_data.items():
            if mod in model.encoder_embeddings:
                input_dict[mod] = d['tensor']
            if mod in model.decoder_embeddings:
                target_mask[mod] = d['target_mask']

        # Calculate original loss
        with torch.no_grad():
            # Store original activations
            store_handle = layer.register_forward_hook(store_hook)
            
            # Get model outputs including decoder_mod_mask
            encoder_tokens, encoder_emb, encoder_mask, _ = model.embed_inputs(
                input_dict, num_encoder_tokens=256
            )
            (
                decoder_tokens,
                decoder_emb,
                decoder_mask,
                target_ids,
                decoder_attention_mask,
                decoder_mod_mask,
            ) = model.embed_targets(target_mask, num_decoder_tokens=128)
            
            print(f"\nInput shapes:")
            print(f"Encoder tokens: {encoder_tokens.shape}")
            print(f"Decoder tokens: {decoder_tokens.shape}")
            print(f"Target IDs: {target_ids.shape}")
            print(f"Decoder mod mask: {decoder_mod_mask.shape}")
            
            # Run the encoder and decoder
            encoder_output = model._encode(encoder_tokens, encoder_emb, encoder_mask)
            decoder_output = model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Get logits for each modality
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
                print(f"\nModality {mod} original:")
                print(f"Logits shape: {mod_logits.shape}")
                print(f"Logits mean: {mod_logits.mean().item():.4f}")
                print(f"Logits std: {mod_logits.std().item():.4f}")
                
                # Print top predictions for first few tokens
                if mod_logits.size(0) > 0:
                    top_preds = torch.topk(mod_logits[:5], k=3, dim=-1)
                    print(f"Top 3 predictions for first 5 tokens:")
                    print(f"Values: {top_preds.values}")
                    print(f"Indices: {top_preds.indices}")
            
            # Calculate loss for each modality
            total_loss = 0
            num_modalities = 0
            for mod, mod_logits in logits.items():
                # Get the target tokens and mask for this modality
                targets = processed_data[mod]['tensor']
                mod_target_mask = target_mask[mod]
                
                # Skip if either logits or targets are empty
                if mod_logits.size(0) == 0 or targets.size(0) == 0:
                    continue
                
                # Debug prints to understand shapes
                print(f"\nModality: {mod}")
                print(f"Targets shape: {targets.shape}")
                print(f"Target mask shape: {mod_target_mask.shape}")
                print(f"Logits shape: {mod_logits.shape}")
                
                # Get the target tokens that correspond to the selected logits
                # We use decoder_mod_mask to find which tokens were selected
                idx = model.modality_info[mod]["id"]
                selected_targets = target_ids[decoder_mod_mask == idx]
                
                print(f"\nModality {mod} targets:")
                print(f"Selected targets shape: {selected_targets.shape}")
                print(f"First 5 targets: {selected_targets[:5]}")
                
                # Calculate cross entropy loss
                loss = F.cross_entropy(mod_logits, selected_targets.long(), reduction='mean')
                print(f"Loss: {loss.item():.4f}")
                
                total_loss += loss
                num_modalities += 1
            
            if num_modalities > 0:
                avg_loss = total_loss.item() / num_modalities
                losses['original'] += avg_loss
                print(f"\nOriginal batch loss: {avg_loss:.4f}")
            store_handle.remove()

        # Calculate zero-ablated loss
        with torch.no_grad():
            zero_activations = torch.zeros_like(original_activations['decoder_block_9'])
            zero_handle = layer.register_forward_hook(modify_hook(zero_activations))
            
            # Run model with zero-ablated activations
            decoder_output = model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Get logits for each modality
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
                print(f"\nModality {mod} zero-ablated:")
                print(f"Logits shape: {mod_logits.shape}")
                print(f"Logits mean: {mod_logits.mean().item():.4f}")
                print(f"Logits std: {mod_logits.std().item():.4f}")
                
                # Print top predictions for first few tokens
                if mod_logits.size(0) > 0:
                    top_preds = torch.topk(mod_logits[:5], k=3, dim=-1)
                    print(f"Top 3 predictions for first 5 tokens:")
                    print(f"Values: {top_preds.values}")
                    print(f"Indices: {top_preds.indices}")
            
            # Calculate loss for each modality
            total_loss = 0
            num_modalities = 0
            for mod, mod_logits in logits.items():
                if mod_logits.size(0) == 0:
                    continue
                
                idx = model.modality_info[mod]["id"]
                selected_targets = target_ids[decoder_mod_mask == idx]
                
                loss = F.cross_entropy(mod_logits, selected_targets.long(), reduction='mean')
                print(f"Zero-ablated modality loss: {loss.item():.4f}")
                
                total_loss += loss
                num_modalities += 1
            
            if num_modalities > 0:
                avg_loss = total_loss.item() / num_modalities
                losses['zero_ablated'] += avg_loss
                print(f"\nZero-ablated batch loss: {avg_loss:.4f}")
            zero_handle.remove()

        # Calculate approximation loss
        with torch.no_grad():
            # Flatten activations for autoencoder
            flat_activations = original_activations['decoder_block_9'].reshape(-1, original_activations['decoder_block_9'].shape[-1])
            reconstructed, _ = autoencoder(flat_activations)
            
            print(f"\nSAE reconstruction stats:")
            print(f"Original activations mean: {flat_activations.mean().item():.4f}")
            print(f"Reconstructed activations mean: {reconstructed.mean().item():.4f}")
            print(f"Reconstruction error: {F.mse_loss(reconstructed, flat_activations).item():.4f}")
            
            # Print distribution of reconstruction errors
            errors = (reconstructed - flat_activations).abs()
            print(f"Reconstruction error stats:")
            print(f"Mean error: {errors.mean().item():.4f}")
            print(f"Std error: {errors.std().item():.4f}")
            print(f"Max error: {errors.max().item():.4f}")
            
            # Print correlation between original and reconstructed activations
            orig_flat = flat_activations.reshape(-1)
            recon_flat = reconstructed.reshape(-1)
            correlation = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0,1]
            print(f"Correlation between original and reconstructed: {correlation.item():.4f}")
            
            approximated_activations = reconstructed.reshape(original_activations['decoder_block_9'].shape)
            approx_handle = layer.register_forward_hook(modify_hook(approximated_activations))
            
            # Run model with approximated activations
            decoder_output = model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Get logits for each modality
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
                print(f"\nModality {mod} approximated:")
                print(f"Logits shape: {mod_logits.shape}")
                print(f"Logits mean: {mod_logits.mean().item():.4f}")
                print(f"Logits std: {mod_logits.std().item():.4f}")
                
                # Print top predictions for first few tokens
                if mod_logits.size(0) > 0:
                    top_preds = torch.topk(mod_logits[:5], k=3, dim=-1)
                    print(f"Top 3 predictions for first 5 tokens:")
                    print(f"Values: {top_preds.values}")
                    print(f"Indices: {top_preds.indices}")
            
            # Calculate loss for each modality
            total_loss = 0
            num_modalities = 0
            for mod, mod_logits in logits.items():
                if mod_logits.size(0) == 0:
                    continue
                
                idx = model.modality_info[mod]["id"]
                selected_targets = target_ids[decoder_mod_mask == idx]
                
                loss = F.cross_entropy(mod_logits, selected_targets.long(), reduction='mean')
                print(f"Approximated modality loss: {loss.item():.4f}")
                
                total_loss += loss
                num_modalities += 1
            
            if num_modalities > 0:
                avg_loss = total_loss.item() / num_modalities
                losses['approximation'] += avg_loss
                print(f"\nApproximated batch loss: {avg_loss:.4f}")
            approx_handle.remove()
        
        num_batches += 1
        print(f"\nCompleted batch {batch_idx + 1}")
        print(f"Running averages:")
        print(f"  Original loss: {losses['original'] / num_batches:.4f}")
        print(f"  Zero-ablated loss: {losses['zero_ablated'] / num_batches:.4f}")
        print(f"  Approximation loss: {losses['approximation'] / num_batches:.4f}")

    # Average losses across all batches
    for key in losses:
        losses[key] /= num_batches
    
    # Calculate loss ratio
    mlp_contribution = losses['zero_ablated'] - losses['original']
    autoencoder_contribution = losses['zero_ablated'] - losses['approximation']
    loss_ratio = (autoencoder_contribution / mlp_contribution) * 100
    
    print("\nFinal Loss Analysis:")
    print(f"Original Loss: {losses['original']:.4f}")
    print(f"Zero-ablated Loss: {losses['zero_ablated']:.4f}")
    print(f"Approximation Loss: {losses['approximation']:.4f}")
    print(f"MLP's contribution to loss reduction: {mlp_contribution:.4f}")
    print(f"Autoencoder's contribution to loss reduction: {autoencoder_contribution:.4f}")
    print(f"\nLoss Ratio: {loss_ratio:.2f}% of MLP's contribution captured by autoencoder")
    
    return loss_ratio, losses

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with trained parameters
    print("\n=== Loading Model ===")
    model = AION.from_pretrained('/mnt/ceph/users/polymathic/aion/dec24/base')
    print(f"Model loaded from pretrained weights")
    model.freeze_encoder()
    model.freeze_decoder()
    model = model.to(device).eval()  
    print(f"Model architecture:\n{model}")

    # Define the sparse autoencoder
    input_size = 768
    hidden_size = input_size*4
    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    autoencoder.load_state_dict(torch.load('best_llm_sae_fancy-snowflake-9.pth', weights_only=True, map_location=device))
    autoencoder.eval()
    print(f"Autoencoder loaded")

    # # Read in activations
    # activations_path = '/mnt/home/rzhang/ceph/activations_4992examples.pt'
    # activations_tensor_flat = torch.load(activations_path, map_location=device, weights_only=True)

    # Load tokenized training data
    print("\n=== Setting Up Data ===")

    args = Args()
    data_loader_train = get_data(args)

    # for batch_idx, batch in enumerate(data_loader_train):
    #     print(f"Processing batch {batch_idx + 1}")
    #     current_batch_size = next(iter(batch.values()))['tensor'].shape[0]
    #     print(f"Processing full batch of {current_batch_size} examples")
    #     for mod, d in batch.items():
    #         print(f"Modality: {mod}")
    #         print(f"Shape: {d['tensor'].shape}")
    #         print(f"Target mask shape: {d['target_mask'].shape}")

    loss_ratio, losses = calculate_loss_ratio(model, data_loader_train, autoencoder, device)
    print(f"Loss ratio: {loss_ratio:.2f}%")
    print(f"Losses: {losses}")

