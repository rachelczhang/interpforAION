import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import torch.nn as nn

class ModelWrapper:
    """
    A wrapper around the AION model that handles activation replacement properly,
    including residual connections. This ensures that when we replace activations
    at a specific block, the replacement propagates correctly through the network.
    """
    def __init__(self, model, debug=False):
        self.model = model
        self.replacement_activations = None
        self.target_block = None
        self.block_activations = {}  # Store activations from each block
        self.debug = debug
        self.original_block_activations = {}  # Store original activations for verification
        self._original_forward_done = False  # Track if we've done original forward pass
        self.num_blocks = len(model.decoder)
        
    def set_debug(self, debug: bool):
        """Enable or disable debug mode."""
        self.debug = debug
        
    def set_replacement_activations(self, block_num: int, activations: torch.Tensor):
        """
        Set activations to replace at a specific block.
        
        Args:
            block_num: Index of the block to replace activations at
            activations: Tensor of activations to use as replacement
            
        Raises:
            ValueError: If block number is invalid or if we don't have original activations to compare against
        """
        # Validate block number
        if not 0 <= block_num < self.num_blocks:
            raise ValueError(f"Block number must be between 0 and {self.num_blocks-1}, got {block_num}")
            
        # Require that we have original activations to compare against
        if block_num not in self.block_activations:
            raise ValueError(f"Block {block_num} activations not available. Must run forward pass first. Available blocks: {list(self.block_activations.keys())}")
            
        # Verify shape matches the target block's activations
        expected_shape = self.block_activations[block_num].shape
        if activations.shape != expected_shape:
            raise ValueError(f"Replacement activations shape {activations.shape} does not match target block shape {expected_shape}")
            
        # Store target block and activations
        self.target_block = block_num
        self.replacement_activations = activations
        
    def clear_replacement_activations(self):
        """Clear any set replacement activations."""
        self.replacement_activations = None
        self.target_block = None
        self.block_activations.clear()
        self._original_forward_done = False
        # Don't clear original_block_activations here
        
    def _decode(self, encoder_output, encoder_mask, decoder_tokens, decoder_emb, decoder_attention_mask) -> torch.Tensor:
        """
        Custom decode method that handles activation replacement.
        This ensures that when we replace activations at a block, the replacement
        properly propagates through the network via residual connections.
        
        Args:
            encoder_output: Output from the encoder
            encoder_mask: Mask for encoder output
            decoder_tokens: Decoder input tokens
            decoder_emb: Decoder embeddings
            decoder_attention_mask: Attention mask for decoder
        """
        # Only clear current block activations, not original ones
        self.block_activations.clear()
        
        # Start with decoder tokens + embeddings
        current_activations = decoder_tokens + decoder_emb
        
        # Run through each decoder block
        for i, block in enumerate(self.model.decoder):
            # Store activations before the block
            self.block_activations[i] = current_activations.clone()
            
            if self.debug:
                print(f"\nBlock {i} activations before block:")
                print(f"Shape: {current_activations.shape}")
                print(f"Mean: {current_activations.mean().item():.4f}")
                print(f"Std: {current_activations.std().item():.4f}")
                print(f"Min: {current_activations.min().item():.4f}")
                print(f"Max: {current_activations.max().item():.4f}")
            
            # Run the block
            current_activations = block(
                x=current_activations,
                context=encoder_output,
                sa_mask=decoder_attention_mask,
                xa_mask=encoder_mask
            )
            
            # Store activations after the block
            self.block_activations[i] = current_activations.clone()
            
            # For original forward pass, store activations for verification
            if not self._original_forward_done:
                self.original_block_activations[i] = current_activations.clone()
            
            if self.debug:
                print(f"\nBlock {i} activations after block:")
                print(f"Shape: {current_activations.shape}")
                print(f"Mean: {current_activations.mean().item():.4f}")
                print(f"Std: {current_activations.std().item():.4f}")
                print(f"Min: {current_activations.min().item():.4f}")
                print(f"Max: {current_activations.max().item():.4f}")
            
            # Replace activations if this is the target block
            if i == self.target_block and self.replacement_activations is not None:
                if self.debug:
                    print(f"\nReplacing activations at block {i}")
                    print(f"Before replacement - mean: {current_activations.mean().item():.4f}, std: {current_activations.std().item():.4f}")
                    print(f"Replacement - mean: {self.replacement_activations.mean().item():.4f}, std: {self.replacement_activations.std().item():.4f}")
                
                # Verify we're not accidentally modifying other blocks
                if i > 0 and self._original_forward_done:
                    prev_block_diff = (self.block_activations[i-1] - self.original_block_activations[i-1]).abs().max()
                    if prev_block_diff > 1e-6:
                        print(f"WARNING: Block {i-1} activations were modified! Max diff: {prev_block_diff:.6f}")
                
                current_activations = self.replacement_activations
                
                if self.debug:
                    print(f"After replacement - mean: {current_activations.mean().item():.4f}, std: {current_activations.std().item():.4f}")
        
        # Mark original forward pass as done after first pass
        if not self._original_forward_done:
            self._original_forward_done = True
        
        # Apply final decoder normalization layer
        current_activations = self.model.decoder_norm(current_activations)
        if self.debug:
            print(f"\nAfter final normalization:")
            print(f"Mean: {current_activations.mean().item():.4f}")
            print(f"Std: {current_activations.std().item():.4f}")
            print(f"Min: {current_activations.min().item():.4f}")
            print(f"Max: {current_activations.max().item():.4f}")
            
        return current_activations

def calculate_loss_ratio(model, dataloader, autoencoder, device, num_batches=None, block_num=8):
    """
    Calculate the loss ratio to evaluate autoencoder performance using a model wrapper
    that properly handles activation replacement.
    
    Loss Ratio = (L_zero_ablated - L_approximation) / (L_zero_ablated - L_original)
    
    Args:
        model: AION model instance
        dataloader: DataLoader for evaluation
        autoencoder: Trained sparse autoencoder
        device: Device to run computation on
        num_batches: Number of batches to process (None for all batches)
        block_num: Decoder block number to analyze (default: 8)
    
    Returns:
        loss_ratio: Percentage of MLP's contribution captured by autoencoder
        losses: Dictionary containing original, zero_ablated, and approximation losses
    """
    model.eval()
    autoencoder.eval()
    
    # Create model wrapper and enable debug mode
    wrapped_model = ModelWrapper(model)
    wrapped_model.set_debug(True)  # Enable debug mode
    
    losses = {'zero_ablated': 0, 'approximation': 0, 'original': 0}
    num_batches_processed = 0
    total_examples = 0
    
    # def print_logits_stats(logits, prefix=""):
    #     """Helper function to print logits statistics."""
    #     for mod, mod_logits in logits.items():
    #         if mod_logits.size(0) == 0:
    #             continue
    #         print(f"\n{prefix} Logits stats for {mod}:")
    #         print(f"Mean: {mod_logits.mean().item():.4f}")
    #         print(f"Std: {mod_logits.std().item():.4f}")
    #         print(f"Min: {mod_logits.min().item():.4f}")
    #         print(f"Max: {mod_logits.max().item():.4f}")
            
    #         # Print top predictions for first few tokens
    #         if mod_logits.size(0) > 0:
    #             top_preds = torch.topk(mod_logits[:3], k=3, dim=-1)
    #             print(f"Top 3 predictions for first 3 tokens:")
    #             print(f"Values: {top_preds.values}")
    #             print(f"Indices: {top_preds.indices}")
    
    def calculate_mod_loss(logits, target_ids, decoder_mod_mask, model):
        """Calculate modality-wise loss following the original model's implementation."""
        mod_loss = {}
        for mod in logits.keys():
            idx = model.modality_info[mod]["id"]
            mod_logits = logits[mod]
            if mod_logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=mod_logits.device)
            else:
                loss = F.cross_entropy(
                    mod_logits, target_ids[decoder_mod_mask == idx].long(), reduction="mean"
                )
                mod_loss[mod] = loss
        
        # Average loss across modalities
        loss = sum(mod_loss.values()) / len(mod_loss)
        return loss.item(), mod_loss

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
            
        print(f"\n{'='*50}")
        print(f"Processing batch {batch_idx + 1}")
        print(f"{'='*50}")
        
        # Get current batch size
        current_batch_size = next(iter(batch.values()))['tensor'].shape[0]
        total_examples += current_batch_size
        
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

        with torch.no_grad():
            # Get encoder outputs (same for all three cases)
            encoder_tokens, encoder_emb, encoder_mask, _ = model.embed_inputs(
                input_dict, num_encoder_tokens=256
            )
            encoder_output = model._encode(encoder_tokens, encoder_emb, encoder_mask)
            
            # Get decoder embeddings (same for all three cases)
            (
                decoder_tokens,
                decoder_emb,
                decoder_mask,
                target_ids,
                decoder_attention_mask,
                decoder_mod_mask,
            ) = model.embed_targets(target_mask, num_decoder_tokens=128)
            
            print(f"\n{'='*50}")
            print("1. Original Forward Pass")
            print(f"{'='*50}")
            
            # 1. Calculate original loss
            wrapped_model.clear_replacement_activations()
            decoder_output = wrapped_model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Get the original activations from the target block
            original_activations = wrapped_model.block_activations[block_num]
            
            print(f"\nOriginal activations stats for block {block_num}:")
            print(f"Shape: {original_activations.shape}")
            print(f"Mean: {original_activations.mean().item():.4f}")
            print(f"Std: {original_activations.std().item():.4f}")
            print(f"Min: {original_activations.min().item():.4f}")
            print(f"Max: {original_activations.max().item():.4f}")
            
            # Calculate original loss
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
            
            # Calculate loss using modality-wise averaging
            avg_loss, mod_losses = calculate_mod_loss(logits, target_ids, decoder_mod_mask, model)
            losses['original'] += avg_loss
            print(f"\nOriginal batch loss: {avg_loss:.4f}")
            print("Per-modality losses:")
            for mod, loss in mod_losses.items():
                print(f"{mod}: {loss.item():.4f}")
            
            print(f"\n{'='*50}")
            print("2. Zero-ablated Forward Pass")
            print(f"{'='*50}")
            
            # 2. Calculate zero-ablated loss
            zero_activations = torch.zeros_like(original_activations)
            wrapped_model.set_replacement_activations(block_num, zero_activations)
            
            # Verify zero activations
            if not torch.allclose(zero_activations, torch.zeros_like(zero_activations)):
                print("WARNING: Zero activations are not actually zero!")
                print(f"Mean: {zero_activations.mean().item():.4f}")
                print(f"Std: {zero_activations.std().item():.4f}")
                print(f"Min: {zero_activations.min().item():.4f}")
                print(f"Max: {zero_activations.max().item():.4f}")
            
            decoder_output = wrapped_model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Calculate zero-ablated loss
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
            
            # Calculate loss using modality-wise averaging
            avg_loss, mod_losses = calculate_mod_loss(logits, target_ids, decoder_mod_mask, model)
            losses['zero_ablated'] += avg_loss
            print(f"\nZero-ablated batch loss: {avg_loss:.4f}")
            print("Per-modality losses:")
            for mod, loss in mod_losses.items():
                print(f"{mod}: {loss.item():.4f}")
            
            print(f"\n{'='*50}")
            print("3. Approximation Forward Pass")
            print(f"{'='*50}")
            
            # 3. Calculate approximation loss
            # Flatten activations for autoencoder
            flat_activations = original_activations.reshape(-1, original_activations.shape[-1])
            reconstructed, _ = autoencoder(flat_activations)
            
            print(f"\nSAE reconstruction stats for block {block_num}:")
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
            
            approximated_activations = reconstructed.reshape(original_activations.shape)
            wrapped_model.set_replacement_activations(block_num, approximated_activations)
            
            decoder_output = wrapped_model._decode(
                encoder_output,
                encoder_mask,
                decoder_tokens,
                decoder_emb,
                decoder_attention_mask,
            )
            
            # Calculate approximation loss
            logits = {}
            for mod in target_mask.keys():
                idx = model.modality_info[mod]["id"]
                mod_logits = model.decoder_embeddings[mod].forward_logits(
                    decoder_output[decoder_mod_mask == idx]
                )
                logits[mod] = mod_logits
            
            # Calculate loss using modality-wise averaging
            avg_loss, mod_losses = calculate_mod_loss(logits, target_ids, decoder_mod_mask, model)
            losses['approximation'] += avg_loss
            print(f"\nApproximated batch loss: {avg_loss:.4f}")
            print("Per-modality losses:")
            for mod, loss in mod_losses.items():
                print(f"{mod}: {loss.item():.4f}")
        
        num_batches_processed += 1
        print(f"\n{'='*50}")
        print(f"Completed batch {batch_idx + 1}")
        print(f"Running averages:")
        print(f"  Original loss: {losses['original'] / num_batches_processed:.4f}")
        print(f"  Zero-ablated loss: {losses['zero_ablated'] / num_batches_processed:.4f}")
        print(f"  Approximation loss: {losses['approximation'] / num_batches_processed:.4f}")
        print(f"{'='*50}")

    # Average losses across all batches
    for key in losses:
        losses[key] /= num_batches_processed
    
    # Calculate loss ratio
    mlp_contribution = losses['zero_ablated'] - losses['original']
    autoencoder_contribution = losses['zero_ablated'] - losses['approximation']
    loss_ratio = (autoencoder_contribution / mlp_contribution) * 100 if mlp_contribution != 0 else 0
    
    print("\nFinal Loss Analysis:")
    print(f"Original Loss: {losses['original']:.4f}")
    print(f"Zero-ablated Loss: {losses['zero_ablated']:.4f}")
    print(f"Approximation Loss: {losses['approximation']:.4f}")
    print(f"MLP's contribution to loss reduction: {mlp_contribution:.4f}")
    print(f"Autoencoder's contribution to loss reduction: {autoencoder_contribution:.4f}")
    print(f"\nLoss Ratio: {loss_ratio:.2f}% of MLP's contribution captured by autoencoder")
    
    return loss_ratio, losses 