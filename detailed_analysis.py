import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import numpy as np
import pandas as pd
import heapq
from pathlib import Path
from tqdm import tqdm


# USE FORWARD_MASK_DECODER FROM 4MCLONE fm.py instead of embed_targets everything in fourm is useful 

def transform_aion_data_to_training_format(batch, model):
    """
    Transform AION data format to the training format expected by fm.py functions.
    
    The fm.py functions expect:
    - 'tensor': the raw data (keep as is)
    - 'x': embeddings computed from tensor
    - 'emb': positional embeddings 
    - 'ids': token ids (MUST keep as is - required by cat_decoder_tensors!)
    - 'input_mask', 'target_mask': masks (keep as is)
    - 'decoder_attention_mask': for decoder (create if missing)
    """
    transformed_batch = {}
    
    for mod, d in batch.items():
        transformed_d = {}
        
        # CRITICAL: Always preserve original keys first
        for key in ['tensor', 'ids', 'input_mask', 'target_mask', 'decoder_attention_mask']:
            if key in d:
                transformed_d[key] = d[key]
        
        # Get device and shape info
        if 'tensor' in d:
            batch_size, seq_len = d['tensor'].shape[:2]
            device = d['tensor'].device
        elif 'ids' in d:
            batch_size, seq_len = d['ids'].shape[:2]
            device = d['ids'].device
        else:
            continue  # Skip if no data
            
        # For fm.py, we need to use the embedding layers to compute 'x' and 'emb'
        try:
            if mod in model.encoder_embeddings:
                # For encoder modalities, use the encoder embedding
                embed_result = model.encoder_embeddings[mod]({
                    'tensor': d['tensor'],
                    'input_mask': d.get('input_mask', torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device))
                })
                
                # The embedding returns a dict with 'x' and 'emb'
                if isinstance(embed_result, dict):
                    # Only add 'x' and 'emb', don't overwrite existing keys
                    for key in ['x', 'emb']:
                        if key in embed_result:
                            transformed_d[key] = embed_result[key]
                    
            if mod in model.decoder_embeddings:
                # For decoder modalities, use the decoder embedding  
                # The decoder embedding expects 'tensor' and 'target_mask'
                decoder_input = {
                    'tensor': d['tensor'],
                    'target_mask': d.get('target_mask', torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device))
                }
                
                # Add decoder_attention_mask if not present
                if 'decoder_attention_mask' not in d:
                    decoder_input['decoder_attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
                else:
                    decoder_input['decoder_attention_mask'] = d['decoder_attention_mask']
                
                embed_result = model.decoder_embeddings[mod].forward_embed(decoder_input)
                
                # The embedding returns a dict with 'x', 'emb', and 'ids'
                if isinstance(embed_result, dict):
                    # Add all keys from embed_result, but don't overwrite original keys
                    for key in embed_result.keys():
                        if key not in transformed_d:  # Don't overwrite existing keys
                            transformed_d[key] = embed_result[key]
                    
        except Exception as e:
            print(f"Warning: Could not get embeddings for {mod}: {e}")
            # Fallback: create simple 'x' and 'emb'
            transformed_d['x'] = d.get('tensor', d.get('ids', torch.zeros(batch_size, seq_len, device=device)))
            transformed_d['emb'] = torch.zeros(batch_size, seq_len, model.dim, device=device)
            # For decoder modalities, we need 'ids' - use tensor if available
            if mod in model.decoder_embeddings and 'ids' not in transformed_d:
                transformed_d['ids'] = d.get('tensor', torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        
        # Ensure 'decoder_attention_mask' exists for decoder modalities
        if mod in model.decoder_embeddings and 'decoder_attention_mask' not in transformed_d:
            transformed_d['decoder_attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Verify we have the required keys for decoder modalities
        if mod in model.decoder_embeddings:
            required_keys = ['ids', 'x', 'emb', 'target_mask', 'decoder_attention_mask']
            missing_keys = [key for key in required_keys if key not in transformed_d]
            if missing_keys:
                print(f"ERROR: Decoder modality {mod} missing keys: {missing_keys}")
                return None
        
        transformed_batch[mod] = transformed_d
    
    return transformed_batch

def collect_activations(model, data_loader, device, args):
    """
    Returns:
    1. a full activations tensor flattened containing the activations per token
    2. a modality labels tensor flattened containing the modality per token
    3. a raw values tensor flattened containing the raw original token value
    """
    model = model.to(device).eval()

    activations = []
    modality_labels = []
    raw_values = []
    examples_processed = 0

    # Register a hook on the 9-th decoder block (index 8)
    def _hook(_module, _inp, output):
        # Store hidden states on CPU to save GPU memory
        activations.append(output.detach().cpu())

    hook_handle = model.decoder[8].register_forward_hook(_hook)

    # Iterate over batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch tensors to the correct device
            processed_batch = {
                mod: {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in d.items()
                }
                for mod, d in batch.items()
            }

            # Transform AION data format to training format expected by fm.py
            training_format_batch = transform_aion_data_to_training_format(processed_batch, model)
            if training_format_batch is None:
                print(f"Skipping batch {batch_idx} due to transformation error")
                continue

            # Use PURE TRAINING code from fm.py - this gives us proper target_ids!
            
            # Encoder side - use forward_mask_encoder (training function)
            enc_tokens, enc_emb, enc_mask, enc_mod_mask = model.forward_mask_encoder(
                training_format_batch,
                num_encoder_tokens=args.num_input_tokens,
            )

            # Decoder side - use forward_mask_decoder (training function) 
            # This properly extracts target_ids from the original tensor values!
            (
                dec_tokens,
                dec_emb,
                dec_mask,
                target_ids,  # These are now actual values, not zeros!
                dec_att_mask,
                dec_mod_mask,
            ) = model.forward_mask_decoder(training_format_batch, num_decoder_tokens=args.num_target_tokens)

            # Forward pass: Use pure training functions
            context = model.forward_encoder(enc_tokens + enc_emb, enc_mask)

            _ = model.forward_decoder(
                dec_tokens + dec_emb,  # Use training tokens and embeddings
                context,
                enc_mask,
                dec_att_mask,  # Use training attention mask
            )  # hook fires here

            # Store ground-truth modality labels for this batch
            modality_labels.append(dec_mod_mask.cpu())
            raw_values.append(target_ids.cpu())
            
            # House-keeping
            batch_size = next(iter(processed_batch.values()))["tensor"].shape[0]
            examples_processed += batch_size

            torch.cuda.empty_cache()

    # Cleanup hook
    hook_handle.remove()

    # Stack and flatten tensors
    print(f"Processing {len(activations)} batches...")
    
    activations_tensor = torch.cat(activations, dim=0)           # (N, M, D)
    modality_labels_tensor = torch.cat(modality_labels, dim=0)   # (N, M)
    raw_values_tensor = torch.cat(raw_values, dim=0)             # (N, M)

    assert activations_tensor.shape[:2] == modality_labels_tensor.shape, (
        "Decoder activations and modality labels mis-aligned: "
        f"{activations_tensor.shape[:2]} vs {modality_labels_tensor.shape}"
    )

    assert activations_tensor.shape[:2] == raw_values_tensor.shape, (
        "Decoder activations and raw values mis-aligned: "
        f"{activations_tensor.shape[:2]} vs {raw_values_tensor.shape}"
    )

    N, M, D = activations_tensor.shape
    print(f"Collected {N} samples, {M} tokens per sample, {D} hidden dimensions")
    
    activations_tensor_flat = activations_tensor.reshape(N * M, D)
    modality_labels_flat = modality_labels_tensor.reshape(-1)
    raw_values_flat = raw_values_tensor.reshape(-1)

    print(f"\nShapes after flattening:")
    print(f"  activations_tensor_flat: {activations_tensor_flat.shape}")
    print(f"  modality_labels_flat: {modality_labels_flat.shape}")
    print(f"  raw_values_flat: {raw_values_flat.shape}")
    
    # Verify all flattened tensors have same length
    assert len(activations_tensor_flat) == len(modality_labels_flat) == len(raw_values_flat), (
        f"Flattened tensors have different lengths: "
        f"activations={len(activations_tensor_flat)}, "
        f"modality_labels={len(modality_labels_flat)}, "
        f"raw_values={len(raw_values_flat)}"
    )
    
    print(f"Total flattened tokens: {len(activations_tensor_flat)}")
    
    # Basic statistics
    non_zero_raw = (raw_values_flat != 0).sum().item()
    valid_positions = (modality_labels_flat != -1).sum().item()
    
    print(f"Non-zero target values: {non_zero_raw}/{len(raw_values_flat)} ({non_zero_raw/len(raw_values_flat)*100:.1f}%)")
    print(f"Valid positions (non-padding): {valid_positions}/{len(modality_labels_flat)} ({valid_positions/len(modality_labels_flat)*100:.1f}%)")
    
    if valid_positions > 0:
        valid_nonzero = ((raw_values_flat != 0) & (modality_labels_flat != -1)).sum().item()
        nonzero_percentage = (valid_nonzero / valid_positions) * 100
        print(f"Valid positions with meaningful targets: {valid_nonzero}/{valid_positions} ({nonzero_percentage:.1f}%)")

    print("✅ Activation collection complete!")

    return activations_tensor_flat, modality_labels_flat, raw_values_flat


def top_k_raw_tokens_for_sae_latent_neuron_n(sae, activations_tensor_flat, modality_labels_flat, raw_values_flat, latent_neuron_n, model, k, batch_size, save_dir):
    """
    Returns dataframe of a CSV file
    The CSV file contains: 
    1. ranking of the top-k raw tokens for the given SAE latent neuron index
    2. activation value 
    3. modality_id 
    4. modality_name
    5. raw_value token that the decoder is trying to predict that generated the activations that led to high latent neuron activation
    
    """
    # Ensure output directory exists
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create modality_id_to_name mapping
    modality_id_to_name = {}
    if model is not None and hasattr(model, 'modality_info'):
        for mod_name, info in model.modality_info.items():
            modality_id_to_name[info['id']] = mod_name
    
    # Use min-heap to efficiently find top-k
    top_k_heap = []  # Will store (activation_value, global_token_idx)
    
    print(f"Finding top-{k} activations for SAE latent neuron {latent_neuron_n}...")
    
    sae.eval()
    with torch.no_grad():
        for i in range(0, len(activations_tensor_flat), batch_size):
            end_idx = min(i + batch_size, len(activations_tensor_flat))
            batch_activations = activations_tensor_flat[i:end_idx]
            
            if batch_activations.device != next(sae.parameters()).device:
                batch_activations = batch_activations.to(next(sae.parameters()).device)
            
            # Get SAE latent activations
            decoded, encoded = sae(batch_activations)  # encoded shape: (batch_size, n_latents)
            
            # Extract the specific latent neuron we're interested in
            neuron_activations = encoded[:, latent_neuron_n].cpu()  # Shape: (batch_size,)
            
            # Process each activation in this batch
            for j, activation_val in enumerate(neuron_activations):
                global_token_idx = i + j
                activation_val = float(activation_val.item())
                
                if len(top_k_heap) < k:
                    # Haven't filled k items yet
                    heapq.heappush(top_k_heap, (activation_val, global_token_idx))
                elif activation_val > top_k_heap[0][0]:
                    # This activation is larger than the smallest in our top-k
                    heapq.heapreplace(top_k_heap, (activation_val, global_token_idx))
    
    # Sort results by activation value (descending)
    top_k_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

    results = []
    
    for rank, (activation_value, global_token_idx) in enumerate(top_k_results):
        mod_id_val = int(modality_labels_flat[global_token_idx]) if modality_labels_flat is not None else -1
        raw_value = int(raw_values_flat[global_token_idx]) if raw_values_flat is not None else -1
        
        # Handle special case where modality_id == -1 (these are padding tokens)
        modality_name_val = (
            'padding' if mod_id_val == -1 else modality_id_to_name.get(mod_id_val, f'unknown_mod_{mod_id_val}')
        )

        result = {
            'rank': rank + 1,
            'activation_value': activation_value,
            'modality_id': mod_id_val,
            'modality_name': modality_name_val,
            'raw_value': raw_value,
            'global_token_idx': global_token_idx,  # Include for debugging
        }
        results.append(result)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(results)
    
    # Save results
    csv_filename = f"latent_{latent_neuron_n}_top_{k}_activations.csv"
    csv_path = out_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print summary
    print(f"\nTop-{k} activations for SAE latent neuron {latent_neuron_n}:")
    print(f"{'Rank':<4} {'Activation':<12} {'Modality':<15} {'Raw Value':<10} {'Global Token Index':<12}")
    print("-" * 65)
    for result in results:
        print(f"{result['rank']:<4} {result['activation_value']:<12.6f} {result['modality_name']:<15} {result['raw_value']:<10} {result['global_token_idx']:<12}")
    
    return df
	
	
	
	
	
	