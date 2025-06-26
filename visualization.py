import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import numpy as np
import pandas as pd
import heapq
from pathlib import Path
from tqdm import tqdm

def collect_activations_with_raw_data_mapping(
    model, data_loader, device, args):
    """
    Collect SAE activations and map them to the exact decoder tokens that produced them.
    Also stores original batch data to enable extraction of raw pre-tokenization values.
    
    Returns:
        activations_tensor_flat: Flattened activations (n_tokens, hidden_dim)
        modality_labels_flat: Modality ID for each token  
        sample_mapping: Dict mapping token indices to sample info
        decoder_tokens_cache: Cache of exact decoder tokens processed by model
        original_batch_cache: Cache of original batch data before tokenization
    """
    
    model = model.to(device).eval()

    activations = []
    modality_labels = []
    examples_processed = 0
    
    # Storage for mapping activations back to exact decoder tokens AND original data
    sample_mapping = {}  # {global_token_idx: {'sample_key': str, 'batch_idx': int, 'token_idx': int}}
    decoder_tokens_cache = {}  # Store actual decoder tokens processed by the model
    original_batch_cache = {}  # Store original batch data before any processing
    
    global_token_idx = 0
    
    # Register a hook on the 9-th decoder block (index 8)
    def _hook(_module, _inp, output):
        # Store hidden states on CPU to save GPU memory
        activations.append(output.detach().cpu())

    hook_handle = model.decoder[8].register_forward_hook(_hook)
    print("[INFO] Forward hook registered on decoder block index 8 (9th block) to capture activations.")
    
    # Iterate over batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Collecting activations")):
            
            # Store original batch data BEFORE any processing
            original_batch_cache[batch_idx] = {}
            for mod_name, mod_data in batch.items():
                if 'tensor' in mod_data:
                    # Store the original tensor data on CPU to save memory
                    original_batch_cache[batch_idx][mod_name] = {
                        'tensor': mod_data['tensor'].clone().cpu(),
                        'target_mask': mod_data.get('target_mask', None)
                    }
            
            # # Create simple sample identifiers for this batch
            # batch_size = batch[list(batch.keys())[0]]['tensor'].shape[0]
            # for sample_idx in range(batch_size):
            #     sample_key = f"batch_{batch_idx}_sample_{sample_idx}"
                
                # Store mapping info for later
                # (We'll update with actual token positions after forward pass)
            # Move batch tensors to device
            processed_batch = {
                mod: {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in d.items()
                }
                for mod, d in batch.items()
            }

            # Extract actual tokens and target masks for mapping purposes
            actual_tokens_at_positions = {}
            target_mask = {}
            for mod, d in processed_batch.items():
                actual_tokens_at_positions[mod] = d["tensor"]
                target_mask[mod] = d.get("target_mask", torch.zeros_like(d["tensor"], dtype=torch.bool))

            # 1. Prepare encoder embeddings
            encoder_mod_dict = {
                mod: model.encoder_embeddings[mod](d)
                for mod, d in processed_batch.items()
                if mod in model.encoder_embeddings
            }
            
            # 2. Prepare decoder embeddings  
            decoder_mod_dict = {
                mod: model.decoder_embeddings[mod].forward_embed(d)
                for mod, d in processed_batch.items()
                if mod in model.decoder_embeddings
            }
            
            # 3. Get encoder tokens and embeddings
            encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = \
                model.forward_mask_encoder(encoder_mod_dict, args.num_input_tokens)
            
            # 4. Get decoder tokens and embeddings (this handles masking correctly)
            (decoder_tokens, decoder_emb, decoder_mask, target_ids, 
             decoder_attention_mask, dec_mod_mask) = \
                model.forward_mask_decoder(decoder_mod_dict, args.num_target_tokens)
            
            # DEBUG: Check what the decoder is actually seeing vs actual tokens
            if batch_idx == 0:  # Only print for first batch
                print(f"DEBUG: Decoder input (masked) tokens from batch {batch_idx}:")
                print(f"  target_ids shape: {target_ids.shape}")
                print(f"  First 10 target_ids (decoder sees): {target_ids[0, :10].tolist()}")
                print(f"  Unique target_ids: {torch.unique(target_ids[0]).tolist()}")
                
                # Show what actual tokens were at those positions
                if len(actual_tokens_at_positions) > 0:
                    first_mod = list(actual_tokens_at_positions.keys())[0]
                    actual_sample = actual_tokens_at_positions[first_mod][0, :10]
                    print(f"  First 10 actual tokens ({first_mod}): {actual_sample.tolist()}")
                    print(f"  Decoder mod mask: {dec_mod_mask[0, :10].tolist()}")

            # 5. Forward through encoder
            x = encoder_tokens + encoder_emb
            x = model.forward_encoder(x, encoder_mask=encoder_mask)
            
            # 6. Prepare decoder input (this is the key part you were missing)
            context = model.decoder_proj_context(x) + encoder_emb
            y = decoder_tokens + decoder_emb  # This is what actually goes to decoder
            
            # 7. Forward through decoder (hook captures activations)
            y = model.forward_decoder(
                y,
                context,
                encoder_mask=encoder_mask,
                decoder_attention_mask=decoder_attention_mask,
            )
            # # Build the dictionaries expected by embed_inputs / embed_targets
            # input_dict = {}
            # input_mask = {}
            # target_mask = {}

            # for mod, d in processed_batch.items():
            #     input_dict[mod] = d["tensor"]
            #     if "input_mask" in d:
            #         input_mask[mod] = d["input_mask"].to(torch.bool)
            #     target_mask[mod] = d["target_mask"].to(torch.bool)

            # # Run through token selection utilities
            # enc_tokens, enc_emb, enc_mask, _ = model.embed_inputs(
            #     input_dict,
            #     mask=input_mask if len(input_mask) > 0 else None,
            #     num_encoder_tokens=args.num_input_tokens,
            # )

            # # FIXED: Use proper embed_targets which creates masked tokens for prediction
            # # This is the correct way - decoder should see masked tokens, not actual tokens
            # (
            #     dec_tokens,
            #     dec_emb,
            #     dec_mask,
            #     target_ids,  # These will be zeros where masked, but we track actual tokens separately
            #     dec_att_mask,
            #     dec_mod_mask,
            # ) = model.embed_targets(target_mask, num_decoder_tokens=args.num_target_tokens)
            
            # # CRITICAL: Also extract the actual tokens at masked positions for tracking
            # # We need this to map activations back to what the decoder was trying to predict
            # actual_tokens_at_positions = {}
            # for mod, mask in target_mask.items():
            #     # Get the actual tokens from the original tensor
            #     actual_tokens_at_positions[mod] = processed_batch[mod]["tensor"]

            # # Forward pass: encoder context -> decoder (hook captures block-8)
            # context = model._encode(enc_tokens, enc_emb, enc_mask)
            # _ = model._decode(
            #     context,
            #     enc_mask,
            #     dec_tokens,
            #     dec_emb,
            #     dec_att_mask,
            # )  # hook fires here

            # Store the decoder tokens that were processed AND the actual tokens for mapping
            decoder_tokens_cache[batch_idx] = {
                'dec_tokens': decoder_tokens.detach().cpu(),  # Shape: (batch_size, seq_len, embed_dim) - what decoder saw
                'target_ids': target_ids.detach().cpu(),  # Shape: (batch_size, seq_len) - what decoder saw (mostly zeros)
                'dec_mod_mask': dec_mod_mask.detach().cpu(),  # Modality labels for each token
                'decoder_mask': decoder_mask.detach().cpu(),  # Boolean mask indicating which tokens were masked (True) or kept (False)
                'actual_tokens': {mod: tokens.detach().cpu() for mod, tokens in actual_tokens_at_positions.items()},  # NEW: actual tokens
                'target_mask': {mod: mask.detach().cpu() for mod, mask in target_mask.items()},  # NEW: masks for mapping
                'batch_size': decoder_tokens.shape[0],
                'seq_len': decoder_tokens.shape[1],
            }

            # Store ground-truth modality labels for this batch
            modality_labels.append(dec_mod_mask.cpu())

            # Create token-to-sample mapping for this batch
            if activations:  # Only if activations were captured
                batch_activations = activations[-1]  # Get the most recent activation
                batch_size_act, seq_len, hidden_dim = batch_activations.shape
                
                for sample_idx in range(batch_size_act):
                    sample_key = f"batch_{batch_idx}_sample_{sample_idx}"
                    
                    # Map each token position to this sample
                    for token_idx in range(seq_len):
                        sample_mapping[global_token_idx] = {
                            'sample_key': sample_key,
                            'batch_idx': batch_idx,
                            'sample_idx': sample_idx,
                            'token_idx': token_idx,
                        }
                        global_token_idx += 1

            # House-keeping
            batch_size = next(iter(processed_batch.values()))["tensor"].shape[0]
            examples_processed += batch_size

            torch.cuda.empty_cache()
            
            if batch_idx >= 4:  # Limit for demo purposes
                break

    # Cleanup hook
    hook_handle.remove()

    # Stack and flatten tensors
    activations_tensor = torch.cat(activations, dim=0)           # (N, M, D)
    modality_labels_tensor = torch.cat(modality_labels, dim=0)   # (N, M)

    assert activations_tensor.shape[:2] == modality_labels_tensor.shape, (
        "Decoder activations and modality labels mis-aligned: "
        f"{activations_tensor.shape[:2]} vs {modality_labels_tensor.shape}"
    )

    N, M, D = activations_tensor.shape
    activations_tensor_flat = activations_tensor.reshape(N * M, D)
    modality_labels_flat = modality_labels_tensor.reshape(-1)
    
    print(f"Collected {activations_tensor_flat.shape[0]} token activations from {examples_processed} samples")
    print(f"Decoder tokens cached for {len(decoder_tokens_cache)} batches")
    print(f"Original batch data cached for {len(original_batch_cache)} batches")
    
    return activations_tensor_flat, modality_labels_flat, sample_mapping, decoder_tokens_cache, original_batch_cache

def extract_raw_value_from_original_data(batch_idx, sample_idx, modality_name, original_batch_cache):
    """
    Extract the raw pre-tokenization value(s) for a specific sample and modality.
    
    Args:
        batch_idx: Which batch this sample came from
        sample_idx: Which sample within the batch
        modality_name: Name of the modality (e.g., 'tok_flux_g')
        original_batch_cache: Cache of original batch data
        
    Returns:
        dict with raw value information
    """
    if batch_idx not in original_batch_cache:
        return {'error': 'Batch not found in original data cache'}
    
    batch_data = original_batch_cache[batch_idx]
    
    if modality_name not in batch_data:
        return {'error': f'Modality {modality_name} not found in batch {batch_idx}'}
    
    modality_data = batch_data[modality_name]
    tensor = modality_data['tensor']  # Shape varies by modality
    
    # Extract data for this specific sample
    if sample_idx >= tensor.shape[0]:
        return {'error': f'Sample index {sample_idx} out of bounds for batch size {tensor.shape[0]}'}
    
    sample_tensor = tensor[sample_idx]  # Remove batch dimension
    
    # Handle different modality types
    result = {
        'modality_name': modality_name,
        'original_shape': list(sample_tensor.shape),
        'tensor_dtype': str(sample_tensor.dtype),
    }
    
    if sample_tensor.ndim == 0:
        # Scalar value (e.g., single flux measurement, redshift)
        result['raw_value'] = float(sample_tensor.item())
        result['value_type'] = 'scalar'
        
    elif sample_tensor.ndim == 1:
        # 1D array (e.g., spectrum, flux measurements)
        result['raw_values'] = sample_tensor.tolist()
        result['value_type'] = 'array_1d'
        result['array_length'] = sample_tensor.shape[0]
        # Show first few values for preview
        result['preview_values'] = sample_tensor[:min(10, len(sample_tensor))].tolist()
        
    elif sample_tensor.ndim == 2:
        # 2D array (e.g., image patches)
        result['raw_values'] = sample_tensor.tolist()  # Full 2D array
        result['value_type'] = 'array_2d'
        result['array_shape'] = list(sample_tensor.shape)
        # Show corner for preview
        preview_size = min(5, sample_tensor.shape[0]), min(5, sample_tensor.shape[1])
        result['preview_corner'] = sample_tensor[:preview_size[0], :preview_size[1]].tolist()
        
    elif sample_tensor.ndim == 3:
        # 3D array (e.g., multi-channel image patches)
        result['value_type'] = 'array_3d'
        result['array_shape'] = list(sample_tensor.shape)
        # Don't store full 3D array (too large), just metadata and preview
        result['preview_slice'] = sample_tensor[0, :min(5, sample_tensor.shape[1]), :min(5, sample_tensor.shape[2])].tolist()
        
    else:
        # Higher dimensional - just store metadata
        result['value_type'] = f'array_{sample_tensor.ndim}d'
        result['array_shape'] = list(sample_tensor.shape)
        result['note'] = 'high_dimensional_array_not_fully_stored'
    
    return result

def get_exact_decoder_token_info(global_token_idx, sample_mapping, decoder_tokens_cache, original_batch_cache=None, model=None):
    """
    Get the exact decoder token ID, modality, AND raw pre-tokenization value for a given activation index.
    Now correctly handles masked tokens vs actual tokens for proper SAE analysis.
    
    Args:
        global_token_idx: Index into the flattened activation tensor
        sample_mapping: Mapping from token indices to sample info  
        decoder_tokens_cache: Cache of actual decoder tokens processed
        original_batch_cache: Cache of original batch data before tokenization
        model: Model for accessing modality info
        
    Returns:
        dict with exact token information AND raw values
    """
    if global_token_idx not in sample_mapping:
        return {'error': 'Token index not found in sample mapping'}
    
    # Get sample info
    sample_info = sample_mapping[global_token_idx]
    batch_idx = sample_info['batch_idx']
    sample_idx = sample_info['sample_idx'] 
    token_position = sample_info['token_idx']
    
    # Get the exact decoder token that was processed
    if batch_idx not in decoder_tokens_cache:
        return {'error': f'Decoder tokens not cached for batch {batch_idx}'}
    
    batch_cache = decoder_tokens_cache[batch_idx]
    dec_tokens = batch_cache['dec_tokens']  # Shape: (batch_size, seq_len, embed_dim) - what decoder saw
    dec_mod_mask = batch_cache['dec_mod_mask']  # Shape: (batch_size, seq_len)
    target_ids = batch_cache['target_ids']  # Shape: (batch_size, seq_len) - what decoder saw (mostly zeros)
    actual_tokens = batch_cache['actual_tokens']  # Dict of actual tokens per modality
    target_masks = batch_cache['target_mask']  # Dict of masks per modality
    
    # Extract the specific token
    if sample_idx >= target_ids.shape[0] or token_position >= target_ids.shape[1]:
        return {'error': f'Token position out of bounds: sample_idx={sample_idx}, token_position={token_position}, target_ids.shape={target_ids.shape}'}
    
    # Get what the decoder actually saw at this position (likely 0 if masked)
    decoder_saw_token_id = int(target_ids[sample_idx, token_position].item())
    
    # Get modality ID from dec_mod_mask
    modality_id = int(dec_mod_mask[sample_idx, token_position].item())
    
    # Get modality name
    if modality_id == -1:
        modality_name = 'query_token'  # Special case for masked positions
    else:
        modality_name = 'unknown'
        if model is not None and hasattr(model, 'modality_info'):
            for name, info in model.modality_info.items():
                if isinstance(info, dict) and info.get('id') == modality_id:
                    modality_name = name
                    break
    
    # CRITICAL: Get the actual token that was at this position (what decoder was trying to predict)
    actual_token_id = None

    # Retrieve decoder_mask (True = position was masked, hence decoder input is the mask token)
    was_masked = None
    decoder_mask = batch_cache.get('decoder_mask', None)
    if decoder_mask is not None:
        # decoder_mask is returned from forward_mask_decoder with shape (B, 1, M) where the 
        # middle dimension of 1 is added for attention compatibility via rearrange("b n2 -> b 1 n2").
        # We need to squeeze it to (B, M) for proper indexing: decoder_mask[sample_idx, token_position]
        if decoder_mask.ndim == 3 and decoder_mask.shape[1] == 1:
            decoder_mask = decoder_mask.squeeze(1)  # -> (B, M)
        was_masked = bool(decoder_mask[sample_idx, token_position].item())
    
    # If we have access to the original token sequence for this modality we can attempt to fetch it
    if (
        was_masked is not None
        and modality_name in actual_tokens
        and sample_idx < actual_tokens[modality_name].shape[0]
    ):
        # Map token_position within the flattened decoder sequence back into the modality sequence
        # by using dec_mod_mask equality.
        modality_positions = (dec_mod_mask[sample_idx] == modality_id).nonzero(as_tuple=False).squeeze(-1)
        if token_position in modality_positions:
            # Index into modality_positions array to find offset inside modality sequence
            idx_in_modality = (modality_positions == token_position).nonzero(as_tuple=False).item()
            seq_len_mod = actual_tokens[modality_name].shape[1]
            if idx_in_modality < seq_len_mod:
                actual_token_id = int(actual_tokens[modality_name][sample_idx, idx_in_modality].item())
    
    result = {
        'decoder_saw_token_id': decoder_saw_token_id,  # What decoder actually processed (likely 0 if masked)
        'actual_token_id': actual_token_id,  # What the correct answer was (what decoder was trying to predict)
        'was_masked_for_prediction': was_masked,  # Whether this was a position decoder had to predict
        'modality_id': modality_id,
        'modality_name': modality_name,
        'batch_idx': batch_idx,
        'sample_idx': sample_idx,
        'token_position': token_position,
        'decoder_sequence_length': target_ids.shape[1],
    }
    
    # Extract raw pre-tokenization values using the actual token (not what decoder saw)
    if original_batch_cache is not None and actual_token_id is not None:
        raw_value_info = extract_raw_value_from_original_data(
            batch_idx, sample_idx, modality_name, original_batch_cache
        )
        
        # Add raw value info to result with clear prefixes
        for key, value in raw_value_info.items():
            result[f'raw_{key}'] = value
            
        # Also add some convenient extracted fields for CSV
        if 'raw_value' in raw_value_info:
            result['raw_scalar_value'] = raw_value_info['raw_value']
        elif 'raw_values' in raw_value_info and isinstance(raw_value_info['raw_values'], list):
            # For arrays, maybe show the first value or some summary
            if len(raw_value_info['raw_values']) > 0:
                result['raw_first_value'] = raw_value_info['raw_values'][0]
                result['raw_array_length'] = len(raw_value_info['raw_values'])
        
        # Add a human-readable summary
        if raw_value_info.get('value_type') == 'scalar':
            result['raw_value_summary'] = f"scalar: {raw_value_info.get('raw_value', 'N/A')}"
        elif raw_value_info.get('value_type') == 'array_1d':
            length = raw_value_info.get('array_length', 0)
            first_val = raw_value_info.get('preview_values', [None])[0]
            result['raw_value_summary'] = f"1D array[{length}], starts with: {first_val}"
        elif raw_value_info.get('value_type') == 'array_2d':
            shape = raw_value_info.get('array_shape', [])
            result['raw_value_summary'] = f"2D array{shape}"
        else:
            result['raw_value_summary'] = f"{raw_value_info.get('value_type', 'unknown')}"
    else:
        result['raw_value_summary'] = 'original_data_not_available'
    
    return result

def top_k_inputs_for_sae_latent_neuron_n(
    sae, 
    activations_tensor_flat, 
    modality_labels_flat, 
    latent_neuron_n, 
    k=50, 
    batch_size=8192,
    sample_mapping=None, 
    decoder_tokens_cache=None,
    original_batch_cache=None,
    model=None,
    save_dir="latent_topk_visualizations",
    filename_prefix="",
    create_plots=True
):
    """
    Find top-k inputs for SAE latent neuron and map to exact decoder tokens AND raw pre-tokenization values.
    Now correctly analyzes activations from masked token prediction (Option A approach).
    
    Args:
        sae: Trained sparse autoencoder
        activations_tensor_flat: Flattened activations (n_tokens, hidden_dim)
        modality_labels_flat: Modality ID for each token
        latent_neuron_n: Which latent neuron to analyze (0-indexed)
        k: Number of top inputs to return
        batch_size: Batch size for SAE processing  
        sample_mapping: Mapping from token indices to sample info
        decoder_tokens_cache: Cache of exact decoder tokens processed by model
        original_batch_cache: Cache of original batch data before tokenization
        model: Model for accessing modality info
        save_dir: Directory to save results
        filename_prefix: Prefix for saved files
        create_plots: Whether to create visualization plots
        
    Returns:
        DataFrame with top-k results including exact decoder tokens AND raw values
    """
    print(f"\nAnalyzing SAE latent neuron {latent_neuron_n}...")
    print("ANALYSIS TYPE: Studying activations from masked token prediction (decoder sees masked tokens)")

    # Ensure output directory exists
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(activations_tensor_flat)} tokens in batches of {batch_size}")
    
    # Build modality ID -> name map
    modality_id_to_name = {}
    if model is not None and hasattr(model, 'modality_info'):
        for _name, _info in model.modality_info.items():
            if isinstance(_info, dict) and 'id' in _info:
                modality_id_to_name[int(_info['id'])] = _name
    
    # Run SAE and get latent activations for the specific neuron
    print(f"Running SAE inference for latent {latent_neuron_n}...")
    
    # Use min-heap to efficiently find top-k
    top_k_heap = []  # Will store (activation_value, global_token_idx)
    
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
    
    # DEBUG: Check for activation diversity
    unique_activations = len(set([x[0] for x in top_k_results]))
    print(f"DEBUG: Found {unique_activations} unique activation values among top-{k} results")
    if unique_activations < 5:
        print(f"WARNING: Very few unique activations - this suggests potential issues:")
        for i, (act_val, _) in enumerate(top_k_results[:5]):
            print(f"  Rank {i+1}: {act_val}")
    
    print(f"Found top-{k} activating tokens for latent {latent_neuron_n}")
    print(f"Activation range: {top_k_results[-1][0]:.6f} to {top_k_results[0][0]:.6f}")
    
    # Create detailed results with exact decoder token information AND raw values
    results = []
    
    for rank, (activation_value, global_token_idx) in enumerate(top_k_results):
        mod_id_val = int(modality_labels_flat[global_token_idx]) if modality_labels_flat is not None else -1
        
        # Handle special case where modality_id == -1 (these are the query / target tokens with mask ID)
        modality_name_val = (
            'query_token' if mod_id_val == -1 else modality_id_to_name.get(mod_id_val, 'unknown')
        )

        result = {
            'rank': rank + 1,
            'global_token_index': global_token_idx,
            'activation_value': activation_value,
            'modality_id': mod_id_val,
            'modality_name': modality_name_val,
        }
        
        # Add exact decoder token information AND raw values
        if sample_mapping is not None and decoder_tokens_cache is not None:
            token_info = get_exact_decoder_token_info(
                global_token_idx, sample_mapping, decoder_tokens_cache, original_batch_cache, model
            )
            result.update(token_info)
        else:
            result.update({
                'decoder_saw_token_id': None,
                'actual_token_id': None,
                'was_masked_for_prediction': None,
                'batch_idx': -1,
                'sample_idx': -1,
                'token_position': -1,
                'raw_value_summary': 'missing_mapping_or_cache',
                'error': 'missing_mapping_or_cache'
            })
        
        results.append(result)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(results)
    
    # DEBUG: Show sample and modality diversity
    print(f"\n=== Debugging: Sample and Modality Diversity ===")
    if 'batch_idx' in df.columns and 'sample_idx' in df.columns:
        unique_samples = df[['batch_idx', 'sample_idx']].drop_duplicates()
        print(f"Unique samples represented: {len(unique_samples)}/{len(df)}")
        print("Sample distribution:")
        sample_counts = df.groupby(['batch_idx', 'sample_idx']).size().head(10)
        for (batch, sample), count in sample_counts.items():
            print(f"  Batch {batch}, Sample {sample}: {count} tokens")
    
    if 'modality_name' in df.columns:
        modality_counts = df['modality_name'].value_counts()
        print(f"Modality distribution: {dict(modality_counts)}")
    
    if 'was_masked_for_prediction' in df.columns:
        mask_counts = df['was_masked_for_prediction'].value_counts()
        print(f"Mask status: {dict(mask_counts)}")
    
    # Print first few results to verify - now with masked vs actual token info!
    print("\n=== Top 3 Results (Masked Token Prediction Analysis) ===")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        decoder_saw = row.get('decoder_saw_token_id', 'N/A')
        actual_token = row.get('actual_token_id', 'N/A')
        was_masked = row.get('was_masked_for_prediction', 'N/A')
        raw_summary = row.get('raw_value_summary', 'N/A')
        
        print(f"Rank {row['rank']}: Decoder saw token {decoder_saw}, actual token was {actual_token}")
        print(f"   Modality: {row['modality_name']}, Was masked for prediction: {was_masked}")
        print(f"   Activation: {row['activation_value']:.6f}")
        print(f"   Raw value: {raw_summary}")
        print()
    
    # Count masked vs non-masked predictions
    if 'was_masked_for_prediction' in df.columns:
        masked_count = df['was_masked_for_prediction'].sum() if df['was_masked_for_prediction'].dtype == bool else (df['was_masked_for_prediction'] == True).sum()
        print(f"Analysis summary: {masked_count}/{len(df)} top activations came from masked token prediction positions")
    
    # Save results
    csv_filename = f"{filename_prefix}latent_{latent_neuron_n}_top_{k}_masked_prediction_analysis.csv"
    csv_path = out_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # ================================================================
    # NEW: Save an additional CSV focused solely on the query tokens
    # (i.e., positions that were masked for prediction) so that we can
    # inspect exactly which inputs triggered the strongest response in
    # latent neuron 0.
    # ================================================================

    if 'was_masked_for_prediction' in df.columns:
        df_query_tokens = df[df['was_masked_for_prediction'] == True].copy()

        print(f"[INFO] Identified {len(df_query_tokens)} masked (query) tokens within the initial top-{k} set.")

        # Sort by activation value and keep only the top 10
        top_query_k = 10
        df_query_tokens = df_query_tokens.sort_values('activation_value', ascending=False).head(top_query_k)

        # Keep only the columns required by the user
        desired_columns = [
            'global_token_index',
            'modality_name',
            'modality_id',
            'activation_value'
        ]
        missing_cols = [c for c in desired_columns if c not in df_query_tokens.columns]
        if missing_cols:
            print(f"[WARNING] The following expected columns were not found in the DataFrame and will be skipped: {missing_cols}")
        exported_columns = [c for c in desired_columns if c in df_query_tokens.columns]

        df_query_export = df_query_tokens[exported_columns]

        query_csv_filename = f"{filename_prefix}latent_{latent_neuron_n}_top_{top_query_k}_query_tokens.csv"
        query_csv_path = out_dir / query_csv_filename
        df_query_export.to_csv(query_csv_path, index=False)

        print(f"[INFO] Top-{top_query_k} query tokens saved (masked prediction positions only): {query_csv_path}")
        print("[INFO] Preview of the saved query-token CSV:")
        print(df_query_export.head())
    else:
        print("[WARNING] 'was_masked_for_prediction' column missing – cannot create query-token specific CSV.")
    
    # Create visualizations if requested
    if create_plots and modality_labels_flat is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Histogram of activation values
        ax1.hist(df['activation_value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Top-{k} Activation Values\n(Latent Neuron {latent_neuron_n})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Count by modality type
        if 'modality_name' in df.columns:
            modality_counts = df['modality_name'].value_counts()
            ax2.bar(range(len(modality_counts)), modality_counts.values, color='lightcoral')
            ax2.set_xlabel('Modality Type')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Top-{k} Activations by Modality Type\n(Latent Neuron {latent_neuron_n})')
            ax2.set_xticks(range(len(modality_counts)))
            ax2.set_xticklabels(modality_counts.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Masked vs Non-masked predictions
        if 'was_masked_for_prediction' in df.columns:
            mask_counts = df['was_masked_for_prediction'].value_counts()
            # Create dynamic labels based on actual data
            labels = []
            colors = []
            for val in mask_counts.index:
                if val == True:
                    labels.append('Masked')
                    colors.append('orange')
                else:
                    labels.append('Non-masked') 
                    colors.append('lightgreen')
            ax3.pie(mask_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax3.set_title(f'Masked vs Non-masked Predictions\n(Latent Neuron {latent_neuron_n})')
        
        # Plot 4: Activation values for masked vs non-masked
        if 'was_masked_for_prediction' in df.columns:
            masked_activations = df[df['was_masked_for_prediction'] == True]['activation_value']
            non_masked_activations = df[df['was_masked_for_prediction'] == False]['activation_value']
            
            # Only plot if we have data for each category
            if len(masked_activations) > 0:
                ax4.hist(masked_activations, bins=10, alpha=0.7, label='Masked predictions', color='orange')
            if len(non_masked_activations) > 0:
                ax4.hist(non_masked_activations, bins=10, alpha=0.7, label='Non-masked', color='lightgreen')
            
            # Only show legend if we actually plotted something
            if len(masked_activations) > 0 or len(non_masked_activations) > 0:
                ax4.legend()
            
            ax4.set_xlabel('Activation Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Activation Distribution by Mask Status\n(Latent Neuron {latent_neuron_n})')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{filename_prefix}latent_{latent_neuron_n}_top_{k}_masked_prediction_analysis.png"
        plot_path = out_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {plot_path}")
    
    return df
	