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

            # ------------------------------------------------------------------
            # Build the dictionaries expected by `embed_inputs` / `embed_targets`
            # ------------------------------------------------------------------
            input_dict = {}
            input_mask = {}
            target_mask = {}

            for mod, d in processed_batch.items():
                # Encoder side (always use the raw token tensor)
                input_dict[mod] = d["tensor"]

                if "input_mask" in d:
                    input_mask[mod] = d["input_mask"].to(torch.bool)

                # Decoder side: boolean mask indicating which positions are to be predicted
                target_mask[mod] = d["target_mask"].to(torch.bool)

            # Extract actual tokens and target masks for mapping purposes
            actual_tokens_at_positions = {}
            for mod, d in processed_batch.items():
                actual_tokens_at_positions[mod] = d["tensor"]

            # ------------------------------------------------------------------
            # Run through token selection utilities using AION methods
            # ------------------------------------------------------------------
            enc_tokens, enc_emb, enc_mask, encoder_mod_mask = model.embed_inputs(
                input_dict,
                mask=input_mask if len(input_mask) > 0 else None,
                num_encoder_tokens=args.num_input_tokens,
            )

            (
                dec_tokens,
                dec_emb,
                dec_mask,
                target_ids,
                dec_att_mask,
                dec_mod_mask,
            ) = model.embed_targets(target_mask, num_decoder_tokens=args.num_target_tokens)

            # NEW: Check if entire target_ids tensor is just zeros
            print(f'\n=== CHECKING IF TARGET_IDS IS ALL ZEROS ===')
            print(f'target_ids shape: {target_ids.shape}')
            print(f'Are ALL target_ids zeros? {torch.all(target_ids == 0)}')
            print(f'Number of zero values: {(target_ids == 0).sum().item()} / {target_ids.numel()}')
            print(f'Unique values in target_ids: {torch.unique(target_ids)}')
            print('=== END TARGET_IDS CHECK ===\n')

            # DEBUG: Check why target_ids are all zeros
            print('\n=== DEBUGGING TARGET_IDS ===')
            print('Target mask contents for tok_spectrum_sdss modality:')
            target_mod = 'tok_spectrum_sdss'
            print(f'Modality: {target_mod}')
            print(f'Target mask shape: {target_mask[target_mod].shape}')
            print(f'Target mask first sample: {target_mask[target_mod][0]}')
            print(f'Positions to predict (False): {(~target_mask[target_mod][0]).nonzero().flatten()}')
            print(f'Positions not to predict (True): {target_mask[target_mod][0].nonzero().flatten()}')

            print(f'\nActual tokens for {target_mod} (first sample):')
            actual_tokens_target_mod = processed_batch[target_mod]["tensor"][0]
            print(f'Shape: {actual_tokens_target_mod.shape}')
            print(f'First 10 tokens: {actual_tokens_target_mod[:10]}')
            print(f'Last 10 tokens: {actual_tokens_target_mod[-10:]}')
            
            # Check if there are any non-zero tokens - FIX THE BUG
            print(f'All tokens in sequence: {actual_tokens_target_mod}')
            non_zero_mask = (actual_tokens_target_mod != 0)
            non_zero_positions = non_zero_mask.nonzero().flatten()
            zero_positions = (~non_zero_mask).nonzero().flatten()
            print(f'Non-zero token positions (total {len(non_zero_positions)}): {non_zero_positions}')
            print(f'Zero token positions (total {len(zero_positions)}): {zero_positions}')
                
            # NEW: Show how the target mask maps to actual positions that will be predicted
            print(f'\n--- MAPPING TARGET MASK TO ACTUAL TOKENS ---')
            positions_to_predict = (~target_mask[target_mod][0]).nonzero().flatten()
            print(f'Total positions to predict: {len(positions_to_predict)}')
            print(f'First 10 positions to predict: {positions_to_predict[:10]}')
            print(f'Tokens at those positions: {actual_tokens_target_mod[positions_to_predict[:10]]}')
            # Show what tokens are NOT being predicted (kept visible)
            positions_kept_visible = target_mask[target_mod][0].nonzero().flatten()
            print(f'Total positions kept visible: {len(positions_kept_visible)}')
            print(f'First 10 positions kept visible: {positions_kept_visible[:10]}')
            print(f'Tokens at visible positions: {actual_tokens_target_mod[positions_kept_visible[:10]]}')
            print('=== END DEBUGGING ===\n')
            
            # NEW: Detailed analysis of embed_targets output specifically for tok_spectrum_sdss
            print('\n=== EMBED_TARGETS OUTPUT ANALYSIS ===')
            print('Understanding what embed_targets produces for tok_spectrum_sdss...')
            
            # Find which positions in the decoder sequence correspond to tok_spectrum_sdss
            if target_mod in model.modality_info:
                target_mod_id = model.modality_info[target_mod]["id"]
                print(f'{target_mod} modality ID: {target_mod_id}')
                
                # Find positions in decoder sequence that belong to this modality
                sample_0_mod_mask = dec_mod_mask[0]  # First sample
                target_mod_positions = (sample_0_mod_mask == target_mod_id).nonzero().flatten()
                print(f'Positions in decoder sequence for {target_mod}: {target_mod_positions}')
                print(f'Total {target_mod} positions in decoder: {len(target_mod_positions)}')
                
                if len(target_mod_positions) > 0:
                    print(f'\n--- DECODER TOKENS FOR {target_mod.upper()} ---')
                    print(f'dec_tokens shape: {dec_tokens.shape}')
                    print(f'Decoder tokens at {target_mod} positions (first 5):')
                    for i, pos in enumerate(target_mod_positions[:5]):
                        token_embedding = dec_tokens[0, pos, :]
                        print(f'  Position {pos}: embedding shape {token_embedding.shape}, first 5 values: {token_embedding[:5]}')
                    print(f' Position 0: embedding shape {dec_tokens[0, 0, :].shape}, first 5 values: {dec_tokens[0, 0, :5]}')
                    
                    # NEW: Print first few and last few 768-dimensional decoder token vectors
                    print(f'\n--- ANALYZING 768-DIMENSIONAL DECODER TOKEN VECTORS ---')
                    print(f'dec_tokens shape: {dec_tokens.shape}')
                    
                    # Print first few decoder tokens (should be mask tokens if they're for prediction)
                    print(f'\nFIRST FEW DECODER TOKENS (positions 0-4):')
                    for pos in range(min(5, dec_tokens.shape[1])):
                        token_vec = dec_tokens[0, pos, :]  # First sample, position pos
                        print(f'  Position {pos}:')
                        print(f'    First 10 values: {token_vec[:10].tolist()}')
                        print(f'    Last 10 values:  {token_vec[-10:].tolist()}')
                        print(f'    All zeros? {torch.all(token_vec == 0.0)}')
                        print(f'    All same value? {torch.all(token_vec == token_vec[0])}')
                        print(f'    Mean: {token_vec.mean():.6f}, Std: {token_vec.std():.6f}')
                        print()
                    
                    # Print last few decoder tokens (might be padding)
                    print(f'LAST FEW DECODER TOKENS (positions {dec_tokens.shape[1]-5}-{dec_tokens.shape[1]-1}):')
                    for pos in range(max(0, dec_tokens.shape[1]-5), dec_tokens.shape[1]):
                        token_vec = dec_tokens[0, pos, :]  # First sample, position pos
                        print(f'  Position {pos}:')
                        print(f'    First 10 values: {token_vec[:10].tolist()}')
                        print(f'    Last 10 values:  {token_vec[-10:].tolist()}')
                        print(f'    All zeros? {torch.all(token_vec == 0.0)}')
                        print(f'    All same value? {torch.all(token_vec == token_vec[0])}')
                        print(f'    Mean: {token_vec.mean():.6f}, Std: {token_vec.std():.6f}')
                        print()
                    print(f'\n--- TARGET IDS FOR {target_mod.upper()} ---')
                    print(f'target_ids shape: {target_ids.shape}')
                    target_ids_for_mod = target_ids[0, target_mod_positions]
                    print(f'Target IDs at {target_mod} positions: {target_ids_for_mod}')
                    print(f'Target IDs first 10: {target_ids_for_mod[:10]}')
                    print(f'Target IDs last 10: {target_ids_for_mod[-10:]}')
                    
                    print(f'\n--- DECODER MASK FOR {target_mod.upper()} ---')
                    print(f'dec_mask shape: {dec_mask.shape}')
                    # dec_mask has shape (B, 1, M) - squeeze to (B, M)
                    decoder_mask_squeezed = dec_mask.squeeze(1) if dec_mask.ndim == 3 else dec_mask
                    dec_mask_for_mod = decoder_mask_squeezed[0, target_mod_positions]
                    print(f'Decoder mask at {target_mod} positions: {dec_mask_for_mod}')
                    print(f'Decoder mask (first 10): {dec_mask_for_mod[:10]}')
                    print(f'Decoder mask (last 10): {dec_mask_for_mod[-10:]}')
                    print(f'True (masked) positions: {dec_mask_for_mod.sum()} / {len(dec_mask_for_mod)}')
                    
                    print(f'\n--- DECODER EMBEDDINGS FOR {target_mod.upper()} ---')
                    print(f'dec_emb shape: {dec_emb.shape}')
                    print(f'Decoder embeddings at {target_mod} positions (first 3):')
                    for i, pos in enumerate(target_mod_positions[:3]):
                        emb = dec_emb[0, pos, :]
                        print(f'  Position {pos}: embedding shape {emb.shape}, first 5 values: {emb[:5]}')
                    print(f' Position 0: embedding shape {dec_emb[0, 0, :].shape}, first 5 values: {dec_emb[0, 0, :5]}')

                    print(f'\n--- DECODER ATTENTION MASK FOR {target_mod.upper()} ---')
                    print(f'dec_att_mask shape: {dec_att_mask.shape}')
                    print(f'Attention mask at {target_mod} positions (first 5x5 submatrix):')
                    att_mask_for_mod = dec_att_mask[0, target_mod_positions[:5], :][:, target_mod_positions[:5]]
                    print(att_mask_for_mod)
                    print(f'Attention mask allows attention to itself? {dec_att_mask[0, target_mod_positions[0], target_mod_positions[0]]}')
                    
                    # NEW: Let's specifically check attention patterns within vs between modalities
                    print(f'\n--- DETAILED ATTENTION PATTERN ANALYSIS ---')
                    if len(target_mod_positions) > 1:
                        print(f'Checking attention WITHIN {target_mod} modality:')
                        # Check attention between first two tok_spectrum_sdss positions
                        pos1, pos2 = target_mod_positions[0], target_mod_positions[1]
                        can_attend = not dec_att_mask[0, pos1, pos2].item()  # False = can attend
                        print(f'  Position {pos1} can attend to position {pos2}? {can_attend}')
                        print(f'  Position {pos2} can attend to position {pos1}? {not dec_att_mask[0, pos2, pos1].item()}')
                        
                        # Show the full attention pattern within this modality
                        within_modality_mask = dec_att_mask[0, target_mod_positions[:4], :][:, target_mod_positions[:4]]
                        print(f'  Attention mask within {target_mod} (4x4 submatrix, False=can_attend):')
                        print(f'  {within_modality_mask}')
                    
                    # Check attention between different modalities
                    other_modality_positions = (dec_mod_mask[0] != target_mod_id).nonzero().flatten()
                    if len(other_modality_positions) > 0 and len(target_mod_positions) > 0:
                        print(f'\nChecking attention BETWEEN modalities:')
                        pos_target = target_mod_positions[0]
                        pos_other = other_modality_positions[0]
                        can_attend_cross = not dec_att_mask[0, pos_target, pos_other].item()
                        print(f'  {target_mod} position {pos_target} can attend to other modality position {pos_other}? {can_attend_cross}')
                        print(f'  Other modality position {pos_other} can attend to {target_mod} position {pos_target}? {not dec_att_mask[0, pos_other, pos_target].item()}')
                        
                        # Show cross-modality attention pattern
                        cross_modality_mask = dec_att_mask[0, target_mod_positions[:2], :][:, other_modality_positions[:2]]
                        print(f'  Cross-modality attention mask (2x2 submatrix, False=can_attend):')
                        print(f'  {cross_modality_mask}')
                    
                    print(f'--- END DETAILED ATTENTION PATTERN ANALYSIS ---')
                    
                    # NEW: Debug the decoder attention mask computation
                    print(f'\n--- DEBUGGING DECODER ATTENTION MASK COMPUTATION ---')
                    
                    # First, let's see the original decoder_attention_mask before adapt_decoder_attention_mask
                    print(f'Original target_mask for {target_mod}: {target_mask[target_mod][0]}')
                    print(f'False positions (to predict): {(~target_mask[target_mod][0]).nonzero().flatten()}')
                    print(f'True positions (keep visible): {target_mask[target_mod][0].nonzero().flatten()}')
                    
                    # Let's also check the decoder_attention_mask from the modality dict
                    if target_mod in processed_batch:
                        if "decoder_attention_mask" in processed_batch[target_mod]:
                            raw_decoder_att_mask = processed_batch[target_mod]["decoder_attention_mask"][0]
                            print(f'Raw decoder_attention_mask from {target_mod}: {raw_decoder_att_mask}')
                            print(f'Raw decoder_attention_mask shape: {raw_decoder_att_mask.shape}')
                            print(f'Raw decoder_attention_mask values: {torch.unique(raw_decoder_att_mask)}')
                        else:
                            print(f'No decoder_attention_mask found in processed_batch[{target_mod}]')
                    
                    # Check what happens in adapt_decoder_attention_mask
                    print(f'\n--- ANALYZING adapt_decoder_attention_mask LOGIC ---')
                    print(f'Model decoder_causal_mask: {model.decoder_causal_mask}')
                    print(f'Model decoder_sep_mask: {model.decoder_sep_mask}')
                    
                    # Let's manually trace through the logic for the tok_spectrum_sdss positions
                    if len(target_mod_positions) > 0:
                        print(f'\nFor {target_mod} positions {target_mod_positions[:3]}:')
                        
                        # Check modality IDs
                        mod_ids_for_positions = dec_mod_mask[0, target_mod_positions[:3]]
                        print(f'Modality IDs: {mod_ids_for_positions}')
                        
                        # If decoder_sep_mask is True, different modalities can't attend to each other
                        if model.decoder_sep_mask:
                            print(f'decoder_sep_mask is True - tokens can only attend within same modality')
                            
                            # Check which positions have the same modality ID as our target_mod
                            target_mod_id = model.modality_info[target_mod]["id"]
                            same_modality_positions = (dec_mod_mask[0] == target_mod_id).nonzero().flatten()
                            different_modality_positions = (dec_mod_mask[0] != target_mod_id).nonzero().flatten()
                            
                            print(f'Positions with same modality ID ({target_mod_id}): {same_modality_positions}')
                            print(f'Positions with different modality IDs: {different_modality_positions[:10]} (first 10)')
                            
                            # For sep_mask: positions with different modalities get True (blocked)
                            print(f'sep_mask logic: different modalities get True (blocked from attending)')
                            
                        # Let's also check the actual decoder_attention_mask that was input
                        print(f'\n--- CHECKING INPUT TO adapt_decoder_attention_mask ---')
                        # We need to look at the decoder_attention_mask before it gets adapted
                        # This should be in the cat_decoder_tensors output
                        
                        # Let's manually run through what cat_decoder_tensors would produce
                        print(f'For {target_mod} - checking cat_decoder_tensors logic:')
                        original_attention_mask = processed_batch[target_mod].get("decoder_attention_mask", None)
                        if original_attention_mask is not None:
                            print(f'  Original decoder_attention_mask shape: {original_attention_mask.shape}')
                            print(f'  Original decoder_attention_mask[0]: {original_attention_mask[0]}')
                            print(f'  Original decoder_attention_mask unique values: {torch.unique(original_attention_mask)}')
                            
                            # This gets fed into adapt_decoder_attention_mask as decoder_attention_mask_all
                            # After gather/sorting, what would be the input to adapt_decoder_attention_mask?
                            
                        # Also check: what should cumsum_mask be?
                        print(f'\nManual cumsum calculation for debugging:')
                        if original_attention_mask is not None:
                            sample_att_mask = original_attention_mask[0]  # First sample
                            print(f'  Sample attention mask: {sample_att_mask}')
                            manual_cumsum = torch.cumsum(sample_att_mask.int(), dim=-1)
                            print(f'  Manual cumsum: {manual_cumsum}')
                            
                            # What would attention_arange >= cumsum give us?
                            N = len(sample_att_mask)
                            attention_arange = torch.arange(N, device=sample_att_mask.device)
                            print(f'  attention_arange: {attention_arange}')
                            comparison_result = attention_arange.unsqueeze(0) >= manual_cumsum.unsqueeze(1)
                            print(f'  attention_arange >= cumsum_mask result: {comparison_result}')
                        else:
                            print(f'  No original decoder_attention_mask to analyze')

                    print(f'--- END DECODER ATTENTION MASK DEBUGGING ---\n')

                    print(f'\n--- DECODER MODALITY MASK FOR {target_mod.upper()} ---')
                    print(f'dec_mod_mask shape: {dec_mod_mask.shape}')
                    dec_mod_mask_for_mod = dec_mod_mask[0, target_mod_positions]
                    print(f'Modality IDs at {target_mod} positions: {dec_mod_mask_for_mod}')
                    print(f'All positions have correct modality ID? {torch.all(dec_mod_mask_for_mod == target_mod_id)}')
                        
                    print(f'\n--- RELATIONSHIP BETWEEN ORIGINAL AND DECODER ---')
                    print(f'Original {target_mod} sequence length: {actual_tokens_target_mod.shape[0]}')
                    print(f'Decoder positions for {target_mod}: {len(target_mod_positions)}')
                    print(f'Are they equal? {actual_tokens_target_mod.shape[0] == len(target_mod_positions)}')
                    
                    # Show the mapping between original positions and decoder positions
                    print(f'\nMapping original -> decoder positions (first 10):')
                    for i, decoder_pos in enumerate(target_mod_positions[:10]):
                        original_token = actual_tokens_target_mod[i]
                        target_id = target_ids[0, decoder_pos] 
                        was_masked = decoder_mask_squeezed[0, decoder_pos]
                        print(f'  Original pos {i}: token {original_token} -> Decoder pos {decoder_pos}: target_id {target_id}, masked={was_masked}')
                        
                    # NEW: Critical analysis - understand what target_ids should actually be
                    print(f'\n--- WHAT SHOULD TARGET_IDS ACTUALLY BE? ---')
                    print(f'The target_ids we see: {target_ids[0, target_mod_positions]}')
                    print(f'But positions to predict in original sequence were: {positions_to_predict}')
                    print(f'Tokens at those positions were: {actual_tokens_target_mod[positions_to_predict]}')
                    
                    # Check if embed_targets is supposed to copy the actual tokens that need prediction
                    print(f'\nLet\'s examine what embed_targets should be doing:')
                    print(f'1. Original sequence has {len(actual_tokens_target_mod)} tokens')
                    print(f'2. Only {len(positions_to_predict)} positions need prediction')
                    print(f'3. But decoder has {len(target_mod_positions)} positions for this modality')
                    print(f'4. If decoder only gets predicted positions, target_ids should be:')
                    if len(positions_to_predict) == len(target_mod_positions):
                        expected_target_ids = actual_tokens_target_mod[positions_to_predict]
                        print(f'   Expected target_ids: {expected_target_ids}')
                        print(f'   Actual target_ids: {target_ids[0, target_mod_positions]}')
                        print(f'   Do they match? {torch.equal(expected_target_ids, target_ids[0, target_mod_positions])}')
                    else:
                        print(f'   Mismatch: {len(positions_to_predict)} positions to predict but {len(target_mod_positions)} decoder positions')
                        print(f'   This suggests embed_targets uses a different selection strategy')
                        
                    # Let's also check if the decoder tokens are actually the right embeddings
                    print(f'\n--- ARE DECODER TOKENS CORRECT? ---')
                    print(f'Decoder token at position {target_mod_positions[0]}: {dec_tokens[0, target_mod_positions[0], :5]}')
                    print(f'This looks like mask token embedding (all positions identical)')
                    print(f'For masked positions, decoder should get mask token embedding')
                    print(f'For visible positions, decoder should get actual token embedding')
                    print(f'Expected: If position was masked for prediction -> mask embedding')
                    print(f'Expected: If position was visible -> actual token embedding')

            print('=== END EMBED_TARGETS OUTPUT ANALYSIS ===\n')

            # Analyze which positions have zero vs non-zero tensors
            print('\nANALYZING ZERO vs NON-ZERO POSITIONS:')

            zero_positions = []
            nonzero_positions = []

            # Check each position for the first sample
            for pos in range(dec_tokens.shape[1]):
                position_tensor = dec_tokens[0, pos, :]  # First sample, position pos
                
                # Check if all values are zero
                if torch.all(position_tensor == 0.0):
                    zero_positions.append(pos)
                else:
                    nonzero_positions.append(pos)

            print(f'Zero tensor positions ({len(zero_positions)}): {zero_positions}')
            print(f'Non-zero tensor positions ({len(nonzero_positions)}): {nonzero_positions}')
            # Check if all non-zero positions have the same values
            if len(nonzero_positions) > 1:
                print(f'\nCHECKING IF NON-ZERO POSITIONS ARE IDENTICAL:')
                first_nonzero = dec_tokens[0, nonzero_positions[0], :]
                all_nonzero_same = True
                
                for pos in nonzero_positions[1:]:
                    if not torch.equal(first_nonzero, dec_tokens[0, pos, :]):
                        all_nonzero_same = False
                        print(f'Position {pos} differs from position {nonzero_positions[0]}')
                        break
                
                print(f'All non-zero positions identical? {all_nonzero_same}')

            # ------------------------------------------------------------------
            # Forward pass: encoder context -> decoder (hook captures block-8)
            # ------------------------------------------------------------------
            context = model._encode(enc_tokens, enc_emb, enc_mask)

            decoder_output = model._decode(
                context,
                enc_mask,
                dec_tokens,
                dec_emb,
                dec_att_mask,
            )  # hook fires here

            # ------------------------------------------------------------------
            # COMPUTE ACTUAL PREDICTIONS (logits -> predicted token IDs)
            # ------------------------------------------------------------------
            if batch_idx == 0:  # Only for first batch to see predictions
                print(f"\n=== MODEL PREDICTIONS ===")
                mod_logits = {}
                predicted_tokens = {}
                
                for mod in target_mask.keys():
                    idx = model.modality_info[mod]["id"]
                    
                    # Get positions for this modality in the decoder output
                    mod_positions = (dec_mod_mask == idx)
                    if mod_positions.any():
                        # Compute logits for this modality
                        mod_decoder_output = decoder_output[mod_positions]  # Shape: (num_positions, hidden_dim)
                        mod_logits[mod] = model.decoder_embeddings[mod].forward_logits(mod_decoder_output)
                        
                        # Convert logits to predicted token IDs
                        predicted_tokens[mod] = torch.argmax(mod_logits[mod], dim=-1)  # Shape: (num_positions,)
                        
                        # Get actual ground truth for comparison
                        actual_tokens_for_mod = processed_batch[mod]["tensor"][0]  # First sample
                        target_mask_for_mod = target_mask[mod][0]  # First sample
                        
                        print(f"\nModality {mod}:")
                        print(f"  Predicted tokens (first 10): {predicted_tokens[mod][:10].tolist()}")
                        print(f"  Actual tokens    (first 10): {actual_tokens_for_mod[:10].tolist()}")
                        
                        # Show predictions specifically for masked positions (what model was trying to predict)
                        masked_positions = ~target_mask_for_mod  # False = positions to predict
                        if masked_positions.any():
                            actual_masked = actual_tokens_for_mod[masked_positions]
                            # Map decoder positions back to original positions for this modality
                            first_sample_mod_mask = dec_mod_mask[0] == idx
                            if first_sample_mod_mask.sum() > 0:
                                pred_for_masked = predicted_tokens[mod]
                                print(f"  Predictions for masked positions: {pred_for_masked[:min(10, len(pred_for_masked))].tolist()}")
                                print(f"  Ground truth for masked positions: {actual_masked[:min(10, len(actual_masked))].tolist()}")
                            
                                # Calculate accuracy for masked predictions
                                if len(pred_for_masked) >= len(actual_masked):
                                    matches = (pred_for_masked[:len(actual_masked)] == actual_masked.to(pred_for_masked.device)).sum().item()
                                    accuracy = matches / len(actual_masked)
                                    print(f"  Accuracy on masked positions: {accuracy:.3f} ({matches}/{len(actual_masked)})")
                
                print(f"=== END MODEL PREDICTIONS ===\n")

            # Store the decoder tokens that were processed
            decoder_tokens_cache[batch_idx] = {
                'target_ids': target_ids.detach().cpu(),  # Shape: (batch_size, seq_len) - what decoder saw (mostly zeros)
                'decoder_mask': dec_mask.detach().cpu(),  # Boolean mask indicating which tokens were masked (True) or kept (False)
                'batch_size': dec_tokens.shape[0],
                'seq_len': dec_tokens.shape[1],
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

def get_exact_decoder_token_info(global_token_idx, sample_mapping, decoder_tokens_cache, original_batch_cache=None, model=None, modality_labels_flat=None):
    """
    Get the exact decoder token ID, modality, AND raw pre-tokenization value for a given activation index.
    Now correctly handles masked tokens vs actual tokens for proper SAE analysis.
    
    Args:
        global_token_idx: Index into the flattened activation tensor
        sample_mapping: Mapping from token indices to sample info  
        decoder_tokens_cache: Cache of decoder tokens processed (target_ids, decoder_mask)
        original_batch_cache: Cache of original batch data before tokenization
        model: Model for accessing modality info
        modality_labels_flat: Flattened modality labels for each token position
        
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
    target_ids = batch_cache['target_ids']  # Shape: (batch_size, seq_len) - what decoder saw (mostly zeros)
    
    # Extract the specific token
    if sample_idx >= target_ids.shape[0] or token_position >= target_ids.shape[1]:
        return {'error': f'Token position out of bounds: sample_idx={sample_idx}, token_position={token_position}, target_ids.shape={target_ids.shape}'}
    
    # Get what the decoder actually saw at this position (likely 0 if masked)
    decoder_saw_token_id = int(target_ids[sample_idx, token_position].item())
    
    # Get modality ID from modality_labels_flat
    modality_id = int(modality_labels_flat[global_token_idx].item()) if modality_labels_flat is not None else -1
    
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
    
    # Get the actual token from original_batch_cache
    if (original_batch_cache is not None 
        and batch_idx in original_batch_cache 
        and modality_name in original_batch_cache[batch_idx]
        and sample_idx < original_batch_cache[batch_idx][modality_name]['tensor'].shape[0]):
        
        # Get the flattened sequence position and map back to modality position
        # We need to find which position within this modality this token corresponds to
        
        # Calculate how many tokens come before this position from other modalities in this batch
        total_tokens_before = 0
        for b_idx in range(batch_idx):
            if b_idx in decoder_tokens_cache:
                total_tokens_before += decoder_tokens_cache[b_idx]['batch_size'] * decoder_tokens_cache[b_idx]['seq_len']
        
        # Add tokens from previous samples in this batch
        total_tokens_before += sample_idx * decoder_tokens_cache[batch_idx]['seq_len']
        
        # The position within this sample's decoder sequence
        position_in_sample = global_token_idx - total_tokens_before
        
        # Find positions in the decoder sequence that belong to this modality
        sample_start_idx = total_tokens_before
        sample_end_idx = sample_start_idx + decoder_tokens_cache[batch_idx]['seq_len']
        
        if modality_labels_flat is not None:
            # Get modality labels for this sample
            sample_modality_labels = modality_labels_flat[sample_start_idx:sample_end_idx]
            modality_positions = (sample_modality_labels == modality_id).nonzero(as_tuple=False).squeeze(-1)
            
            if token_position in modality_positions:
                # Find the index within the modality sequence
                idx_in_modality = (modality_positions == token_position).nonzero(as_tuple=False).item()
                
                # Get the actual token from original batch cache
                original_tensor = original_batch_cache[batch_idx][modality_name]['tensor']
                if idx_in_modality < original_tensor.shape[1]:  # Check bounds
                    actual_token_id = int(original_tensor[sample_idx, idx_in_modality].item())
    
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
    
    # Extract raw tensor data directly from original_batch_cache
    if (original_batch_cache is not None 
        and batch_idx in original_batch_cache 
        and modality_name in original_batch_cache[batch_idx]
        and sample_idx < original_batch_cache[batch_idx][modality_name]['tensor'].shape[0]):
        
        raw_tensor = original_batch_cache[batch_idx][modality_name]['tensor'][sample_idx]
        
        # Add tensor metadata
        result['raw_tensor_shape'] = list(raw_tensor.shape)
        result['raw_tensor_dtype'] = str(raw_tensor.dtype)
        
        # Handle different tensor types for CSV-friendly output
        if raw_tensor.ndim == 0:
            # Scalar value
            result['raw_value'] = float(raw_tensor.item())
            result['raw_value_summary'] = f"scalar: {result['raw_value']}"
        elif raw_tensor.ndim == 1:
            # 1D array
            result['raw_first_value'] = float(raw_tensor[0].item()) if len(raw_tensor) > 0 else None
            result['raw_array_length'] = raw_tensor.shape[0]
            result['raw_value_summary'] = f"1D array[{raw_tensor.shape[0]}], starts with: {result['raw_first_value']}"
        elif raw_tensor.ndim == 2:
            # 2D array
            result['raw_array_shape'] = list(raw_tensor.shape)
            result['raw_value_summary'] = f"2D array{list(raw_tensor.shape)}"
        else:
            # Higher dimensional
            result['raw_array_shape'] = list(raw_tensor.shape)
            result['raw_value_summary'] = f"{raw_tensor.ndim}D array{list(raw_tensor.shape)}"
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
                global_token_idx, sample_mapping, decoder_tokens_cache, original_batch_cache, model, modality_labels_flat
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
    
    # Save results
    csv_filename = f"{filename_prefix}latent_{latent_neuron_n}_top_{k}_masked_prediction_analysis.csv"
    csv_path = out_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # # ================================================================
    # # NEW: Save an additional CSV focused solely on the query tokens
    # # (i.e., positions that were masked for prediction) so that we can
    # # inspect exactly which inputs triggered the strongest response in
    # # latent neuron 0.
    # # ================================================================

    # if 'was_masked_for_prediction' in df.columns:
    #     df_query_tokens = df[df['was_masked_for_prediction'] == True].copy()

    #     print(f"[INFO] Identified {len(df_query_tokens)} masked (query) tokens within the initial top-{k} set.")

    #     # Sort by activation value and keep only the top 10
    #     top_query_k = 10
    #     df_query_tokens = df_query_tokens.sort_values('activation_value', ascending=False).head(top_query_k)

    #     # Keep only the columns required by the user
    #     desired_columns = [
    #         'global_token_index',
    #         'modality_name',
    #         'modality_id',
    #         'activation_value'
    #     ]
    #     missing_cols = [c for c in desired_columns if c not in df_query_tokens.columns]
    #     if missing_cols:
    #         print(f"[WARNING] The following expected columns were not found in the DataFrame and will be skipped: {missing_cols}")
    #     exported_columns = [c for c in desired_columns if c in df_query_tokens.columns]

    #     df_query_export = df_query_tokens[exported_columns]

    #     query_csv_filename = f"{filename_prefix}latent_{latent_neuron_n}_top_{top_query_k}_query_tokens.csv"
    #     query_csv_path = out_dir / query_csv_filename
    #     df_query_export.to_csv(query_csv_path, index=False)

    #     print(f"[INFO] Top-{top_query_k} query tokens saved (masked prediction positions only): {query_csv_path}")
    #     print("[INFO] Preview of the saved query-token CSV:")
    #     print(df_query_export.head())
    # else:
    #     print("[WARNING] 'was_masked_for_prediction' column missing – cannot create query-token specific CSV.")
    
    return df
	