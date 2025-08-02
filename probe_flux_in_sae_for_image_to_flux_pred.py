import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, Dict
import pandas as pd
from tqdm import tqdm
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage, 
    HSCMagG,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
)
from sparse_autoencoder import SparseAutoencoder
from scipy.stats import pearsonr, spearmanr

def load_image_data(path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess the image data."""
    img_data = torch.load(path, weights_only=True)
    # Use only DES-G, DES-R, DES-I, DES-Z bands (channels 4:8)
    image_tensor = img_data['image']['array'][:, 4:]  # shape: [batch, 4, 96, 96]
    image_tensor = image_tensor.float().to(device)
    print(f"Loaded image data: shape={image_tensor.shape}")
    print(f"Image data min={image_tensor.min().item()}, max={image_tensor.max().item()}")
    return image_tensor

def predict_flux_from_images(model, codec_manager, images, bands, batch_size=32):
    """
    Predict flux values from images using AION.
    
    Args:
        model: AION model
        codec_manager: CodecManager instance
        images: Image tensor [N, C, H, W]
        bands: List of band names
        batch_size: Batch size for prediction
        
    Returns:
        Dict with predicted flux values for each band
    """
    print(f"Predicting flux from {len(images)} images...")
    
    # Define target modalities
    target_modalities = {
        'flux_g': LegacySurveyFluxG,
        'flux_r': LegacySurveyFluxR,
        'flux_i': LegacySurveyFluxI,
        'flux_z': LegacySurveyFluxZ,
    }
    
    all_predicted_fluxes = {name: [] for name in target_modalities.keys()}
    
    # Process in batches
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting flux"):
        batch_img = images[i:i+batch_size]
        img_mod = LegacySurveyImage(flux=batch_img, bands=bands)
        
        tokens = codec_manager.encode(img_mod)
        
        # Predict for each modality
        for modality_name, modality_class in target_modalities.items():
            with torch.no_grad():
                preds = model(tokens, target_modality=modality_class)
                
                # Get the raw logits from model predictions
                raw_logits = preds[modality_class.token_key]  # [batch, num_tokens, vocab_size]
                
                # Convert logits to discrete token indices
                token_indices = raw_logits.argmax(dim=-1)  # [batch, num_tokens]
                
                # Create tokens dict for decode
                tokens_for_decode = {modality_class.token_key: token_indices}
                
                # Decode the discrete tokens to get actual flux values
                pred_values = codec_manager.decode(tokens_for_decode, modality_class)
                
                # Extract flux values
                flux_values = pred_values.value.squeeze().cpu().numpy()
                all_predicted_fluxes[modality_name].append(flux_values)
    
    # Concatenate all batches
    predicted_fluxes = {}
    for modality_name in target_modalities.keys():
        predicted_flux = np.concatenate(all_predicted_fluxes[modality_name])
        predicted_fluxes[modality_name] = predicted_flux
        
        print(f"{modality_name}: predicted {len(predicted_flux)} values")
        print(f"  Range: {predicted_flux.min():.6f} to {predicted_flux.max():.6f}")
    
    return predicted_fluxes

def get_aion_activations_and_flux_predictions(
    image_tensor: torch.Tensor,
    model: AION,
    codec_manager: CodecManager,
    block_idx: int = 8,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pass images through AION, collect activations from the specified decoder block
    and predict flux values for each band.
    Returns (activations, predicted_fluxes) as numpy arrays.
    """
    activations = []
    bands = ["DES-G", "DES-R", "DES-I", "DES-Z"]
    
    # Define target modalities
    target_modalities = {
        'flux_g': LegacySurveyFluxG,
        'flux_r': LegacySurveyFluxR,
        'flux_i': LegacySurveyFluxI,
        'flux_z': LegacySurveyFluxZ,
    }
    
    all_predicted_fluxes = {name: [] for name in target_modalities.keys()}
    
    # Register a forward hook to collect activations
    activation_collected_for_batch = False
    def _hook(module, input, output):
        nonlocal activation_collected_for_batch
        if not activation_collected_for_batch:
            activations.append(output.detach().cpu())
            activation_collected_for_batch = True
    
    handle = model.decoder[block_idx].register_forward_hook(_hook)
    
    # Process in batches to ensure activations and predictions are aligned
    print(f"Processing {len(image_tensor)} images in batches of {batch_size}...")
    for i in tqdm(range(0, len(image_tensor), batch_size), desc="Processing batches"):
        batch_img = image_tensor[i:i+batch_size]
        img_mod = LegacySurveyImage(flux=batch_img, bands=bands)
        
        tokens = codec_manager.encode(img_mod)
        
        # Reset activation collection flag for this batch
        activation_collected_for_batch = False
        
        # Predict for each modality - this will trigger the hook to collect activations only once per batch
        for modality_name, modality_class in target_modalities.items():
            with torch.no_grad():
                preds = model(tokens, target_modality=modality_class)
                
                # Get the raw logits from model predictions
                raw_logits = preds[modality_class.token_key]  # [batch, num_tokens, vocab_size]
                
                # Convert logits to discrete token indices
                token_indices = raw_logits.argmax(dim=-1)  # [batch, num_tokens]
                
                # Create tokens dict for decode
                tokens_for_decode = {modality_class.token_key: token_indices}
                
                # Decode the discrete tokens to get actual flux values
                pred_values = codec_manager.decode(tokens_for_decode, modality_class)
                
                # Extract flux values
                flux_values = pred_values.value.squeeze().cpu().numpy()
                all_predicted_fluxes[modality_name].append(flux_values)
    
    handle.remove()
    
    # Concatenate predictions
    predicted_fluxes = {}
    for modality_name in target_modalities.keys():
        predicted_flux = np.concatenate(all_predicted_fluxes[modality_name])
        predicted_fluxes[modality_name] = predicted_flux
        print(f"{modality_name}: predicted {len(predicted_flux)} values")
        print(f"  Range: {predicted_flux.min():.6f} to {predicted_flux.max():.6f}")
    
    # Process activations
    activations_np = torch.cat(activations, dim=0).numpy()  # shape: (N, seq_length, 768)
    sample_size, seq_length, embed_dim = activations_np.shape
    activations_np_flat = activations_np.reshape(sample_size * seq_length, embed_dim)
    
    # Repeat flux values to match flattened activations (each sample repeated seq_length times)
    predicted_fluxes_repeated = {}
    for band, flux_values in predicted_fluxes.items():
        predicted_fluxes_repeated[band] = np.repeat(flux_values, seq_length)
    
    print(f"AION activations: shape={activations_np_flat.shape}")
    print(f"Predicted flux shapes (repeated): {[(k, v.shape) for k, v in predicted_fluxes_repeated.items()]}")
    print(f"Sample activations[0]: {activations_np_flat[0][:5]}")
    
    if save_path is not None:
        save_data = {'activations': activations_np_flat}
        save_data.update(predicted_fluxes_repeated)
        np.savez(save_path, **save_data)
        print(f"Saved activations and predicted fluxes to {save_path}")
    
    return activations_np_flat, predicted_fluxes_repeated

def get_sae_latents(
    activations: np.ndarray,
    sae_model,
    device: torch.device,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Pass activations through SAE, return latents (numpy array).
    """
    with torch.no_grad():
        acts = torch.tensor(activations, dtype=torch.float32, device=device)
        _, latents = sae_model(acts)
        latents_np = latents.cpu().numpy()
    print(f"SAE latents: shape={latents_np.shape}")
    if save_path is not None:
        np.save(save_path, latents_np)
        print(f"Saved SAE latents to {save_path}")
    return latents_np

def probe_latent_flux_correlation(
    latents: np.ndarray,
    predicted_fluxes: Dict[str, np.ndarray],
    plot: bool = True,
    top_k: int = 5,
    save_prefix: Optional[str] = None
):
    """
    Compute Pearson correlation between each latent and predicted flux for each band.
    Optionally plot top_k most correlated latents.
    """
    bands = ['flux_g', 'flux_r', 'flux_i', 'flux_z']
    band_labels = ['G', 'R', 'I', 'Z']
    
    n_latents = latents.shape[1]
    all_results = {}
    
    # The flux values are repeated to match the flattened activations
    # We need to get the original flux values (before repetition) for proper masking
    # The repetition factor is seq_length from the activations
    first_band_flux = predicted_fluxes[bands[0]]
    total_samples = latents.shape[0]
    repetition_factor = total_samples // len(first_band_flux)
    
    # Get the original flux values (before repetition)
    # Since the flux values are repeated consecutively (each value repeated repetition_factor times),
    # we take the first original_flux_length elements
    original_flux_length = len(first_band_flux) // repetition_factor
    original_flux_g = predicted_fluxes['flux_g'][:original_flux_length]
    original_flux_r = predicted_fluxes['flux_r'][:original_flux_length]
    original_flux_i = predicted_fluxes['flux_i'][:original_flux_length]
    original_flux_z = predicted_fluxes['flux_z'][:original_flux_length]
    
    # Create positive mask for original flux values
    positive_mask = original_flux_g > 0  # Use G band as reference
    
    if np.sum(positive_mask) == 0:
        print("No positive flux values found for any band!")
        return {}
    
    # Apply mask to original flux values
    original_flux_g_pos = original_flux_g[positive_mask]
    original_flux_r_pos = original_flux_r[positive_mask]
    original_flux_i_pos = original_flux_i[positive_mask]
    original_flux_z_pos = original_flux_z[positive_mask]
    
    # Create repeated positive mask for latents
    repeated_positive_mask = np.repeat(positive_mask, repetition_factor)
    latents_pos = latents[repeated_positive_mask]
    
    print(f"Using {np.sum(positive_mask)} positive flux samples out of {len(original_flux_g)} total")
    print(f"Latents shape after masking: {latents_pos.shape}")
    
    for band_key, band_label, original_flux_pos in zip(bands, band_labels, 
                                                      [original_flux_g_pos, original_flux_r_pos, 
                                                       original_flux_i_pos, original_flux_z_pos]):
        # Repeat the original positive flux values to match latents_pos
        flux_pos = np.repeat(original_flux_pos, repetition_factor)
        log_flux = np.log10(flux_pos)
        
        # Compute correlations for each latent
        n_samples = latents_pos.shape[0]
        
        # Create results list
        results = []
        for i in range(n_latents):
            latent_col = latents_pos[:, i]
            
            # Use scipy.stats for reliable correlation computation
            pearson_raw, _ = pearsonr(latent_col, flux_pos)
            pearson_log, _ = pearsonr(latent_col, log_flux)
            spearman_raw, _ = spearmanr(latent_col, flux_pos)
            
            results.append({
                'latent_idx': i,
                'band': band_label,
                'pearson_raw': pearson_raw,
                'pearson_log': pearson_log,
                'spearman_raw': spearman_raw
            })
        
        # Filter out results with nan correlations
        valid_results = [r for r in results if not np.isnan(r['pearson_log'])]
        
        # Sort by absolute Pearson correlation with log(flux)
        results_sorted = sorted(valid_results, key=lambda x: abs(x['pearson_log']), reverse=True)
        
        all_results[band_label] = results_sorted
        
        if save_prefix is not None:
            pd.DataFrame(results).to_csv(f"{save_prefix}_latent_{band_label}flux_correlation.csv", index=False)
            print(f"Saved {band_label} band correlation results to {save_prefix}_latent_{band_label}flux_correlation.csv")
        
        print(f"\n{band_label} Band Results:")
        print(f"Total latents analyzed: {len(results)}")
        print(f"Valid latents (non-nan): {len(valid_results)}")
        
        if len(valid_results) == 0:
            print("No valid correlations found (all latents are constant)")
            continue
        
        # Use min of top_k and available valid results
        actual_top_k = min(top_k, len(valid_results))
        print("Top-{} latent correlations with log10({} band flux):".format(actual_top_k, band_label))
        for j in range(actual_top_k):
            idx = results_sorted[j]['latent_idx']
            print(f" Latent {idx}: Pearson={results_sorted[j]['pearson_log']:.4f}, Spearman={results_sorted[j]['spearman_raw']:.4f}")

        # Print top 10 by pearson_log
        print(f"\nTop 10 by Pearson correlation with log10({band_label} band flux):")
        top_10_pearson_log = sorted(valid_results, key=lambda x: abs(x['pearson_log']), reverse=True)[:10]
        for r in top_10_pearson_log:
            print(f" Latent {r['latent_idx']}: Pearson_log={r['pearson_log']:.4f}, Pearson_raw={r['pearson_raw']:.4f}, Spearman_raw={r['spearman_raw']:.4f}")
        
        all_corrs = [abs(r['pearson_log']) for r in valid_results]
        print(f"Correlation stats for {band_label} band (valid only): min={np.min(all_corrs):.4f}, max={np.max(all_corrs):.4f}, mean={np.mean(all_corrs):.4f}")
        
        # Plot for this band
        if plot and len(valid_results) > 0:
            for j in range(actual_top_k):
                idx = results_sorted[j]['latent_idx']
                
                # Plot raw flux vs latent
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.scatter(flux_pos, latents_pos[:, idx], alpha=0.5)
                plt.xlabel(f'{band_label} band flux (nanomaggies)')
                plt.ylabel(f'Latent {idx}')
                plt.title(f'Latent {idx} vs raw {band_label} band flux\nPearson: {results_sorted[j]["pearson_raw"]:.3f}')
                
                plt.subplot(1, 2, 2)
                plt.scatter(log_flux, latents_pos[:, idx], alpha=0.5)
                plt.xlabel(f'log10({band_label} band flux)')
                plt.ylabel(f'Latent {idx}')
                plt.title(f'Latent {idx} vs log10({band_label} band flux)\nPearson: {results_sorted[j]["pearson_log"]:.3f}')
                
                plt.tight_layout()
                if save_prefix is not None:
                    plt.savefig(f"{save_prefix}_latent{idx}_vs_{band_label}flux_both.png")
                plt.show()
            
            # Add scatter plot of all latent neurons vs. their absolute Pearson correlation with log(flux)
            plt.figure(figsize=(10, 4))
            abs_corrs = [abs(r['pearson_log']) for r in results]
            plt.scatter(range(n_latents), abs_corrs, alpha=0.7)
            plt.xlabel('Latent Neuron Index')
            plt.ylabel(f'Absolute Pearson Correlation with log10({band_label} band flux)')
            plt.title(f'Latent Neuron vs. log10({band_label} band flux) Correlation (Scatter)')
            if save_prefix is not None:
                plt.savefig(f"{save_prefix}_all_latents_{band_label}flux_scatter.png")
            plt.show()
    
    return all_results

def main(
    image_path='AION/tests/test_data/image_codec_input_batch.pt',
    sae_weights_path='best_llm_sae_rural-wood-16.pth',
    save_dir='probe_flux_in_sae_image_to_flux_results',
    batch_size=32,
    block_idx=8,
    device=None
):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
    codec_manager = CodecManager(device=device)
    
    # Load image data
    image_tensor = load_image_data(image_path, device)
    print(f"Image data checksum: {torch.sum(image_tensor).item()}")
    
    # Get AION activations and flux predictions from images
    activations_path = os.path.join(save_dir, 'aion_activations_and_flux_predictions.npz')
    if os.path.exists(activations_path):
        data = np.load(activations_path)
        activations_np_flat = data['activations']
        predicted_fluxes_repeated = {
            'flux_g': data['flux_g'],
            'flux_r': data['flux_r'],
            'flux_i': data['flux_i'],
            'flux_z': data['flux_z']
        }
        print(f"AION activations: shape={activations_np_flat.shape}")
        print(f"Predicted fluxes shapes: {[(k, v.shape) for k, v in predicted_fluxes_repeated.items()]}")
        print(f"Loaded activations and flux predictions from {activations_path}")
        print(f"Activations checksum: {np.sum(activations_np_flat)}")
        for band, flux in predicted_fluxes_repeated.items():
            print(f"{band} checksum: {np.sum(flux)}")
    else:
        activations_np_flat, predicted_fluxes_repeated = get_aion_activations_and_flux_predictions(
            image_tensor, model, codec_manager, block_idx=block_idx, batch_size=batch_size, save_path=activations_path)
        print(f"Generated activations checksum: {np.sum(activations_np_flat)}")
        for band, flux in predicted_fluxes_repeated.items():
            print(f"Generated {band} checksum: {np.sum(flux)}")
    
    # Load SAE
    input_size = activations_np_flat.shape[1]
    print('input size', input_size)
    hidden_size = input_size * 4
    k = max(1, int(hidden_size * 0.02))  # Use 2% sparsity by default
    sae_model = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    sae_model.load_state_dict(torch.load(sae_weights_path, weights_only=True, map_location=device))
    sae_model.eval()
    
    # Get SAE latents
    latents_path = os.path.join(save_dir, 'sae_latents_flux_pred.npy')
    if os.path.exists(latents_path):
        latents_np = np.load(latents_path)
        print(f"Loaded SAE latents from {latents_path}")
        print("Latents shape:", latents_np.shape)
        print("Latents mean (first 10):", np.mean(latents_np, axis=0)[:10])
        print("Latents std (first 10):", np.std(latents_np, axis=0)[:10])
        print("Number of nonzero elements:", np.count_nonzero(latents_np))
        print("Fraction of nonzero elements:", np.count_nonzero(latents_np) / latents_np.size)
        print(f"Latents checksum: {np.sum(latents_np)}")
    else:
        latents_np = get_sae_latents(activations_np_flat, sae_model, device, save_path=latents_path)
        print(f"Generated latents checksum: {np.sum(latents_np)}")
    
    # Probe correlation between latents and predicted flux for each band
    results = probe_latent_flux_correlation(
        latents_np, predicted_fluxes_repeated, plot=True, top_k=5, 
        save_prefix=os.path.join(save_dir, 'probe_flux_pred')
    )
    
    print(f"\nSaved results to {save_dir}")

if __name__ == '__main__':
    main() 