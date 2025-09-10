import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
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
from predict_flux_from_image import load_legacysurvey_data, predict_flux_from_images

# def load_image_data(path: str, device: torch.device) -> torch.Tensor:
#     """Load and preprocess the image data."""
#     img_data = torch.load(path, weights_only=True)
#     # Use only DES-G, DES-R, DES-I, DES-Z bands (channels 4:8)
#     image_tensor = img_data['image']['array'][:, 4:]  # shape: [batch, 4, 96, 96]
#     image_tensor = image_tensor.float().to(device)
#     print(f"Loaded image data: shape={image_tensor.shape}")
#     print(f"Image data min={image_tensor.min().item()}, max={image_tensor.max().item()}")
#     return image_tensor

def get_aion_activations_and_flux_predictions(
    image_tensor: torch.Tensor,
    model: AION,
    codec_manager: CodecManager,
    bands: list,
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
    print(f"AION activations: shape={activations_np.shape}")
    activations_np_flat = activations_np.reshape(sample_size * seq_length, embed_dim)
    
    # Repeat flux values to match flattened activations (each sample repeated seq_length times)
    predicted_fluxes_repeated = {}
    for band, flux_values in predicted_fluxes.items():
        predicted_fluxes_repeated[band] = np.repeat(flux_values, seq_length)
    
    print(f"AION activations after flattening: shape={activations_np_flat.shape}")
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
    Pass activations through SAE with proper normalization, return latents (numpy array).
    """
    with torch.no_grad():
        acts = torch.tensor(activations, dtype=torch.float32, device=device)
        
        # Apply same normalization as training
        # Mean subtraction per sample (dim=1) 
        acts_centered = acts - acts.mean(dim=1, keepdim=True)
        
        # Unit normalization
        norms = torch.norm(acts_centered, dim=1, keepdim=True)
        acts_normalized = acts_centered / (norms + 1e-8)
        
        # Pass normalized activations through SAE
        _, latents = sae_model(acts_normalized)
        latents_np = latents.cpu().numpy()
        
    print(f"SAE latents: shape={latents_np.shape}")
    print(f"Applied normalization: mean subtraction + unit norm")
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
        
        # Print top 10 by pearson_raw
        print(f"\nTop 10 by Pearson correlation with raw {band_label} band flux:")
        top_10_pearson_raw = sorted(valid_results, key=lambda x: abs(x['pearson_raw']), reverse=True)[:10]
        for r in top_10_pearson_raw:
            print(f" Latent {r['latent_idx']}: Pearson_raw={r['pearson_raw']:.4f}, Pearson_log={r['pearson_log']:.4f}, Spearman_raw={r['spearman_raw']:.4f}")
        
        all_corrs = [abs(r['pearson_log']) for r in valid_results]
        print(f"Correlation stats for {band_label} band (valid only): min={np.min(all_corrs):.4f}, max={np.max(all_corrs):.4f}, mean={np.mean(all_corrs):.4f}")
        
        # Plot for this band
        if plot and len(valid_results) > 0:
            # Plot top k latents based on pearson_log
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
            
            # Plot top k latents based on pearson_raw
            results_sorted_raw = sorted(valid_results, key=lambda x: abs(x['pearson_raw']), reverse=True)
            for j in range(actual_top_k):
                idx = results_sorted_raw[j]['latent_idx']
                
                # Plot raw flux vs latent
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.scatter(flux_pos, latents_pos[:, idx], alpha=0.5)
                plt.xlabel(f'{band_label} band flux (nanomaggies)')
                plt.ylabel(f'Latent {idx}')
                plt.title(f'Latent {idx} vs raw {band_label} band flux\nPearson: {results_sorted_raw[j]["pearson_raw"]:.3f}')
                
                plt.subplot(1, 2, 2)
                plt.scatter(log_flux, latents_pos[:, idx], alpha=0.5)
                plt.xlabel(f'log10({band_label} band flux)')
                plt.ylabel(f'Latent {idx}')
                plt.title(f'Latent {idx} vs log10({band_label} band flux)\nPearson: {results_sorted_raw[j]["pearson_log"]:.3f}')
                
                plt.tight_layout()
                if save_prefix is not None:
                    plt.savefig(f"{save_prefix}_latent{idx}_vs_{band_label}flux_both_raw.png")
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
            
            # Add histogram plots showing distribution of correlation values for both raw and log
            plt.figure(figsize=(16, 6))
            
            # Get correlation values from already-filtered valid_results
            raw_corrs = [r['pearson_raw'] for r in valid_results]
            log_corrs = [r['pearson_log'] for r in valid_results]
            
            # Create histogram for raw correlations
            plt.subplot(1, 2, 1)
            plt.hist(raw_corrs, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel(f'Pearson Correlation with raw {band_label} band flux')
            plt.ylabel('Number of Latent Neurons')
            plt.title(f'Distribution of Raw Correlation Values\n{band_label} Band')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero correlation')
            plt.legend()
            
            # Add statistics text for raw correlations
            pos_corrs_raw = [c for c in raw_corrs if c > 0]
            neg_corrs_raw = [c for c in raw_corrs if c < 0]
            mean_raw = np.mean(raw_corrs) if raw_corrs else 0
            std_raw = np.std(raw_corrs) if raw_corrs else 0
            plt.text(0.02, 0.98, f'Positive correlations: {len(pos_corrs_raw)} ({len(pos_corrs_raw)/len(raw_corrs)*100:.1f}%)\nNegative correlations: {len(neg_corrs_raw)} ({len(neg_corrs_raw)/len(raw_corrs)*100:.1f}%)\nMean: {mean_raw:.3f}\nStd: {std_raw:.3f}', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Create histogram for log correlations
            plt.subplot(1, 2, 2)
            plt.hist(log_corrs, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel(f'Pearson Correlation with log10({band_label} band flux)')
            plt.ylabel('Number of Latent Neurons')
            plt.title(f'Distribution of Log Correlation Values\n{band_label} Band')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero correlation')
            plt.legend()
            
            # Add statistics text for log correlations
            pos_corrs_log = [c for c in log_corrs if c > 0]
            neg_corrs_log = [c for c in log_corrs if c < 0]
            mean_log = np.mean(log_corrs) if log_corrs else 0
            std_log = np.std(log_corrs) if log_corrs else 0
            plt.text(0.02, 0.98, f'Positive correlations: {len(pos_corrs_log)} ({len(pos_corrs_log)/len(log_corrs)*100:.1f}%)\nNegative correlations: {len(neg_corrs_log)} ({len(neg_corrs_log)/len(log_corrs)*100:.1f}%)\nMean: {mean_log:.3f}\nStd: {std_log:.3f}', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            if save_prefix is not None:
                plt.savefig(f"{save_prefix}_all_latents_{band_label}flux_histograms.png")
            plt.show()
    
    return all_results

def plot_latent_vs_flux_bands(
    latents: np.ndarray,
    predicted_fluxes: Dict[str, np.ndarray],
    latent_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Plot the relationship between a specific latent neuron and predicted flux values for all 4 bands.
    Creates 4 separate PNG files (one for each band) using the "both_raw" format.
    
    Args:
        latents: SAE latents array of shape (N, n_latents)
        predicted_fluxes: Dictionary with keys 'flux_g', 'flux_r', 'flux_i', 'flux_z' containing predicted flux arrays
        latent_idx: Index of the latent neuron to plot
        save_path: Optional path prefix to save the plots (if None, uses default naming)
        figsize: Figure size as (width, height)
    
    Returns:
        Dictionary with correlation statistics for each band
    """
    bands = ['flux_g', 'flux_r', 'flux_i', 'flux_z']
    band_labels = ['G', 'R', 'I', 'Z']
    
    # Check if latent_idx is valid
    if latent_idx >= latents.shape[1]:
        raise ValueError(f"latent_idx {latent_idx} is out of range. Max valid index is {latents.shape[1] - 1}")
    
    # Get the latent activations for the specified neuron
    latent_activations = latents[:, latent_idx]
    
    # The flux values are repeated to match the flattened activations
    # We need to get the original flux values (before repetition) for proper masking
    first_band_flux = predicted_fluxes[bands[0]]
    total_samples = latents.shape[0]
    repetition_factor = total_samples // len(first_band_flux)
    
    # Get the original flux values (before repetition)
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
    latent_activations_pos = latent_activations[repeated_positive_mask]
    
    print(f"Using {np.sum(positive_mask)} positive flux samples out of {len(original_flux_g)} total")
    print(f"Latents shape after masking: {latents_pos.shape}")
    
    # Store correlation results
    correlation_results = {}
    
    # Create separate plots for each band (both_raw format)
    for i, (band_key, band_label, original_flux_pos) in enumerate(zip(bands, band_labels, 
                                                                      [original_flux_g_pos, original_flux_r_pos, 
                                                                       original_flux_i_pos, original_flux_z_pos])):
        # Repeat the original positive flux values to match latents_pos
        flux_pos = np.repeat(original_flux_pos, repetition_factor)
        log_flux = np.log10(flux_pos)
        
        # Calculate correlations
        pearson_raw, _ = pearsonr(latent_activations_pos, flux_pos)
        pearson_log, _ = pearsonr(latent_activations_pos, log_flux)
        spearman_raw, _ = spearmanr(latent_activations_pos, flux_pos)
        
        correlation_results[band_label] = {
            'pearson_raw': pearson_raw,
            'pearson_log': pearson_log,
            'spearman_raw': spearman_raw
        }
        
        # Create the plot for this band (both_raw format - 2 subplots: raw and log)
        plt.figure(figsize=(12, 4))
        
        # Plot raw flux vs latent
        plt.subplot(1, 2, 1)
        plt.scatter(flux_pos, latent_activations_pos, alpha=0.5, s=20)
        plt.xlabel(f'{band_label} band flux (nanomaggies)')
        plt.ylabel(f'Latent {latent_idx}')
        plt.title(f'Latent {latent_idx} vs raw {band_label} band flux\nPearson: {pearson_raw:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Add trend line for raw flux
        z_raw = np.polyfit(flux_pos, latent_activations_pos, 1)
        p_raw = np.poly1d(z_raw)
        plt.plot(flux_pos, p_raw(flux_pos), "r--", alpha=0.8)
        
        # Plot log flux vs latent
        plt.subplot(1, 2, 2)
        plt.scatter(log_flux, latent_activations_pos, alpha=0.5, s=20)
        plt.xlabel(f'log10({band_label} band flux)')
        plt.ylabel(f'Latent {latent_idx}')
        plt.title(f'Latent {latent_idx} vs log10({band_label} band flux)\nPearson: {pearson_log:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Add trend line for log flux
        z_log = np.polyfit(log_flux, latent_activations_pos, 1)
        p_log = np.poly1d(z_log)
        plt.plot(log_flux, p_log(log_flux), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # Save the plot for this band
        if save_path is not None:
            # Extract directory and base name from save_path
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            band_save_path = os.path.join(save_dir, f'probe_flux_pred_latent{latent_idx}_vs_{band_label}flux_both_raw.png')
        else:
            band_save_path = f'probe_flux_pred_latent{latent_idx}_vs_{band_label}flux_both_raw.png'
        
        plt.savefig(band_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {band_label} band plot to {band_save_path}")
        plt.show()
    
    # Print correlation summary
    print(f"\nCorrelation summary for Latent {latent_idx}:")
    print("-" * 50)
    for band_label, corrs in correlation_results.items():
        print(f"{band_label} band:")
        print(f"  Pearson (raw): {corrs['pearson_raw']:.4f}")
        print(f"  Pearson (log): {corrs['pearson_log']:.4f}")
        print(f"  Spearman (raw): {corrs['spearman_raw']:.4f}")
    
    return correlation_results

def plot_specific_latent_vs_bands(
    latent_idx: int,
    save_dir: str = 'probe_flux_in_sae_image_to_flux_results',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Convenience function to plot a specific latent neuron vs all 4 bands.
    This function automatically loads the required data from the save_dir.
    Creates 4 separate PNG files (one for each band) using the "both_raw" format.
    
    Args:
        latent_idx: Index of the latent neuron to plot
        save_dir: Directory where the data files are saved
        save_path: Optional path prefix to save the plots (if None, uses default naming)
        figsize: Figure size as (width, height)
    
    Returns:
        Dictionary with correlation statistics for each band
    """
    # Construct default save path if not provided
    if save_path is None:
        save_path = os.path.join(save_dir, f'probe_flux_pred_latent{latent_idx}')
    
    # Load the required data
    latents_path = os.path.join(save_dir, 'sae_latents_flux_pred.npy')
    activations_path = os.path.join(save_dir, 'aion_activations_and_flux_predictions.npz')
    
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"Latents file not found at {latents_path}. Please run probe() first.")
    
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations file not found at {activations_path}. Please run probe() first.")
    
    # Load latents
    print(f"Loading latents from {latents_path}...")
    latents_np = np.load(latents_path)
    print(f"Loaded latents with shape: {latents_np.shape}")
    
    # Load predicted fluxes
    print(f"Loading predicted fluxes from {activations_path}...")
    data = np.load(activations_path)
    predicted_fluxes_repeated = {
        'flux_g': data['flux_g'],
        'flux_r': data['flux_r'],
        'flux_i': data['flux_i'],
        'flux_z': data['flux_z']
    }
    print(f"Loaded predicted fluxes with shapes: {[(k, v.shape) for k, v in predicted_fluxes_repeated.items()]}")
    
    # Plot the latent neuron vs all bands
    print(f"Plotting latent neuron {latent_idx} vs all 4 bands (both_raw format)...")
    correlation_results = plot_latent_vs_flux_bands(
        latents=latents_np,
        predicted_fluxes=predicted_fluxes_repeated,
        latent_idx=latent_idx,
        save_path=save_path,
        figsize=figsize
    )
    
    print(f"Plots saved to {save_dir} with naming pattern 'probe_flux_pred_latent{latent_idx}_vs_*flux_both_raw.png'")
    return correlation_results

def probe(sae_weights_path='best_llm_sae_rural-wood-16.pth',
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
    
    # Load larger dataset from predict_flux_from_image.py
    print("Loading larger dataset...")
    data_path = '/mnt/home/polymathic/ceph/MultimodalUniverse/legacysurvey/dr10_south_21/healpix=299/001-of-001.hdf5'
    dataset_data = load_legacysurvey_data(data_path, max_samples=10000)
    
    # Convert images to tensor and move to device
    image_tensor = torch.tensor(dataset_data['images'], dtype=torch.float32).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image value range: {image_tensor.min():.6f} to {image_tensor.max():.6f}")
    print(f"Image data checksum: {torch.sum(image_tensor).item()}")
    
    # Get AION activations and flux predictions from images
    activations_path = os.path.join(save_dir, 'aion_activations_and_flux_predictions.npz')
    if os.path.exists(activations_path):
        saved_data = np.load(activations_path)
        activations_np_flat = saved_data['activations']
        predicted_fluxes_repeated = {
            'flux_g': saved_data['flux_g'],
            'flux_r': saved_data['flux_r'],
            'flux_i': saved_data['flux_i'],
            'flux_z': saved_data['flux_z']
        }
        print(f"AION activations: shape={activations_np_flat.shape}")
        print(f"Predicted fluxes shapes: {[(k, v.shape) for k, v in predicted_fluxes_repeated.items()]}")
        print(f"Loaded activations and flux predictions from {activations_path}")
        print(f"Activations checksum: {np.sum(activations_np_flat)}")
        for band, flux in predicted_fluxes_repeated.items():
            print(f"{band} checksum: {np.sum(flux)}")
    else:
        activations_np_flat, predicted_fluxes_repeated = get_aion_activations_and_flux_predictions(
            image_tensor, model, codec_manager, dataset_data['bands'], block_idx=block_idx, batch_size=batch_size, save_path=activations_path)
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


def intervention_experiment(sae_weights_path='best_llm_sae_rural-wood-16.pth',
    save_dir='intervention_results',
    batch_size=32,
    block_idx=8,
    target_latent=1689,
    device=None
):
    # Set random seeds for reproducibility (same as probe function)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. SETUP - Use existing probe function to load everything
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load models (same as probe function)
    model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
    codec_manager = CodecManager(device=device)
    
    # Load dataset using existing function
    print("Loading dataset...")
    data_path = '/mnt/home/polymathic/ceph/MultimodalUniverse/legacysurvey/dr10_south_21/healpix=299/001-of-001.hdf5'
    dataset = load_legacysurvey_data(data_path, max_samples=10000)
    
    # Convert images to tensor and move to device
    image_tensor = torch.tensor(dataset['images'], dtype=torch.float32).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Get TRUE flux values (these are your ground truth)
    true_flux_data = {
        'flux_g': dataset['true_flux_g'],
        'flux_r': dataset['true_flux_r'], 
        'flux_i': dataset['true_flux_i'],
        'flux_z': dataset['true_flux_z']
    }
    print("True flux data loaded.")
    
    # 2. BASELINE PREDICTIONS (no intervention) 
    activations_path = os.path.join(save_dir, 'aion_activations_and_flux_predictions.npz')
    if os.path.exists(activations_path):
        print(f"Loading existing activations and predictions from {activations_path}")
        saved_data = np.load(activations_path)
        activations_np_flat = saved_data['activations']
        baseline_flux_predictions = {
            'flux_g': saved_data['flux_g'],
            'flux_r': saved_data['flux_r'],
            'flux_i': saved_data['flux_i'],
            'flux_z': saved_data['flux_z']
        }

    else:
        print("Generating activations and predictions...")
        activations_np_flat, baseline_flux_predictions = get_aion_activations_and_flux_predictions(
            image_tensor, model, codec_manager, dataset['bands'], 
            block_idx=block_idx, batch_size=batch_size, save_path=activations_path
        )
    
    # Load SAE using existing pattern
    input_size = activations_np_flat.shape[1]
    hidden_size = input_size * 4
    k = max(1, int(hidden_size * 0.02))  # Use 2% sparsity by default
    sae_model = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    sae_model.load_state_dict(torch.load(sae_weights_path, weights_only=True, map_location=device))
    sae_model.eval()
    
    # 3. EXTRACT THE DECODER VECTOR FOR TARGET LATENT
    print(f"SAE decoder weight shape: {sae_model.decoder.weight.shape}")
    
    feature_decoder_vector = sae_model.decoder.weight[:, target_latent]  # Shape: (768,)    
    print(f"Feature decoder vector shape: {feature_decoder_vector.shape}")
    print(f"Feature decoder vector norm: {torch.norm(feature_decoder_vector).item():.6f}")
    
    # 4. DEFINE THE INTERVENTION HOOK WITH ACTIVATION TRACKING AND PROPER NORMALIZATION
    def create_steering_hook_with_tracking(amp_factor, decoder_vector):
        """Create a steering hook that tracks both original and intervened activations with proper normalization."""
        # Storage for activation analysis
        original_activations = []
        intervened_activations = []
        
        def steering_hook(module, input, output):
            # Get original activations
            original_activation = output  # Shape: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = original_activation.shape
            
            # Store original activation for analysis (detached copy)
            original_activations.append(original_activation.detach().cpu())
            
            # Flatten for normalization
            flat_activations = original_activation.view(-1, hidden_dim)
            
            # Apply same normalization as training
            # Mean subtraction per sample (dim=1)
            flat_activations_centered = flat_activations - flat_activations.mean(dim=1, keepdim=True)
            
            # Unit normalization
            norms = torch.norm(flat_activations_centered, dim=1, keepdim=True)
            flat_activations_normalized = flat_activations_centered / (norms + 1e-8)
            
            # Apply intervention in normalized space
            # The decoder_vector is learned in normalized space, so we apply amplification there
            intervened_flat_normalized = flat_activations_normalized - amp_factor * decoder_vector
            
            # Denormalize back to original space
            original_means = flat_activations.mean(dim=1, keepdim=True)
            intervened_flat = intervened_flat_normalized * norms + original_means
            
            # Reshape back to original format
            modified_activation = intervened_flat.view(batch_size, seq_len, hidden_dim)
            
            # Store intervened activation for analysis (detached copy)
            intervened_activations.append(modified_activation.detach().cpu())
            
            return modified_activation
        
        # Attach storage to the hook function for later access
        steering_hook.original_activations = original_activations
        steering_hook.intervened_activations = intervened_activations
        
        return steering_hook
    
    # 5. RUN INTERVENTION EXPERIMENT
    # NOTE: Using much smaller factors now since we work in normalized space
    # The old factors (300-400) were needed because decoder vectors were being applied to wrong scale
    amplification_factors = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] 
    
    # Store results for plotting
    all_results = {}
    activation_analysis = {}  # Store activation similarity metrics
    
    for amp_factor in amplification_factors:
        print(f"\n=== Testing amplification factor: {amp_factor} ===")
        
        # Create the hook with the current amplification factor
        hook_fn = create_steering_hook_with_tracking(amp_factor, feature_decoder_vector)
        
        # Register hook on the AION decoder block where SAE was trained
        handle = model.decoder[block_idx].register_forward_hook(hook_fn)
        
        try:
            # Get flux predictions with intervention using the same pattern as get_aion_activations_and_flux_predictions
            intervened_flux_predictions = {}
            target_modalities = {
                'flux_g': LegacySurveyFluxG,
                'flux_r': LegacySurveyFluxR,
                'flux_i': LegacySurveyFluxI,
                'flux_z': LegacySurveyFluxZ,
            }
            
            all_predicted_fluxes = {name: [] for name in target_modalities.keys()}
            
            # Process in batches
            for i in tqdm(range(0, len(image_tensor), batch_size), desc=f"Intervening (amp={amp_factor})"):
                batch_img = image_tensor[i:i+batch_size]
                img_mod = LegacySurveyImage(flux=batch_img, bands=dataset['bands'])
                
                tokens = codec_manager.encode(img_mod)
                # Debug prints moved outside batch loop to avoid spam
                if i == 0:  # Only print for first batch
                    print(f"Tokens keys: {tokens.keys()}")
                    print(f"Tokens type: {type(tokens)}")
                    if isinstance(tokens, dict):
                        for key, value in tokens.items():
                            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"  Direct tensor: shape {tokens.shape}, dtype {tokens.dtype}")
                
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
            
            # Concatenate predictions
            for modality_name in target_modalities.keys():
                predicted_flux = np.concatenate(all_predicted_fluxes[modality_name])
                intervened_flux_predictions[modality_name] = predicted_flux
            
            # Store results
            all_results[amp_factor] = intervened_flux_predictions
            
            # ACTIVATION ANALYSIS - Compute cosine similarity and norm differences
            if len(hook_fn.original_activations) > 0 and len(hook_fn.intervened_activations) > 0:
                # Concatenate all activations from all batches
                orig_acts = torch.cat(hook_fn.original_activations, dim=0)  # [total_samples, seq_len, hidden_dim]
                inter_acts = torch.cat(hook_fn.intervened_activations, dim=0)  # [total_samples, seq_len, hidden_dim]
                
                # Flatten to [total_samples * seq_len, hidden_dim] for analysis
                orig_acts_flat = orig_acts.view(-1, orig_acts.shape[-1])  # [N, hidden_dim]
                inter_acts_flat = inter_acts.view(-1, inter_acts.shape[-1])  # [N, hidden_dim]
                
                # Compute cosine similarities
                cosine_sims = torch.nn.functional.cosine_similarity(orig_acts_flat, inter_acts_flat, dim=1)
                mean_cosine_sim = cosine_sims.mean().item()
                std_cosine_sim = cosine_sims.std().item()
                
                # Compute norm differences
                orig_norms = torch.norm(orig_acts_flat, dim=1)
                inter_norms = torch.norm(inter_acts_flat, dim=1)
                norm_diffs = (inter_norms - orig_norms).abs()
                mean_norm_diff = norm_diffs.mean().item()
                std_norm_diff = norm_diffs.std().item()
                
                # Compute relative norm change
                relative_norm_change = (norm_diffs / (orig_norms + 1e-8)).mean().item()
                
                activation_analysis[amp_factor] = {
                    'mean_cosine_sim': mean_cosine_sim,
                    'std_cosine_sim': std_cosine_sim,
                    'mean_norm_diff': mean_norm_diff,
                    'std_norm_diff': std_norm_diff,
                    'relative_norm_change': relative_norm_change,
                    'mean_orig_norm': orig_norms.mean().item(),
                    'mean_inter_norm': inter_norms.mean().item()
                }
                
                print(f"Activation Analysis for amp={amp_factor}:")
                print(f"  Mean cosine similarity: {mean_cosine_sim:.4f} ± {std_cosine_sim:.4f}")
                print(f"  Mean norm difference: {mean_norm_diff:.4f} ± {std_norm_diff:.4f}")
                print(f"  Relative norm change: {relative_norm_change:.4f}")
                print(f"  Original norm: {orig_norms.mean().item():.4f}, Intervened norm: {inter_norms.mean().item():.4f}")
            
            # Compare predictions
            for band in ['flux_g', 'flux_r', 'flux_i', 'flux_z']:
                baseline_flux = baseline_flux_predictions[band]
                intervened_flux = intervened_flux_predictions[band]
                
                # Calculate percentage change with division by zero protection
                with np.errstate(divide='ignore', invalid='ignore'):
                    percent_change = np.where(baseline_flux != 0, 
                                            ((intervened_flux - baseline_flux) / baseline_flux) * 100, 
                                            0)
                
                print(f"{band}:")
                print(f"  Mean baseline: {np.mean(baseline_flux):.6f}")
                print(f"  Mean intervened: {np.mean(intervened_flux):.6f}")
                print(f"  Mean % change: {np.mean(percent_change[np.isfinite(percent_change)]):.2f}%")
                print(f"  Max % change: {np.max(np.abs(percent_change[np.isfinite(percent_change)])):.2f}%")
        
        finally:
            # Always remove the hook
            handle.remove()
    
    # 6. PLOT ACTIVATION ANALYSIS
    print("\n=== Creating Activation Analysis Plots ===")
    
    # Extract metrics for plotting
    amp_factors_for_plot = [amp for amp in amplification_factors if amp in activation_analysis]
    cosine_sims = [activation_analysis[amp]['mean_cosine_sim'] for amp in amp_factors_for_plot]
    cosine_stds = [activation_analysis[amp]['std_cosine_sim'] for amp in amp_factors_for_plot]
    norm_diffs = [activation_analysis[amp]['mean_norm_diff'] for amp in amp_factors_for_plot]
    norm_stds = [activation_analysis[amp]['std_norm_diff'] for amp in amp_factors_for_plot]
    rel_norm_changes = [activation_analysis[amp]['relative_norm_change'] for amp in amp_factors_for_plot]
    
    # Create activation analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cosine similarity plot
    axes[0, 0].errorbar(amp_factors_for_plot, cosine_sims, yerr=cosine_stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
    axes[0, 0].set_xlabel('Amplification Factor')
    axes[0, 0].set_ylabel('Mean Cosine Similarity')
    axes[0, 0].set_title('Cosine Similarity: Original vs Intervened Activations')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)
    
    # Add reference lines for cosine similarity
    axes[0, 0].axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='High similarity (0.9)')
    axes[0, 0].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Moderate similarity (0.7)')
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Low similarity (0.5)')
    axes[0, 0].legend(fontsize=8)
    
    # Norm difference plot
    axes[0, 1].errorbar(amp_factors_for_plot, norm_diffs, yerr=norm_stds,
                       marker='o', capsize=5, capthick=2, linewidth=2)
    axes[0, 1].set_xlabel('Amplification Factor')
    axes[0, 1].set_ylabel('Mean Norm Difference')
    axes[0, 1].set_title('Norm Difference: |norm(intervened) - norm(original)|')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative norm change plot
    axes[1, 0].plot(amp_factors_for_plot, rel_norm_changes, 'o-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Amplification Factor')
    axes[1, 0].set_ylabel('Relative Norm Change')
    axes[1, 0].set_title('Relative Norm Change: |Δnorm| / norm(original)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined plot: Cosine sim vs norm change
    scatter = axes[1, 1].scatter(cosine_sims, rel_norm_changes, 
                                c=amp_factors_for_plot, cmap='viridis', s=100)
    axes[1, 1].set_xlabel('Mean Cosine Similarity')
    axes[1, 1].set_ylabel('Relative Norm Change')
    axes[1, 1].set_title('Cosine Similarity vs Norm Change')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Amplification Factor')
    
    # Add annotations for each point
    for i, amp in enumerate(amp_factors_for_plot):
        axes[1, 1].annotate(f'{amp}', (cosine_sims[i], rel_norm_changes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'activation_analysis_latent{target_latent}_amp300400.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # # 7. VALIDATION TESTS (as your advisor suggested)
    # print("\n=== Validation Tests ===")
    
    # # Test a=0 is no-op
    # try:
    #     assert np.allclose(baseline_flux_predictions['flux_g'], 
    #                       all_results[0]['flux_g'], rtol=1e-5), "a=0 should be no-op"
    #     print("✓ Test passed: a=0 is no-op")
    # except AssertionError as e:
    #     print(f"✗ Test failed: {e}")
   
    # 8. PLOT THE PREDICTED VS TRUE COMPARISON
    print("\n=== Creating flux prediction plots ===")
    
    bands = ['flux_g', 'flux_r', 'flux_i', 'flux_z']
    band_labels = ['G-band', 'R-band', 'I-band', 'Z-band']
    
    # 9. PLOT THE AMPLIFICATION EFFECT - CREATE SEPARATE PLOTS FOR EACH AMPLIFICATION
    print("Creating separate plots for each amplification factor...")
    
    for amp_factor in amplification_factors:
        plt.figure(figsize=(15, 10))
        
        for i, (band, label) in enumerate(zip(bands, band_labels)):
            plt.subplot(2, 2, i+1)
            
            true_flux = true_flux_data[band]
            pred_flux = all_results[amp_factor][band]
            
            # Plot true vs predicted for this amplification factor
            plt.scatter(true_flux, pred_flux, alpha=0.6, s=20)
            plt.plot([0, max(true_flux)], [0, max(true_flux)], 'k--', alpha=0.7, label='Perfect prediction')
            
            # Calculate correlation
            r = np.corrcoef(true_flux, pred_flux)[0, 1]
            
            plt.xlabel(f'True {label} Flux')
            plt.ylabel(f'Predicted {label} Flux')
            plt.title(f'{label}: amp={amp_factor}, r={r:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Amplification Factor = {amp_factor}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'intervention_amp{amp_factor}_latent{target_latent}.png'), dpi=300)
        plt.show()
    
    # 10. PLOT AMPLIFICATION VS MEAN SHIFT
    plt.figure(figsize=(12, 8))
    
    for i, (band, label) in enumerate(zip(bands, band_labels)):
        shifts = []
        baseline_flux = all_results[0][band]
        
        for amp_factor in amplification_factors:
            intervened_flux = all_results[amp_factor][band]
            mean_shift = np.mean(intervened_flux - baseline_flux)
            shifts.append(mean_shift)
        
        plt.subplot(2, 2, i+1)
        
        # Handle log scale for x-axis, treating 0 specially
        # Use a small offset for zero to make it visible on log scale
        x_values = [0.1 if amp == 0 else amp for amp in amplification_factors]
        
        plt.semilogx(x_values, shifts, 'o-', linewidth=2, markersize=6)
        
        # Custom x-tick labels to show the actual amplification factors
        plt.xticks(x_values, [str(amp) for amp in amplification_factors])
        
        plt.xlabel('Amplification Factor (log scale)')
        plt.ylabel('Mean Flux Shift')
        plt.title(f'{label}: Amplification vs Mean Shift')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'amplification_vs_shift_latent{target_latent}_amp300400.png'), dpi=300)
    plt.show()
    
    # Save numerical results
    results_summary = {
        'amplification_factors': amplification_factors,
        'all_results': all_results,
        'true_flux_data': true_flux_data,
        'baseline_predictions': baseline_flux_predictions,
        'target_latent': target_latent,
        'block_idx': block_idx,
        'activation_analysis': activation_analysis,  # Add activation analysis to saved results
        'decoder_vector_norm': torch.norm(feature_decoder_vector).item()
    }
    # np.save(os.path.join(save_dir, f'intervention_results_latent{target_latent}.npy'), results_summary)
    np.save(os.path.join(save_dir, f'intervention_results_latent{target_latent}_amp300400.npy'), results_summary)
    
    print(f"\nIntervention experiment complete! Results saved to {save_dir}")
    return all_results, true_flux_data

def ablate_all_sae_features(
    sae_weights_path='best_llm_sae_rural-wood-16.pth',
    save_dir='feature_ablation_results',
    batch_size=32,
    block_idx=8,
    device=None,
    max_features_to_test=None,
    subset_indices=None,
    sample_size=10000
):
    """
    Systematically ablate each SAE feature and measure impact on flux predictions.
    This helps identify which features are most important for flux prediction.
    
    Args:
        sae_weights_path: Path to trained SAE weights
        save_dir: Directory to save results
        batch_size: Batch size for inference
        block_idx: AION decoder block index where SAE was trained
        device: Compute device
        max_features_to_test: Limit number of features to test (for faster experiments)
        subset_indices: Specific list of feature indices to test (overrides max_features_to_test)
        sample_size: Number of samples to use for testing (smaller = faster)
    
    Returns:
        Dictionary with ablation results and impact rankings
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=== SAE Feature Ablation Experiment ===")
    print(f"Save directory: {save_dir}")
    print(f"Device: {device}")
    print(f"Sample size: {sample_size}")
    
    # Load models
    print("Loading models...")
    model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
    codec_manager = CodecManager(device=device)
    
    # Load dataset (use smaller subset for faster testing)
    print("Loading dataset...")
    data_path = '/mnt/home/polymathic/ceph/MultimodalUniverse/legacysurvey/dr10_south_21/healpix=299/001-of-001.hdf5'
    dataset = load_legacysurvey_data(data_path, max_samples=sample_size)
    
    # Convert images to tensor and move to device
    image_tensor = torch.tensor(dataset['images'], dtype=torch.float32).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Load SAE model - first determine input size
    print("Loading SAE...")
    with torch.no_grad():
        sample_img = image_tensor[:1]
        img_mod = LegacySurveyImage(flux=sample_img, bands=dataset['bands'])
        tokens = codec_manager.encode(img_mod)
        
        # Get sample activation to determine size
        sample_activation = None
        def size_hook(module, input, output):
            nonlocal sample_activation
            sample_activation = output.detach()
        
        handle = model.decoder[block_idx].register_forward_hook(size_hook)
        _ = model(tokens, target_modality=LegacySurveyFluxG)
        handle.remove()
        
        input_size = sample_activation.shape[-1]
        print(f"Determined input size: {input_size}")
    
    # Load SAE model
    hidden_size = input_size * 4
    k = max(1, int(hidden_size * 0.02))
    sae_model = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    sae_model.load_state_dict(torch.load(sae_weights_path, weights_only=True, map_location=device))
    sae_model.eval()
    
    print(f"SAE loaded - Input: {input_size}, Hidden: {hidden_size}, k: {k}")
    
    # Determine which features to test
    if subset_indices is not None:
        features_to_test = subset_indices
        print(f"Testing specific feature indices: {len(features_to_test)} features")
    elif max_features_to_test is not None:
        features_to_test = list(range(min(max_features_to_test, hidden_size)))
        print(f"Testing first {len(features_to_test)} features")
    else:
        features_to_test = list(range(hidden_size))
        print(f"Testing all {len(features_to_test)} features")
    
    # Define target modalities
    target_modalities = {
        'flux_g': LegacySurveyFluxG,
        'flux_r': LegacySurveyFluxR,
        'flux_i': LegacySurveyFluxI,
        'flux_z': LegacySurveyFluxZ,
    }
    
    # 1. GET BASELINE PREDICTIONS (no ablation)
    print("\nGetting baseline predictions (no ablation)...")
    baseline_predictions = {}
    
    for modality_name, modality_class in target_modalities.items():
        all_predicted_fluxes = []
        
        for i in tqdm(range(0, len(image_tensor), batch_size), desc=f"Baseline {modality_name}"):
            batch_img = image_tensor[i:i+batch_size]
            img_mod = LegacySurveyImage(flux=batch_img, bands=dataset['bands'])
            tokens = codec_manager.encode(img_mod)
            
            with torch.no_grad():
                preds = model(tokens, target_modality=modality_class)
                raw_logits = preds[modality_class.token_key]
                token_indices = raw_logits.argmax(dim=-1)
                tokens_for_decode = {modality_class.token_key: token_indices}
                pred_values = codec_manager.decode(tokens_for_decode, modality_class)
                flux_values = pred_values.value.squeeze().cpu().numpy()
                all_predicted_fluxes.append(flux_values)
        
        baseline_predictions[modality_name] = np.concatenate(all_predicted_fluxes)
    
    print("Baseline predictions computed:")
    for modality_name, flux_values in baseline_predictions.items():
        print(f"  {modality_name}: {len(flux_values)} samples, range [{flux_values.min():.6f}, {flux_values.max():.6f}]")
    
    # 2. ABLATION EXPERIMENT - Test each feature
    print(f"\nStarting ablation experiment for {len(features_to_test)} features...")
    
    ablation_results = {}
    impact_metrics = {
        'feature_idx': [],
        'flux_g_mse': [],
        'flux_r_mse': [],
        'flux_i_mse': [],
        'flux_z_mse': [],
        'flux_g_mae': [],
        'flux_r_mae': [],
        'flux_i_mae': [],
        'flux_z_mae': [],
        'total_mse': [],
        'total_mae': []
    }
    
    def create_ablation_hook(target_feature_idx, sae_model):
        """Create a hook that ablates a specific SAE feature using activation patching with proper normalization."""
        def ablation_hook(module, input, output):
            # Get original activations
            original_activations = output  # Shape: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = original_activations.shape
            
            # Flatten activations for SAE
            flat_activations = original_activations.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
            
            with torch.no_grad():
                # STEP 1: Apply same normalization as training
                # Mean subtraction per sample (dim=1)
                flat_activations_centered = flat_activations - flat_activations.mean(dim=1, keepdim=True)
                
                # Unit normalization
                norms = torch.norm(flat_activations_centered, dim=1, keepdim=True)
                flat_activations_normalized = flat_activations_centered / (norms + 1e-8)
                
                # STEP 2: Pass through SAE encoder to get latents
                latents = sae_model.encoder(flat_activations_normalized)  # [batch*seq_len, sae_hidden_dim]
                
                # STEP 3: Get the decoder vector for the target feature
                feature_decoder_vector = sae_model.decoder.weight[:, target_feature_idx]  # Shape: [hidden_dim]
                
                # STEP 4: Compute the contribution of this specific feature (in normalized space)
                feature_activations = latents[:, target_feature_idx:target_feature_idx+1]  # [batch*seq_len, 1]
                feature_contribution_normalized = feature_activations @ feature_decoder_vector.unsqueeze(0)  # [batch*seq_len, hidden_dim]
                
                # STEP 5: Subtract the feature's contribution in normalized space
                ablated_flat_normalized = flat_activations_normalized - feature_contribution_normalized
                
                # STEP 6: Denormalize back to original space
                # Multiply by original norms and add back original means
                original_means = flat_activations.mean(dim=1, keepdim=True)
                ablated_flat = ablated_flat_normalized * norms + original_means
                
                # STEP 7: Reshape back to original format
                ablated_activations = ablated_flat.view(batch_size, seq_len, hidden_dim)
            
            return ablated_activations
        
        return ablation_hook
    
    # Test each feature
    for i, feature_idx in enumerate(tqdm(features_to_test, desc="Testing features")):
        if (i + 1) % 100 == 0:
            print(f"\nTesting feature {feature_idx} ({i+1}/{len(features_to_test)})...")
        
        # Create ablation hook for this feature
        hook_fn = create_ablation_hook(feature_idx, sae_model)
        
        # Register hook
        handle = model.decoder[block_idx].register_forward_hook(hook_fn)
        
        try:
            # Get predictions with this feature ablated
            ablated_predictions = {}
            
            for modality_name, modality_class in target_modalities.items():
                all_predicted_fluxes = []
                
                for j in range(0, len(image_tensor), batch_size):
                    batch_img = image_tensor[j:j+batch_size]
                    img_mod = LegacySurveyImage(flux=batch_img, bands=dataset['bands'])
                    tokens = codec_manager.encode(img_mod)
                    
                    with torch.no_grad():
                        preds = model(tokens, target_modality=modality_class)
                        raw_logits = preds[modality_class.token_key]
                        token_indices = raw_logits.argmax(dim=-1)
                        tokens_for_decode = {modality_class.token_key: token_indices}
                        pred_values = codec_manager.decode(tokens_for_decode, modality_class)
                        flux_values = pred_values.value.squeeze().cpu().numpy()
                        all_predicted_fluxes.append(flux_values)
                
                ablated_predictions[modality_name] = np.concatenate(all_predicted_fluxes)
            
            # Calculate impact metrics (how much did ablating this feature change predictions?)
            total_mse = 0.0
            total_mae = 0.0
            
            for modality_name in target_modalities.keys():
                baseline_flux = baseline_predictions[modality_name]
                ablated_flux = ablated_predictions[modality_name]
                
                # Calculate MSE and MAE
                mse = np.mean((baseline_flux - ablated_flux) ** 2)
                mae = np.mean(np.abs(baseline_flux - ablated_flux))
                
                impact_metrics[f'{modality_name}_mse'].append(mse)
                impact_metrics[f'{modality_name}_mae'].append(mae)
                
                total_mse += mse
                total_mae += mae
            
            impact_metrics['feature_idx'].append(feature_idx)
            impact_metrics['total_mse'].append(total_mse)
            impact_metrics['total_mae'].append(total_mae)
            
            # Store detailed results for significant features (to save memory)
            if total_mse > 0:  # Only store if there's any impact
                ablation_results[feature_idx] = {
                    'ablated_predictions': ablated_predictions,
                    'total_mse': total_mse,
                    'total_mae': total_mae
                }
            
            # Print progress for significant features
            if len(impact_metrics['total_mse']) > 10:  # After we have some baseline
                recent_median = np.median(impact_metrics['total_mse'][-100:])  # Recent median
                if total_mse > recent_median * 5:  # Significantly above recent median
                    print(f"  🔥 Feature {feature_idx}: Total MSE = {total_mse:.6f}, Total MAE = {total_mae:.6f}")
        
        finally:
            # Always remove hook
            handle.remove()
    
    # 3. ANALYZE RESULTS AND CREATE RANKINGS
    print("\nAnalyzing results...")
    
    # Convert to DataFrame for easier analysis
    df_metrics = pd.DataFrame(impact_metrics)
    
    # Sort by total impact
    df_metrics_sorted = df_metrics.sort_values('total_mse', ascending=False)
    
    print("\nTop 20 most impactful features (by Total MSE):")
    print(df_metrics_sorted.head(20)[['feature_idx', 'total_mse', 'total_mae']])
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Features tested: {len(features_to_test)}")
    print(f"  Features with impact > 0: {(df_metrics['total_mse'] > 0).sum()}")
    print(f"  Mean impact (MSE): {df_metrics['total_mse'].mean():.8f}")
    print(f"  Std impact (MSE): {df_metrics['total_mse'].std():.8f}")
    print(f"  Max impact (MSE): {df_metrics['total_mse'].max():.6f}")
    print(f"  Top 1% threshold: {df_metrics['total_mse'].quantile(0.99):.6f}")
    
    # Save detailed results
    results_summary = {
        'features_tested': features_to_test,
        'impact_metrics': impact_metrics,
        'baseline_predictions': baseline_predictions,
        'ablation_results': {k: v for k, v in ablation_results.items() if v['total_mse'] > df_metrics['total_mse'].quantile(0.95)},  # Only top 5%
        'top_features': df_metrics_sorted.head(50)['feature_idx'].tolist(),
        'experiment_config': {
            'sae_weights_path': sae_weights_path,
            'block_idx': block_idx,
            'batch_size': batch_size,
            'sample_size': sample_size,
            'features_tested': len(features_to_test),
            'input_size': input_size,
            'hidden_size': hidden_size
        }
    }
    
    # Save results
    results_file = os.path.join(save_dir, 'feature_ablation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"Detailed results saved to {results_file}")
    
    # Save CSV of metrics for easy analysis
    csv_file = os.path.join(save_dir, 'feature_impact_metrics.csv')
    df_metrics_sorted.to_csv(csv_file, index=False)
    print(f"Impact metrics CSV saved to {csv_file}")
    
    # 4. VISUALIZATION
    print("\nCreating visualizations...")
    
    # Create a single figure with 4 panels as requested
    plt.figure(figsize=(12, 8))
    
    # Panel 1: Histogram of MSE values per feature
    plt.subplot(2, 2, 1)
    plt.hist(impact_metrics['total_mse'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Total MSE Impact')
    plt.ylabel('Number of Features')
    plt.title('Distribution of MSE Impact per Feature')
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Histogram of MAE values per feature  
    plt.subplot(2, 2, 2)
    plt.hist(impact_metrics['total_mae'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Total MAE Impact')
    plt.ylabel('Number of Features')
    plt.title('Distribution of MAE Impact per Feature')
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Scatter plot of latent feature # vs. MSE value
    plt.subplot(2, 2, 3)
    plt.scatter(impact_metrics['feature_idx'], impact_metrics['total_mse'], alpha=0.6, s=20)
    plt.xlabel('Latent Feature Index')
    plt.ylabel('Total MSE Impact')
    plt.title('Feature Index vs MSE Impact')
    plt.grid(True, alpha=0.3)
    
    # Panel 4: Scatter plot of latent feature # vs. MAE value
    plt.subplot(2, 2, 4)
    plt.scatter(impact_metrics['feature_idx'], impact_metrics['total_mae'], alpha=0.6, s=20)
    plt.xlabel('Latent Feature Index')
    plt.ylabel('Total MAE Impact')
    plt.title('Feature Index vs MAE Impact')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_ablation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFeature ablation experiment complete!")
    print(f"Results saved to {save_dir}")
    print(f"Most impactful feature: {df_metrics_sorted.iloc[0]['feature_idx']} (MSE impact: {df_metrics_sorted.iloc[0]['total_mse']:.6f})")
    print(f"Top 10 most impactful features: {df_metrics_sorted.head(10)['feature_idx'].tolist()}")
    
    return results_summary


def simple_unembedding_check(
    sae_weights_path='best_llm_sae_rural-wood-16.pth',
    target_latent=1689,
    device=None
):
    """
    Simple check to understand dimensions and verify the concept of comparing
    SAE features with unembedding vectors.
    
    IMPORTANT ARCHITECTURAL NOTE:
    - SAE feature: Trained on activations from decoder block 8 (dim: 768)
    - After block 8: Data flows through blocks 9, 10, 11, then decoder_norm
    - Unembedding input: The normalized output after all blocks (dim: 768)
    - Unembedding matrix: [vocab_size, 768]
    
    The comparison is meaningful because:
    1. Residual connections preserve core information through later blocks
    2. LayerNorm only rescales, doesn't rotate vector space
    3. We're checking if the SAE feature at block 8 is already aligned with
       the "flux direction" that will be enhanced by final blocks
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== Simple Unembedding Dimension Check ===")
    print(f"Target latent: {target_latent}")
    print(f"\nArchitectural Flow:")
    print(f"  1. SAE hook location: Decoder block 8 output")
    print(f"  2. Remaining processing: Blocks 9, 10, 11 + decoder_norm")
    print(f"  3. Then: Unembedding for each modality")
    
    # Load AION model
    print("\nLoading AION model...")
    model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
    
    # Print model architecture info
    print(f"  Decoder blocks: {len(model.decoder)} blocks")
    print(f"  Model dimension: {model.dim}")
    print(f"  Decoder norm: {model.decoder_norm}")
    
    # Load SAE model
    print("Loading SAE model...")
    input_size = 768  # AION's hidden dimension
    hidden_size = input_size * 4
    k = max(1, int(hidden_size * 0.02))
    
    sae_model = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    sae_model.load_state_dict(torch.load(sae_weights_path, weights_only=True, map_location=device))
    sae_model.eval()
    
    # Get SAE feature decoder vector
    feature_vector = sae_model.decoder.weight[:, target_latent]  # Shape: [768]
    print(f"\nSAE feature decoder vector:")
    print(f"  Shape: {feature_vector.shape}")
    print(f"  Norm: {torch.norm(feature_vector).item():.6f}")
    print(f"  Sample values: {feature_vector[:5].tolist()}")
    
    # Check flux modalities
    flux_modalities = {
        'flux_g': LegacySurveyFluxG,
        'flux_r': LegacySurveyFluxR,
        'flux_i': LegacySurveyFluxI,
        'flux_z': LegacySurveyFluxZ,
    }
    
    print(f"\nChecking unembedding matrices for flux modalities...")
    
    overall_max_similarity = 0.0
    overall_results = {}
    
    for band_name, modality_class in flux_modalities.items():
        print(f"\n{band_name} ({modality_class.__name__}):")
        
        # Get token key
        token_key = modality_class.token_key
        print(f"  Token key: {token_key}")
        
        # Check if this modality exists in the model
        if token_key not in model.decoder_embeddings:
            print(f"  ❌ Not found in decoder_embeddings")
            continue
        
        # Get the decoder embedding
        decoder_emb = model.decoder_embeddings[token_key]
        print(f"  Decoder embedding type: {type(decoder_emb).__name__}")
        
        # Get the unembedding matrix
        if hasattr(decoder_emb, 'to_logits'):
            unembedding_matrix = decoder_emb.to_logits.weight  # Shape: [vocab_size, hidden_dim]
            vocab_size, hidden_dim = unembedding_matrix.shape
            print(f"  ✅ Unembedding matrix shape: {unembedding_matrix.shape}")
            print(f"     Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")
            
            # Check dimensional compatibility
            if hidden_dim == feature_vector.shape[0]:
                print(f"  ✅ Dimensions compatible! (both {hidden_dim})")
                
                # Compute a few example similarities
                with torch.no_grad():
                    feature_norm = feature_vector / torch.norm(feature_vector)
                    
                    # Check similarity with first few unembedding vectors
                    print(f"     Sample similarities with first few tokens:")
                    for i in range(min(5, vocab_size)):
                        unemb_vec = unembedding_matrix[i, :]
                        if torch.norm(unemb_vec) > 1e-8:  # Skip zero vectors
                            unemb_norm = unemb_vec / torch.norm(unemb_vec)
                            similarity = torch.dot(feature_norm, unemb_norm).item()
                            print(f"       Token {i}: cosine similarity = {similarity:.4f}")
                    
                    # Find max similarity across all tokens
                    unemb_norms = torch.norm(unembedding_matrix, dim=1)
                    valid_tokens = unemb_norms > 1e-8
                    if valid_tokens.sum() > 0:
                        valid_unembedding = unembedding_matrix[valid_tokens]
                        valid_unembedding_norm = valid_unembedding / torch.norm(valid_unembedding, dim=1, keepdim=True)
                        all_similarities = torch.matmul(valid_unembedding_norm, feature_norm)
                        max_sim = torch.max(all_similarities).item()
                        max_idx_in_valid = torch.argmax(all_similarities).item()
                        # Convert back to original index
                        valid_indices = torch.where(valid_tokens)[0]
                        max_idx = valid_indices[max_idx_in_valid].item()
                        
                        print(f"     Maximum similarity: {max_sim:.4f} (token {max_idx})")
                        print(f"     Mean abs similarity: {torch.mean(torch.abs(all_similarities)).item():.4f}")
                        print(f"     Std of similarities: {torch.std(all_similarities).item():.4f}")
                        
                        overall_max_similarity = max(overall_max_similarity, max_sim)
                        overall_results[band_name] = {
                            'max_similarity': max_sim,
                            'max_token': max_idx,
                            'mean_abs_similarity': torch.mean(torch.abs(all_similarities)).item(),
                            'std_similarity': torch.std(all_similarities).item(),
                            'vocab_size': vocab_size
                        }
                    else:
                        print(f"     ⚠️ All unembedding vectors are zero!")
                        
            else:
                print(f"  ❌ Dimension mismatch! Unembedding: {hidden_dim}, SAE feature: {feature_vector.shape[0]}")
        else:
            print(f"  ❌ No 'to_logits' attribute found")
    
    print(f"\n=== Summary ===")
    print("The concept: If SAE feature decoder vector has high cosine similarity")
    print("with any unembedding vector, then the SAE feature might just be")
    print("learning to 'push towards' that output token, which would be trivial.")
    print("\nArchitectural consideration: The SAE feature is from block 8,")
    print("but unembedding happens after blocks 9-11 + norm. However, residual")
    print("connections mean the core direction may already be present at block 8.")
    
    if overall_results:
        print(f"\nResults across all bands:")
        for band, results in overall_results.items():
            print(f"  {band}: max_sim={results['max_similarity']:.4f}, mean_abs={results['mean_abs_similarity']:.4f}")
        
        print(f"\nOverall maximum similarity: {overall_max_similarity:.4f}")
        
        if overall_max_similarity > 0.8:
            print("⚠️  HIGH SIMILARITY: Feature may be trivially aligned with unembedding")
        elif overall_max_similarity > 0.5:
            print("⚡ MODERATE SIMILARITY: Worth investigating further")
        else:
            print("✅ LOW SIMILARITY: Feature likely captures more complex patterns")
    
    return model, sae_model, feature_vector, overall_results

if __name__ == '__main__':
    # probe()

    # print("Plotting latent neuron 2023 vs all 4 bands using convenience function...")
    # correlation_results = plot_specific_latent_vs_bands(
    #     latent_idx=2023,
    #     save_dir='probe_flux_in_sae_image_to_flux_results'
    # )

    # First, check if the SAE feature is aligned with unembedding directions
    # print("Running unembedding similarity analysis...")
    # similarity_results = analyze_sae_feature_vs_unembedding_similarity(
    #     target_latent=1689,
    #     save_dir='unembedding_analysis_results'
    # )
    # model, sae_model, feature_vector, overall_results = simple_unembedding_check(target_latent=1689)

    # Run intervention experiment
    # print("Running intervention experiment...")
    # results, true_data = intervention_experiment()
    
    
    # NEW: Test the FIXED SAE feature ablation experiment with proper normalization
    # Test with small sample to verify the normalization fix works
    print("Testing FIXED SAE feature ablation experiment with proper normalization...")
    ablation_results = ablate_all_sae_features(
        max_features_to_test=None, 
        sample_size=100,  
    )