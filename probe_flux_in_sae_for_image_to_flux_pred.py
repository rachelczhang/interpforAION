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
    target_latent=2023,
    device=None
):
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
    dataset = load_legacysurvey_data(data_path, max_samples=200)
    
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
    
    # 4. DEFINE THE INTERVENTION HOOK
    def create_steering_hook(amp_factor, decoder_vector):
        """Create a steering hook with captured amplification factor and decoder vector."""
        def steering_hook(module, input, output):
            # module and input are required by PyTorch's hook signature but not used
            # output shape could be (batch_size, seq_len, 768) or (batch_size, 768)
            original_activation = output
            
            # Add the amplified feature direction
            modified_activation = original_activation + amp_factor * decoder_vector
            
            return modified_activation
        return steering_hook
    
    # 5. RUN INTERVENTION EXPERIMENT
    amplification_factors = [0, 1.0, 10.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Store results for plotting
    all_results = {}
    
    for amp_factor in amplification_factors:
        print(f"\n=== Testing amplification factor: {amp_factor} ===")
        
        # Create the hook with the current amplification factor
        hook_fn = create_steering_hook(amp_factor, feature_decoder_vector)
        
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
    
    # 6. VALIDATION TESTS (as your advisor suggested)
    print("\n=== Validation Tests ===")
    
    # Test a=0 is no-op
    try:
        assert np.allclose(baseline_flux_predictions['flux_g'], 
                          all_results[0]['flux_g'], rtol=1e-5), "a=0 should be no-op"
        print("✓ Test passed: a=0 is no-op")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    
    # Test a=100 creates significant changes
    max_change = np.max(np.abs(all_results[1000.0]['flux_g'] - baseline_flux_predictions['flux_g']))
    print(f"a=1000 creates changes (max change: {max_change:.3f})")
   
    # 7. PLOT THE PREDICTED VS TRUE COMPARISON
    print("\n=== Creating plots ===")
    
    bands = ['flux_g', 'flux_r', 'flux_i', 'flux_z']
    band_labels = ['G-band', 'R-band', 'I-band', 'Z-band']
    
    # 8. PLOT THE AMPLIFICATION EFFECT - CREATE SEPARATE PLOTS FOR EACH AMPLIFICATION
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
    
    # 9. PLOT AMPLIFICATION VS MEAN SHIFT
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
    plt.savefig(os.path.join(save_dir, f'amplification_vs_shift_latent{target_latent}.png'), dpi=300)
    plt.show()
    
    # Save numerical results
    results_summary = {
        'amplification_factors': amplification_factors,
        'all_results': all_results,
        'true_flux_data': true_flux_data,
        'baseline_predictions': baseline_flux_predictions,
        'target_latent': target_latent,
        'block_idx': block_idx
    }
    
    np.save(os.path.join(save_dir, f'intervention_results_latent{target_latent}.npy'), results_summary)
    
    print(f"\nIntervention experiment complete! Results saved to {save_dir}")
    return all_results, true_flux_data

if __name__ == '__main__':
    # probe()

    # print("Plotting latent neuron 2023 vs all 4 bands using convenience function...")
    # correlation_results = plot_specific_latent_vs_bands(
    #     latent_idx=2023,
    #     save_dir='probe_flux_in_sae_image_to_flux_results'
    # )

    results, true_data = intervention_experiment()