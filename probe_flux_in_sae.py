import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional
import pandas as pd
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyFluxG, HSCMagG
from sparse_autoencoder import SparseAutoencoder
from scipy.stats import pearsonr, spearmanr

def load_flux_data(path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess the flux data (nanomaggies)."""
    flux_g_data = torch.load(path, weights_only=True)
    flux_g_data = flux_g_data.float().to(device)
    # Only use positive fluxes for valid magnitude prediction
    positive_mask = flux_g_data > 0
    flux_g_data_pos = flux_g_data[positive_mask]
    print(f"Loaded flux data: shape={flux_g_data_pos.shape}, min={flux_g_data_pos.min().item()}, max={flux_g_data_pos.max().item()}")
    print(f"Sample flux values: {flux_g_data_pos[:5].cpu().numpy()}")
    return flux_g_data_pos

def get_aion_activations_from_flux(
    flux_g_data_pos: torch.Tensor,
    model: AION,
    codec_manager: CodecManager,
    block_idx: int = 8,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pass flux through AION, collect activations from the specified decoder block.
    Returns (activations, flux_values) as numpy arrays.
    """
    activations = []
    flux_values = []
    # Register a forward hook
    def _hook(module, input, output):
        activations.append(output.detach().cpu())
    handle = model.decoder[block_idx].register_forward_hook(_hook)
    
    # Run in batches
    for i in range(0, len(flux_g_data_pos), batch_size):
        batch_flux = flux_g_data_pos[i:i+batch_size]
        flux_mod = LegacySurveyFluxG(value=batch_flux)
        tokens = codec_manager.encode(flux_mod)
        with torch.no_grad():
            _ = model(tokens, target_modality=HSCMagG)
        flux_values.append(batch_flux.cpu())
    handle.remove()
    activations_np = torch.cat(activations, dim=0).numpy()  # shape: (N, 768)
    sample_size, seq_length, embed_dim = activations_np.shape
    activations_np_flat = activations_np.reshape(sample_size * seq_length, embed_dim)
    flux_np = torch.cat(flux_values, dim=0).numpy()  # shape: (N,)
    print(f"AION activations: shape={activations_np_flat.shape}")
    print(f"Sample activations[0]: {activations_np_flat[0][:5]}")
    if save_path is not None:
        np.savez(save_path, activations=activations_np_flat, flux=flux_np)
        print(f"Saved activations and flux to {save_path}")
    return activations_np_flat, flux_np

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
    flux: np.ndarray,
    plot: bool = True,
    top_k: int = 5,
    save_prefix: Optional[str] = None
):
	"""
	Compute Pearson correlation between each latent and both raw/log flux.
	Optionally plot top_k most correlated latents.
	"""
	log_flux = np.log10(flux)
	n_latents = latents.shape[1]
	results = []
	for i in range(n_latents):
		latent_col = latents[:, i]
		pearson_raw, _ = pearsonr(latent_col, flux)
		pearson_log, _ = pearsonr(latent_col, log_flux)
		spearman_raw, _ = spearmanr(latent_col, flux)
		
		# Calculate correlations only for non-zero activations
		results.append({
			'latent_idx': i,
			'pearson_raw': pearson_raw,
			'pearson_log': pearson_log,
			'spearman_raw': spearman_raw
		})
	
	# Filter out results with nan correlations
	valid_results = [r for r in results if not np.isnan(r['pearson_log'])]
	
	# Sort by absolute Pearson correlation with log(flux)
	results_sorted = sorted(valid_results, key=lambda x: abs(x['pearson_log']), reverse=True)
	
	if save_prefix is not None:
		pd.DataFrame(results).to_csv(f"{save_prefix}_latent_flux_correlation.csv", index=False)
		print(f"Saved correlation results to {save_prefix}_latent_flux_correlation.csv")
	
	print(f"Total latents analyzed: {len(results)}")
	print(f"Valid latents (non-nan): {len(valid_results)}")
	
	if len(valid_results) == 0:
		print("No valid correlations found (all latents are constant)")
		return results
	
	# Use min of top_k and available valid results
	actual_top_k = min(top_k, len(valid_results))
	print("Top-{} latent correlations with log10(flux):".format(actual_top_k))
	for j in range(actual_top_k):
		idx = results_sorted[j]['latent_idx']
		print(f" Latent {idx}: Pearson={results_sorted[j]['pearson_log']:.4f}, Spearman={results_sorted[j]['spearman_raw']:.4f}")

	# Print top 20 by pearson_raw
	print("\nTop 20 by Pearson correlation with raw flux:")
	top_20_pearson_raw = sorted(valid_results, key=lambda x: abs(x['pearson_raw']), reverse=True)[:20]
	for r in top_20_pearson_raw:
		print(f" Latent {r['latent_idx']}: Pearson_raw={r['pearson_raw']:.4f}, Pearson_log={r['pearson_log']:.4f}, Spearman_raw={r['spearman_raw']:.4f}")

	# Print top 20 by pearson_log
	print("\nTop 20 by Pearson correlation with log10(flux):")
	top_20_pearson_log = sorted(valid_results, key=lambda x: abs(x['pearson_log']), reverse=True)[:20]
	for r in top_20_pearson_log:
		print(f" Latent {r['latent_idx']}: Pearson_log={r['pearson_log']:.4f}, Pearson_raw={r['pearson_raw']:.4f}, Spearman_raw={r['spearman_raw']:.4f}")

	# Print top 20 by spearman_raw
	print("\nTop 20 by Spearman correlation with raw flux:")
	top_20_spearman_raw = sorted(valid_results, key=lambda x: abs(x['spearman_raw']), reverse=True)[:20]
	for r in top_20_spearman_raw:
		print(f" Latent {r['latent_idx']}: Spearman_raw={r['spearman_raw']:.4f}, Pearson_log={r['pearson_log']:.4f}, Pearson_raw={r['pearson_raw']:.4f}")
	
	all_corrs = [abs(r['pearson_log']) for r in valid_results]
	print(f"Correlation stats (valid only): min={np.min(all_corrs):.4f}, max={np.max(all_corrs):.4f}, mean={np.mean(all_corrs):.4f}")
	
	if plot and len(valid_results) > 0:
		for j in range(actual_top_k):
			idx = results_sorted[j]['latent_idx']
			
			# Plot raw flux vs latent
			plt.figure(figsize=(12, 4))
			
			plt.subplot(1, 2, 1)
			plt.scatter(flux, latents[:, idx], alpha=0.5)
			plt.xlabel('flux (nanomaggies)')
			plt.ylabel(f'Latent {idx}')
			plt.title(f'Latent {idx} vs raw flux\nPearson: {results_sorted[j]["pearson_raw"]:.3f}')
			
			plt.subplot(1, 2, 2)
			plt.scatter(log_flux, latents[:, idx], alpha=0.5)
			plt.xlabel('log10(flux)')
			plt.ylabel(f'Latent {idx}')
			plt.title(f'Latent {idx} vs log10(flux)\nPearson: {results_sorted[j]["pearson_log"]:.3f}')
			
			plt.tight_layout()
			if save_prefix is not None:
				plt.savefig(f"{save_prefix}_latent{idx}_vs_flux_both.png")
			plt.show()
		
		# Add scatter plot of all latent neurons vs. their absolute Pearson correlation with log(flux)
		plt.figure(figsize=(10, 4))
		abs_corrs = [abs(r['pearson_log']) for r in results]
		plt.scatter(range(n_latents), abs_corrs, alpha=0.7)
		plt.xlabel('Latent Neuron Index')
		plt.ylabel('Absolute Pearson Correlation with log10(flux)')
		plt.title('Latent Neuron vs. log10(flux) Correlation (Scatter)')
		if save_prefix is not None:
			plt.savefig(f"{save_prefix}_all_latents_scatter.png")
		plt.show()
	return results_sorted

def main(
    flux_path='AION/tests/test_data/FLUX_G_codec_input_batch.pt',
    sae_weights_path='best_llm_sae_rural-wood-16.pth',
    save_dir='probe_flux_in_sae_results',
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
    # Load flux data
    flux_g_data_pos = load_flux_data(flux_path, device)
    print(f"Flux data checksum: {torch.sum(flux_g_data_pos).item()}")
    
    # Get AION activations
    activations_path = os.path.join(save_dir, 'aion_activations_flux.npz')
    if os.path.exists(activations_path):
        data = np.load(activations_path)
        activations_np_flat, flux_np = data['activations'], data['flux']
        print(f"AION activations: shape={activations_np_flat.shape}")
        print(f"flux: shape={flux_np.shape}")
        print(f"Loaded activations from {activations_path}")
        print(f"Activations checksum: {np.sum(activations_np_flat)}")
        print(f"Flux checksum: {np.sum(flux_np)}")
    else:
        activations_np_flat, flux_np = get_aion_activations_from_flux(
            flux_g_data_pos, model, codec_manager, block_idx=block_idx, batch_size=batch_size, save_path=activations_path)
        print(f"Generated activations checksum: {np.sum(activations_np_flat)}")
        print(f"Generated flux checksum: {np.sum(flux_np)}")
    # Load SAE
    input_size = activations_np_flat.shape[1]
    print('input size', input_size)
    hidden_size = input_size*4
    k = max(1, int(hidden_size * 0.02))  # Use 2% sparsity by default
    sae_model = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    sae_model.load_state_dict(torch.load(sae_weights_path, weights_only=True, map_location=device))
    sae_model.eval()
    # Get SAE latents
    latents_path = os.path.join(save_dir, 'sae_latents.npy')
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
    # Probe correlation
    results = probe_latent_flux_correlation(latents_np, flux_np, plot=True, top_k=5, save_prefix=os.path.join(save_dir, 'probe'))

if __name__ == '__main__':
    main() 