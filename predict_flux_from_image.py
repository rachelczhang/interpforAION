import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
)

def load_legacysurvey_data(filepath, max_samples=None):
    """
    Load LegacySurvey data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        dict with images, true fluxes, and metadata
    """
    print(f"Loading data from {filepath}...")
    
    with h5py.File(filepath, 'r') as f:
        # Load image data
        image_array = f['image_array'][:max_samples]  # Shape: (N, C, H, W)
        image_band = f['image_band'][:max_samples]    # Band information
        
        # Load true flux values
        true_flux_g = f['FLUX_G'][:max_samples]
        true_flux_r = f['FLUX_R'][:max_samples]
        true_flux_i = f['FLUX_I'][:max_samples]
        true_flux_z = f['FLUX_Z'][:max_samples]
        
        # Load additional metadata
        object_id = f['object_id'][:max_samples]
        ra = f['ra'][:max_samples]
        dec = f['dec'][:max_samples]
        
        print(f"Loaded {len(image_array)} samples")
        print(f"Image shape: {image_array.shape}")
        
        # Convert bands to list of strings if it's a numpy array
        if isinstance(image_band, np.ndarray):
            if image_band.ndim == 1:
                # If it's a 1D array, take the first element as the band list
                bands_raw = image_band[0] if len(image_band) > 0 else image_band
            else:
                # If it's a 2D array, take the first row
                bands_raw = image_band[0] if len(image_band) > 0 else []
        else:
            bands_raw = image_band
        
        # Convert byte strings to regular strings and format for AION
        if isinstance(bands_raw, (list, np.ndarray)):
            bands_list = []
            for band in bands_raw:
                if isinstance(band, bytes):
                    band_str = band.decode('utf-8')
                else:
                    band_str = str(band)
                # Convert to AION expected format (DES-G, DES-R, DES-I, DES-Z)
                if '-' in band_str:
                    survey, band_letter = band_str.split('-', 1)
                    # Convert to uppercase DES format
                    formatted_band = f"DES-{band_letter.upper()}"
                else:
                    # If no survey prefix, assume it's just the band letter
                    formatted_band = f"DES-{band_str.upper()}"
                bands_list.append(formatted_band)
        else:
            # Single band case
            if isinstance(bands_raw, bytes):
                band_str = bands_raw.decode('utf-8')
            else:
                band_str = str(bands_raw)
            # Convert to AION expected format
            if '-' in band_str:
                survey, band_letter = band_str.split('-', 1)
                formatted_band = f"DES-{band_letter.upper()}"
            else:
                formatted_band = f"DES-{band_str.upper()}"
            bands_list = [formatted_band]
        
        if max_samples:
            print(f"Limited to {max_samples} samples for testing")
    
    return {
        'images': image_array,
        'true_flux_g': true_flux_g,
        'true_flux_r': true_flux_r,
        'true_flux_i': true_flux_i,
        'true_flux_z': true_flux_z,
        'object_id': object_id,
        'ra': ra,
        'dec': dec,
        'bands': bands_list
    }

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
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
        batch_img = images[i:i+batch_size]
        img_mod = LegacySurveyImage(flux=batch_img, bands=bands)
        print(f"Batch {i//batch_size}: Image modality shape after LegacySurveyImage: {img_mod.flux.shape}")
        
        tokens = codec_manager.encode(img_mod)
        # tokens is a dictionary, so we need to access the tensor within it
        # The key should be the token_key of the LegacySurveyImage modality
        token_key = img_mod.token_key
        if token_key in tokens:
            token_tensor = tokens[token_key]
            print(f"Batch {i//batch_size}: Token tensor shape: {token_tensor.shape}")
        else:
            print(f"Warning: token_key '{token_key}' not found in tokens dictionary")
            print(f"Available keys: {list(tokens.keys())}")
            continue
        
        # Predict for each modality
        for modality_name, modality_class in target_modalities.items():
            with torch.no_grad():
                preds = model(tokens, target_modality=modality_class)
                print(f"Batch {i//batch_size}, {modality_name}: Raw model predictions keys: {list(preds.keys())}")
                
                # Get the raw logits from model predictions
                raw_logits = preds[modality_class.token_key]  # [batch, num_tokens, vocab_size]
                print(f"Batch {i//batch_size}, {modality_name}: Raw logits shape: {raw_logits.shape}")
                
                # Convert logits to discrete token indices
                token_indices = raw_logits.argmax(dim=-1)  # [batch, num_tokens]
                print(f"Batch {i//batch_size}, {modality_name}: Token indices shape: {token_indices.shape}")
                print(f"Batch {i//batch_size}, {modality_name}: Sample token indices: {token_indices.flatten()[:5]}")
                
                # Create tokens dict for decode
                tokens_for_decode = {modality_class.token_key: token_indices}
                
                # Decode the discrete tokens to get actual flux values
                pred_values = codec_manager.decode(tokens_for_decode, modality_class)
                print(f"Batch {i//batch_size}, {modality_name}: Decoded type: {type(pred_values)}")
                print(f"Batch {i//batch_size}, {modality_name}: Decoded value shape: {pred_values.value.shape}")
                
                # Extract flux values
                flux_values = pred_values.value.squeeze().cpu().numpy()
                print(f"Batch {i//batch_size}, {modality_name}: Final flux shape: {flux_values.shape}")
                print(f"Batch {i//batch_size}, {modality_name}: Sample flux values: {flux_values.flatten()[:5]}")
                
                all_predicted_fluxes[modality_name].append(flux_values)
    
    # Concatenate all batches
    predicted_fluxes = {}
    for modality_name in target_modalities.keys():
        print(f"\n=== Processing {modality_name} ===")
        print(f"Number of batches collected: {len(all_predicted_fluxes[modality_name])}")
        for j, batch_data in enumerate(all_predicted_fluxes[modality_name]):
            print(f"Batch {j} shape: {batch_data.shape}")
        
        # Concatenate all flux values
        predicted_flux = np.concatenate(all_predicted_fluxes[modality_name])
        print(f"Final predicted_flux_{modality_name} shape: {predicted_flux.shape}")
        predicted_fluxes[modality_name] = predicted_flux
        
        print(f"{modality_name}: predicted {len(predicted_flux)} values")
        print(f"  Range: {predicted_flux.min():.6f} to {predicted_flux.max():.6f}")
    
    return predicted_fluxes

def evaluate_predictions(predicted_fluxes, true_fluxes):
    """
    Evaluate flux predictions against true values.
    
    Args:
        predicted_fluxes: Dict of predicted flux values
        true_fluxes: Dict of true flux values
        
    Returns:
        Dict with evaluation metrics
    """
    print("\nEvaluating predictions...")
    
    results = {}
    for band in ['g', 'r', 'i', 'z']:
        pred_key = f'flux_{band}'
        true_key = f'true_flux_{band}'
        
        pred_values = predicted_fluxes[pred_key]
        true_values = true_fluxes[true_key]
        
        print(f"\n{band.upper()}-band raw statistics:")
        print(f"  Total samples: {len(pred_values)}")
        print(f"  Predicted range: {pred_values.min():.6f} to {pred_values.max():.6f}")
        print(f"  True range: {true_values.min():.6f} to {true_values.max():.6f}")
        print(f"  Predicted finite: {np.sum(np.isfinite(pred_values))}/{len(pred_values)}")
        print(f"  True finite: {np.sum(np.isfinite(true_values))}/{len(true_values)}")
        print(f"  Predicted positive: {np.sum(pred_values > 0)}/{len(pred_values)}")
        print(f"  True positive: {np.sum(true_values > 0)}/{len(true_values)}")
        
        # Use all samples directly without any masking
        pred_valid = pred_values
        true_valid = true_values
        
        # Calculate metrics
        mse = np.mean((pred_valid - true_valid) ** 2)
        mae = np.mean(np.abs(pred_valid - true_valid))
        corr = np.corrcoef(pred_valid, true_valid)[0, 1]
        
        # Calculate relative errors
        rel_error = np.abs(pred_valid - true_valid) / true_valid
        mean_rel_error = np.mean(rel_error)
        median_rel_error = np.median(rel_error)
        
        results[band] = {
            'mse': mse,
            'mae': mae,
            'correlation': corr,
            'mean_relative_error': mean_rel_error,
            'median_relative_error': median_rel_error,
            'valid_samples': len(pred_values),
            'total_samples': len(pred_values)
        }
        
        print(f"\n{band.upper()}-band results:")
        print(f"  Total samples: {len(pred_values)}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Mean relative error: {mean_rel_error:.4f}")
        print(f"  Median relative error: {median_rel_error:.4f}")
    
    return results

def create_comparison_plots(predicted_fluxes, true_fluxes, output_dir):
    """
    Create scatter plots comparing predicted vs true flux values.
    """
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(['g', 'r', 'i', 'z']):
        pred_key = f'flux_{band}'
        true_key = f'true_flux_{band}'
        
        pred_values = predicted_fluxes[pred_key]
        true_values = true_fluxes[true_key]
        
        # Use all samples directly without any masking
        pred_valid = pred_values
        true_valid = true_values
        
        ax = axes[idx]
        ax.scatter(true_valid, pred_valid, alpha=0.6, s=20)
        
        # Add 1:1 line
        min_val = min(true_valid.min(), pred_valid.min())
        max_val = max(true_valid.max(), pred_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate correlation
        corr = np.corrcoef(pred_valid, true_valid)[0, 1]
        
        ax.set_xlabel(f'True {band.upper()}-band Flux')
        ax.set_ylabel(f'Predicted {band.upper()}-band Flux')
        ax.set_title(f'{band.upper()}-band: r = {corr:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_true_flux.png'), dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the flux prediction evaluation."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and codec manager
    print("Loading AION model...")
    model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
    codec_manager = CodecManager(device=device)
    print('model loaded')
    
    # Create output directory
    output_dir = 'flux_prediction_results'
    
    # Load data
    data_path = '/mnt/home/polymathic/ceph/MultimodalUniverse/legacysurvey/dr10_south_21/healpix=299/001-of-001.hdf5'
    data = load_legacysurvey_data(data_path, max_samples=200)  # Limit for testing
    print('data loaded')
    
    # Convert images to tensor and move to device
    images = torch.tensor(data['images'], dtype=torch.float32).to(device)
    print(f"Image tensor shape: {images.shape}")
    print(f"Image value range: {images.min():.6f} to {images.max():.6f}")
    
    # Predict flux values
    predicted_fluxes = predict_flux_from_images(model, codec_manager, images, data['bands'])
    
    # Prepare true flux values
    true_fluxes = {
        'true_flux_g': data['true_flux_g'],
        'true_flux_r': data['true_flux_r'],
        'true_flux_i': data['true_flux_i'],
        'true_flux_z': data['true_flux_z'],
    }
    
    # Print true flux ranges
    print("\nTrue flux ranges (nanomaggies):")
    for band in ['g', 'r', 'i', 'z']:
        true_key = f'true_flux_{band}'
        true_values = true_fluxes[true_key]
        print(f"  {band.upper()}-band: {true_values.min():.6f} to {true_values.max():.6f}")
        print(f"    Finite values: {np.sum(np.isfinite(true_values))}/{len(true_values)}")
        print(f"    Positive values: {np.sum(true_values > 0)}/{len(true_values)}")
    
    # Evaluate predictions
    results = evaluate_predictions(predicted_fluxes, true_fluxes)
    
    # Create comparison plots
    create_comparison_plots(predicted_fluxes, true_fluxes, output_dir)
    

if __name__ == "__main__":
    main()