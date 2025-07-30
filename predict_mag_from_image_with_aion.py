import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path:", sys.path)
import huggingface_hub
print("huggingface_hub version:", huggingface_hub.__version__)
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage, 
    # HSC Magnitudes
    HSCMagG, HSCMagR, HSCMagI, HSCMagZ, HSCMagY,
    # LegacySurvey Fluxes
    LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ,
    LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3, LegacySurveyFluxW4,
    # Other interesting modalities
    LegacySurveyShapeR, LegacySurveyShapeE1, LegacySurveyShapeE2, LegacySurveyEBV,
    GaiaFluxG, GaiaFluxBp, GaiaFluxRp, GaiaParallax, Z, Ra, Dec
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Predefined modality groups for easy selection
MODALITY_GROUPS = {
    'hsc_magnitudes': {
        'hsc_g': HSCMagG, 'hsc_r': HSCMagR, 'hsc_i': HSCMagI, 'hsc_z': HSCMagZ, 'hsc_y': HSCMagY
    },
    'legacysurvey_optical': {
        'ls_g': LegacySurveyFluxG, 'ls_r': LegacySurveyFluxR, 'ls_i': LegacySurveyFluxI, 'ls_z': LegacySurveyFluxZ
    },
    'legacysurvey_wise': {
        'w1': LegacySurveyFluxW1, 'w2': LegacySurveyFluxW2, 'w3': LegacySurveyFluxW3, 'w4': LegacySurveyFluxW4
    },
    'legacysurvey_shape': {
        'shape_r': LegacySurveyShapeR, 'shape_e1': LegacySurveyShapeE1, 'shape_e2': LegacySurveyShapeE2
    },
    'gaia_photometry': {
        'gaia_g': GaiaFluxG, 'gaia_bp': GaiaFluxBp, 'gaia_rp': GaiaFluxRp
    },
    'astrometry': {
        'parallax': GaiaParallax, 'redshift': Z, 'ra': Ra, 'dec': Dec, 'ebv': LegacySurveyEBV
    }
}

def predict_modalities_from_images(model, codec_manager, image_tensor, bands, 
                                 target_modalities, batch_size=32):
    """
    Predict multiple target modalities from input images efficiently.
    
    Args:
        model: AION model
        codec_manager: CodecManager instance
        image_tensor: Input images tensor [N, C, H, W]
        bands: List of band names for the input images
        target_modalities: Dict of {name: modality_class} to predict
        batch_size: Batch size for prediction
        
    Returns:
        Dict of {modality_name: predicted_scalar_values}
    """
    print(f"Predicting {len(target_modalities)} modalities...")
    all_predicted_logits = {name: [] for name in target_modalities.keys()}
    
    for i in range(0, len(image_tensor), batch_size):
        batch_img = image_tensor[i:i+batch_size]
        img_mod = LegacySurveyImage(flux=batch_img, bands=bands)
        tokens = codec_manager.encode(img_mod)
        
        # Predict for each modality
        for modality_name, modality_class in target_modalities.items():
            with torch.no_grad():
                preds = model(tokens, target_modality=modality_class)
                # Decode predicted values (returns a Scalar with .value shape [batch, 1, 1024])
                pred_values = codec_manager.decode(preds, modality_class)
                # Squeeze to get 2D array of logits
                all_predicted_logits[modality_name].append(pred_values.value.squeeze().cpu().numpy())
    
    # Convert logits to scalar values for each modality
    all_pred_scalars = {}
    for modality_name, modality_class in target_modalities.items():
        # Concatenate batches
        predicted_logits = np.concatenate(all_predicted_logits[modality_name])  # shape: (N, 1024)
        print(f"predicted_logits_{modality_name} shape:", predicted_logits.shape)
        print(f"Any NaN in predicted_logits_{modality_name}?", np.isnan(predicted_logits).any())
        
        # Convert to scalar using argmax and codebook
        pred_indices = predicted_logits.argmax(axis=1)  # shape: (N,)
        print(f"pred_indices_{modality_name}: max={pred_indices.max()}, min={pred_indices.min()}")
        
        codebook = codec_manager._load_codec(modality_class).quantizer.codebook.cpu().numpy()
        print(f"codebook_{modality_name} shape:", codebook.shape)
        
        # Ensure indices are within valid range
        max_valid_index = len(codebook) - 1
        if pred_indices.max() > max_valid_index:
            print(f"WARNING: Some {modality_name} predictions exceed codebook size! Clamping...")
            pred_indices = np.clip(pred_indices, 0, max_valid_index)
        
        pred_scalar = codebook[pred_indices]  # shape: (N,)
        
        # Report NaN values but don't replace them
        if np.isnan(pred_scalar).any():
            nan_count = np.isnan(pred_scalar).sum()
            total_count = len(pred_scalar)
            print(f"WARNING: Found {nan_count}/{total_count} NaN values in {modality_name} predictions")
            print(f"These will be excluded from correlation analysis")
        
        all_pred_scalars[modality_name] = pred_scalar
        print(f"pred_scalar_{modality_name} sample:", pred_scalar[:5])
        print(f"Any NaN in pred_scalar_{modality_name}?", np.isnan(pred_scalar).any())
    
    return all_pred_scalars

def calculate_center_pixel_features(image_tensor):
    """
    Calculate center pixel features from images.
    
    Args:
        image_tensor: Images tensor [N, C, H, W]
        
    Returns:
        Dict with different center pixel aggregations
    """
    all_bands_images = image_tensor.cpu().numpy()  # shape (N, C, 96, 96)
    
    # Average of the four center pixels [47,47], [47,48], [48,47], [48,48] per band
    center_pixels_coords = [(47, 47), (47, 48), (48, 47), (48, 48)]
    center_flux_per_band = np.zeros((all_bands_images.shape[0], all_bands_images.shape[1]))  # (N, C)
    
    for band_idx in range(all_bands_images.shape[1]):
        # Extract the 4 center pixels for this band and average them
        center_values = []
        for y, x in center_pixels_coords:
            center_values.append(all_bands_images[:, band_idx, y, x])
        center_flux_per_band[:, band_idx] = np.mean(center_values, axis=0)
    
    # Different aggregations across bands
    center_features = {
        'max_across_bands': np.max(center_flux_per_band, axis=1),  # Max flux across bands
        'mean_across_bands': np.mean(center_flux_per_band, axis=1),  # Mean flux across bands
        'sum_across_bands': np.sum(center_flux_per_band, axis=1),   # Sum flux across bands
    }
    
    # Convert to different units - create new dict to avoid modification during iteration
    NANOMAGGIE_TO_FLUX = 3.631e-29
    additional_features = {}
    for feature_name, values in center_features.items():
        flux_cgs = values * NANOMAGGIE_TO_FLUX
        additional_features[feature_name + '_cgs'] = flux_cgs
        additional_features[feature_name + '_log_cgs'] = np.log10(flux_cgs)
    
    # Combine original and additional features
    center_features.update(additional_features)
    
    return center_features

def analyze_correlations(center_features, predicted_scalars, output_dir):
    """
    Analyze correlations between center pixel features and predicted scalars.
    
    Args:
        center_features: Dict of center pixel features
        predicted_scalars: Dict of predicted scalar values
        output_dir: Directory to save results
        
    Returns:
        Dict of correlation results
    """
    correlation_results = {}
    
    # Test each combination of center feature vs predicted modality
    for feature_name, feature_values in center_features.items():
        correlation_results[feature_name] = {}
        
        for modality_name, pred_values in predicted_scalars.items():
            # Keep only finite values
            valid_mask = np.isfinite(pred_values) & np.isfinite(feature_values)
            
            if np.sum(valid_mask) > 1:
                feature_valid = feature_values[valid_mask]
                pred_valid = pred_values[valid_mask]
                
                corr = np.corrcoef(feature_valid, pred_valid)[0, 1]
                correlation_results[feature_name][modality_name] = corr
            else:
                correlation_results[feature_name][modality_name] = np.nan
    
    return correlation_results

def create_correlation_plots(center_features, predicted_scalars, correlation_results, 
                           feature_name, output_dir, max_plots=12):
    """
    Create scatter plots for correlations between a specific center feature and all modalities.
    """
    modalities = list(predicted_scalars.keys())
    n_modalities = len(modalities)
    
    if n_modalities == 0:
        return
    
    # Limit number of plots to avoid overcrowding
    if n_modalities > max_plots:
        print(f"Too many modalities ({n_modalities}), showing only first {max_plots}")
        modalities = modalities[:max_plots]
        n_modalities = max_plots
    
    # Calculate grid size
    n_cols = min(4, n_modalities)
    n_rows = (n_modalities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_modalities == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    feature_values = center_features[feature_name]
    
    for idx, modality_name in enumerate(modalities):
        pred_values = predicted_scalars[modality_name]
        
        # Keep only finite values
        valid_mask = np.isfinite(pred_values) & np.isfinite(feature_values)
        feature_valid = feature_values[valid_mask]
        pred_valid = pred_values[valid_mask]
        
        corr = correlation_results[feature_name][modality_name]
        
        # Create scatter plot
        ax = axes[idx]
        ax.scatter(feature_valid, pred_valid, alpha=0.6, s=20)
        ax.set_xlabel('Log Flux of Max Band of Avg Center Pixels [cgs]')
        ax.set_ylabel(f'Predicted {modality_name}')
        ax.set_title(f'{modality_name}: r = {corr:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_modalities, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filename = f'{feature_name}_vs_all_modalities.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()

def save_correlation_summary(correlation_results, output_dir):
    """Save detailed correlation results to file."""
    with open(os.path.join(output_dir, 'detailed_correlation_results.txt'), 'w') as f:
        f.write("Detailed Correlation Analysis Results\n")
        f.write("="*50 + "\n\n")
        
        for feature_name, modality_corrs in correlation_results.items():
            f.write(f"Feature: {feature_name}\n")
            f.write("-" * 30 + "\n")
            
            # Sort by absolute correlation
            sorted_corrs = sorted(modality_corrs.items(), 
                                key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, 
                                reverse=True)
            
            for modality_name, corr in sorted_corrs:
                f.write(f"  {modality_name:20s}: {corr:8.4f}\n")
            f.write("\n")

def convert_flux_predictions_to_cgs(predicted_scalars):
    """
    Convert LegacySurvey flux predictions from nanomaggies to CGS units.
    
    Args:
        predicted_scalars: Dict of predicted scalar values
        
    Returns:
        Dict with both original and CGS-converted values
    """
    NANOMAGGIE_TO_FLUX = 3.631e-29
    
    converted_scalars = {}
    
    for modality_name, pred_values in predicted_scalars.items():
        # Keep original values
        converted_scalars[modality_name] = pred_values
        
        # Convert LegacySurvey fluxes to CGS units
        if modality_name.startswith('ls_'):
            flux_cgs = pred_values * NANOMAGGIE_TO_FLUX
            flux_log_cgs = np.log10(flux_cgs)
            
            # Add CGS versions
            converted_scalars[f"{modality_name}_cgs"] = flux_cgs
            converted_scalars[f"{modality_name}_log_cgs"] = flux_log_cgs
            
            print(f"Converted {modality_name}:")
            print(f"  Original (nanomaggies): {pred_values[:5]}")
            print(f"  CGS flux: {flux_cgs[:5]}")
            print(f"  Log CGS flux: {flux_log_cgs[:5]}")
    
    return converted_scalars

def filter_relevant_modalities(predicted_scalars):
    """
    Filter to only keep the most relevant modalities for analysis.
    Keep HSC magnitudes and LegacySurvey log CGS fluxes.
    """
    relevant_modalities = {}
    
    for modality_name, pred_values in predicted_scalars.items():
        # Keep all HSC magnitudes
        if modality_name.startswith('hsc_'):
            relevant_modalities[modality_name] = pred_values
        # Keep only log CGS fluxes (not raw nanomaggies or linear CGS)
        elif modality_name.endswith('_log_cgs'):
            relevant_modalities[modality_name] = pred_values
    
    print(f"\nFiltered to {len(relevant_modalities)} relevant modalities:")
    for name in relevant_modalities.keys():
        print(f"  - {name}")
    
    return relevant_modalities

# 1. Model and CodecManager setup
model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
codec_manager = CodecManager(device=device)

# 2. Load test image data
img_data = torch.load('AION/tests/test_data/image_codec_input_batch.pt')
image_tensor = img_data['image']['array'][:, 4:]  # Use DES bands [4:8]
bands = ["DES-G", "DES-R", "DES-I", "DES-Z"]
image_tensor = image_tensor.float().to(device)

# Debug: Check the original image data and band selection
print(f"Original img_data shape: {img_data['image']['array'].shape}")
print(f"Selected image_tensor shape: {image_tensor.shape}")
print(f"Selected tensor min/max: {image_tensor.min():.6f} to {image_tensor.max():.6f}")

# Visualize the first image in the batch
plt.figure(figsize=(12, 3))
for i, band in enumerate(bands):
    plt.subplot(1, 4, i + 1)
    plt.imshow(image_tensor[0, i].cpu(), cmap="gray")
    plt.axis("off")
    plt.title(band)
plt.suptitle("First Image in Batch (DES-G, R, I, Z)")
plt.tight_layout()

# Create output directory
output_dir = 'pred_mag_from_image'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'first_image_in_batch.png'))

# =============================================================================
# CONFIGURE WHICH MODALITIES TO TEST
# =============================================================================

# Choose which modality groups to test (you can mix and match!)
modalities_to_test = {}

# Add HSC magnitudes
modalities_to_test.update(MODALITY_GROUPS['hsc_magnitudes'])

# Add LegacySurvey optical fluxes  
modalities_to_test.update(MODALITY_GROUPS['legacysurvey_optical'])

# Uncomment any of these to test additional modalities:
# modalities_to_test.update(MODALITY_GROUPS['legacysurvey_wise'])
# modalities_to_test.update(MODALITY_GROUPS['legacysurvey_shape'])
# modalities_to_test.update(MODALITY_GROUPS['gaia_photometry'])
# modalities_to_test.update(MODALITY_GROUPS['astrometry'])

# Or add individual modalities:
# modalities_to_test['custom_ebv'] = LegacySurveyEBV
# modalities_to_test['custom_parallax'] = GaiaParallax

print(f"\nTesting {len(modalities_to_test)} modalities:")
for name, modality_class in modalities_to_test.items():
    print(f"  - {name}: {modality_class.__name__}")

# =============================================================================
# RUN ANALYSIS
# =============================================================================

# 3. Predict all selected modalities
predicted_scalars = predict_modalities_from_images(
    model, codec_manager, image_tensor, bands, modalities_to_test, batch_size=32
)

# 4. Convert flux predictions to CGS units for fair comparison
print("\n" + "="*50)
print("CONVERTING FLUX PREDICTIONS TO CGS UNITS")
print("="*50)
predicted_scalars = convert_flux_predictions_to_cgs(predicted_scalars)

# 5. Filter to only relevant modalities
print("\n" + "="*50)
print("FILTERING TO RELEVANT MODALITIES")
print("="*50)
predicted_scalars = filter_relevant_modalities(predicted_scalars)

# 6. Calculate center pixel features
center_features = calculate_center_pixel_features(image_tensor)

# 7. Analyze correlations
correlation_results = analyze_correlations(center_features, predicted_scalars, output_dir)

# Debug: Show NaN counts for each modality
print("\n" + "="*50)
print("NaN ANALYSIS FOR EACH MODALITY")
print("="*50)
for modality_name, pred_values in predicted_scalars.items():
    nan_count = np.isnan(pred_values).sum()
    total_count = len(pred_values)
    print(f"{modality_name:20s}: {nan_count:3d}/{total_count:3d} NaN values ({nan_count/total_count*100:5.1f}%)")

print("\n" + "="*50)
print("VALID SAMPLE COUNTS FOR CORRELATION")
print("="*50)
main_feature = 'max_across_bands_log_cgs'
feature_values = center_features[main_feature]
for modality_name, pred_values in predicted_scalars.items():
    valid_mask = np.isfinite(pred_values) & np.isfinite(feature_values)
    valid_count = np.sum(valid_mask)
    total_count = len(pred_values)
    print(f"{modality_name:20s}: {valid_count:3d}/{total_count:3d} valid samples ({valid_count/total_count*100:5.1f}%)")

# 8. Create plots for the most interpretable center feature
main_feature = 'max_across_bands_log_cgs'  # Usually the most meaningful
create_correlation_plots(center_features, predicted_scalars, correlation_results, 
                        main_feature, output_dir)

# 9. Print summary of best correlations for each modality
print("\n" + "="*70)
print("CORRELATION SUMMARY (using max center-pixel log flux)")
print("="*70)
main_corrs = correlation_results[main_feature]
sorted_corrs = sorted(main_corrs.items(), 
                     key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, 
                     reverse=True)

for modality_name, corr in sorted_corrs:
    print(f"{modality_name:30s}: {corr:8.4f}")

# 10. Save detailed results
save_correlation_summary(correlation_results, output_dir)

print(f"\nDetailed results saved to {output_dir}/")
print(f"Main plot saved as: {main_feature}_vs_all_modalities.png")

