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
from aion.modalities import LegacySurveyFluxG, HSCMagG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Model and CodecManager setup
model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
codec_manager = CodecManager(device=device)

# 2. Load test flux data (nanomaggies)
flux_g_data = torch.load('AION/tests/test_data/FLUX_G_codec_input_batch.pt')
flux_g_data = flux_g_data.float().to(device)
batch_size = 32  # You can change this as needed

# DEBUG: Print shapes to verify if they match
print(f"\n=== DEBUGGING DATASET COMPATIBILITY ===")
print(f"flux_g_data shape: {flux_g_data.shape}")
print(f"flux_g_data samples: {len(flux_g_data)}")

# 3. Only use positive fluxes for valid magnitude prediction
positive_mask = flux_g_data > 0
flux_g_data_pos = flux_g_data[positive_mask]

# 4. Predict magnitude in batches
predicted_mags = []
for i in range(0, len(flux_g_data_pos), batch_size):
    batch_flux = flux_g_data_pos[i:i+batch_size]
    flux_mod = LegacySurveyFluxG(value=batch_flux)
    tokens = codec_manager.encode(flux_mod)
    # Target modality for predicting g-band magnitude
    with torch.no_grad():
        preds = model(tokens, target_modality=HSCMagG)
        # Decode predicted magnitude (returns a Scalar with .value shape [batch, 1])
        pred_mag = codec_manager.decode(preds, HSCMagG)
        # Squeeze to get 1D array of scalars
        predicted_mags.append(pred_mag.value.squeeze().cpu().numpy())

predicted_mags = np.concatenate(predicted_mags)  # shape: (N, 1024)

# Convert logits/distribution to scalar magnitude using argmax and codebook
pred_indices = predicted_mags.argmax(axis=1)  # shape: (N,)
codebook = codec_manager._load_codec(HSCMagG).quantizer.codebook.cpu().numpy()  # shape: (1024,)

# Fix NaN in codebook by removing the last entry if it is NaN
if np.isnan(codebook[-1]):
    print("Removing last codebook entry because it is NaN...")
    codebook = codebook[:-1]
    # If any pred_indices point to the removed index, clip them
    max_valid_index = len(codebook) - 1
    if pred_indices.max() > max_valid_index:
        print(f"Clipping pred_indices to new codebook size {len(codebook)}")
        pred_indices = np.clip(pred_indices, 0, max_valid_index)

pred_mag_scalar = codebook[pred_indices]  # shape: (N,)

# Ensure pred_indices are within valid range
max_valid_index = len(codebook) - 1
if pred_indices.max() > max_valid_index:
    print(f"WARNING: Some predictions exceed codebook size! Max index: {pred_indices.max()}, Codebook size: {len(codebook)}")
    # Clamp to valid range
    pred_indices = np.clip(pred_indices, 0, max_valid_index)
    pred_mag_scalar = codebook[pred_indices]  # Re-extract with clamped indices

# 5. Calculate expected AB magnitude from flux
NANOMAGGIE_TO_FLUX = 3.631e-29
flux_density = flux_g_data_pos.detach().cpu().numpy() * NANOMAGGIE_TO_FLUX
expected_mag = -2.5 * np.log10(flux_density) - 48.60
print("Any NaNs in expected_mag?", np.isnan(expected_mag).any())
print("Any Infs in expected_mag?", np.isinf(expected_mag).any())
print("expected_mag min/max:", np.nanmin(expected_mag), np.nanmax(expected_mag))

print("expected_mag shape:", expected_mag.shape)
print("predicted_mags shape:", predicted_mags.shape)

# Debug: Analyze token predictions
print(f"\n=== TOKEN PREDICTION ANALYSIS ===")
print(f"Token indices range: {pred_indices.min()} to {pred_indices.max()}")
print(f"Codebook range: {codebook.min():.2f} to {codebook.max():.2f}")

# Check if predictions are hitting the floor (low token indices)
low_token_threshold = 50  # Arbitrary threshold for "low" tokens
low_token_mask = pred_indices < low_token_threshold
low_token_count = np.sum(low_token_mask)
print(f"Predictions with low token indices (< {low_token_threshold}): {low_token_count} ({low_token_count/len(pred_indices)*100:.1f}%)")

if low_token_count > 0:
    print(f"  Low token predictions range: {pred_indices[low_token_mask].min()} to {pred_indices[low_token_mask].max()}")
    print(f"  Corresponding magnitudes: {codebook[pred_indices[low_token_mask]].min():.2f} to {codebook[pred_indices[low_token_mask]].max():.2f}")
    print(f"  Expected magnitudes for low tokens: {expected_mag[low_token_mask].min():.2f} to {expected_mag[low_token_mask].max():.2f}")

# Check if predictions are hitting the ceiling (high token indices)
high_token_threshold = 1024 - 50  # Arbitrary threshold for "high" tokens
high_token_mask = pred_indices > high_token_threshold
high_token_count = np.sum(high_token_mask)
print(f"Predictions with high token indices (> {high_token_threshold}): {high_token_count} ({high_token_count/len(pred_indices)*100:.1f}%)")

# Show token distribution
print(f"\nToken index distribution:")
print(f"  Min: {pred_indices.min()}")
print(f"  25th percentile: {np.percentile(pred_indices, 25):.0f}")
print(f"  Median: {np.median(pred_indices):.0f}")
print(f"  75th percentile: {np.percentile(pred_indices, 75):.0f}")
print(f"  Max: {pred_indices.max()}")

# 6. Compare and plot
plt.figure(figsize=(10, 6))
plt.scatter(expected_mag, pred_mag_scalar, alpha=0.7, label='AION Predicted')
plt.plot([expected_mag.min(), expected_mag.max()], [expected_mag.min(), expected_mag.max()], 'k--', label='1:1 Line')
plt.xlabel('Expected AB Magnitude (from flux)')
plt.ylabel('AION Predicted AB Magnitude')
plt.title('AION Predicted vs. Expected g-band Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aion_predicted_vs_expected_mag.png', dpi=150)
print("\n📊 Plot saved as 'aion_predicted_vs_expected_mag.png'")

diff = pred_mag_scalar - expected_mag
print(f"\n=== STATISTICS ===")
print(f"Mean difference (AION - expected): {np.mean(diff):.6f}")
print(f"Std deviation: {np.std(diff):.6f}")
print(f"Min difference: {np.min(diff):.6f}")
print(f"Max difference: {np.max(diff):.6f}")

# Correlation coefficient
correlation = np.corrcoef(expected_mag, pred_mag_scalar)[0, 1]
print(f"Pearson correlation coefficient: {correlation:.6f}")

print(f"\n=== DATA RANGES ===")
print(f"Expected magnitude range: {expected_mag.min():.2f} to {expected_mag.max():.2f}")
print(f"Predicted magnitude range: {pred_mag_scalar.min():.2f} to {pred_mag_scalar.max():.2f}")
print(f"Number of samples: {len(expected_mag)}")

# Analyze the "floor" effect
floor_threshold = 17.0  # Based on the plot showing floor around 16
floor_mask = pred_mag_scalar < floor_threshold
floor_count = np.sum(floor_mask)
floor_percentage = (floor_count / len(pred_mag_scalar)) * 100

print(f"\n=== FLOOR EFFECT ANALYSIS ===")
print(f"Predictions below {floor_threshold} (floor effect): {floor_count} samples ({floor_percentage:.1f}%)")
if floor_count > 0:
    floor_expected_mean = np.mean(expected_mag[floor_mask])
    floor_expected_std = np.std(expected_mag[floor_mask])
    print(f"  Expected magnitudes for floor predictions: {floor_expected_mean:.2f} ± {floor_expected_std:.2f}")
    print(f"  Floor predictions mean: {np.mean(pred_mag_scalar[floor_mask]):.2f}")

# Performance excluding floor effect
non_floor_mask = ~floor_mask
if np.sum(non_floor_mask) > 0:
    print(f"\n=== PERFORMANCE EXCLUDING FLOOR EFFECT ===")
    non_floor_expected = expected_mag[non_floor_mask]
    non_floor_predicted = pred_mag_scalar[non_floor_mask]
    non_floor_diff = non_floor_predicted - non_floor_expected
    
    # Other metrics for non-floor data
    correlation_non_floor = np.corrcoef(non_floor_expected, non_floor_predicted)[0, 1]
    
    print(f"Correlation (excluding floor): {correlation_non_floor:.6f}")
    print(f"Non-floor samples: {np.sum(non_floor_mask)}")

if floor_percentage > 20:
    print(f"⚠️  High floor effect detected ({floor_percentage:.1f}% of predictions)")
    print("   This suggests the model may be hitting quantization limits for faint objects")

# Additional analysis for quantization-limited scenarios
print(f"\n=== QUANTIZATION ANALYSIS ===")

# Check for multiple quantization levels
unique_predicted = np.unique(pred_mag_scalar)
print(f"Number of unique predicted magnitudes: {len(unique_predicted)}")
print(f"Unique predicted magnitudes: {sorted(unique_predicted)[:10]}...")  # Show first 10

# Analyze correlation by magnitude ranges
print(f"\n=== CORRELATION BY MAGNITUDE RANGE ===")
magnitude_ranges = [
    (18, 20, "Bright (18-20)"),
    (20, 22, "Medium (20-22)"), 
    (22, 24, "Faint (22-24)"),
    (24, 28, "Very Faint (24-28)")
]

for min_mag, max_mag, label in magnitude_ranges:
    mask = (expected_mag >= min_mag) & (expected_mag < max_mag)
    if np.sum(mask) > 10:  # Only analyze ranges with enough data
        range_expected = expected_mag[mask]
        range_predicted = pred_mag_scalar[mask]
        range_corr = np.corrcoef(range_expected, range_predicted)[0, 1]
        range_mae = np.mean(np.abs(range_predicted - range_expected))
        print(f"{label}: Correlation = {range_corr:.3f}, MAE = {range_mae:.3f}, N = {np.sum(mask)}")

# Debug: Print codebook and reservoir information
print("\n=== DEBUGGING CODEBOOK ===")
codec = codec_manager._load_codec(HSCMagG)
quantizer = codec.quantizer

print(f"Quantizer type: {type(quantizer)}")
print(f"Codebook size: {quantizer.codebook_size}")
print(f"Reservoir size: {quantizer._reservoir_size}")
print(f"Total samples seen: {quantizer._n_total_samples}")

# Print reservoir values
reservoir_np = quantizer._reservoir.cpu().numpy()
print(f"\nReservoir shape: {reservoir_np.shape}")
print(f"Reservoir min: {reservoir_np.min()}")
print(f"Reservoir max: {reservoir_np.max()}")
print(f"Reservoir mean: {reservoir_np.mean()}")
print(f"Reservoir std: {reservoir_np.std()}")
print(f"Number of unique values in reservoir: {np.unique(reservoir_np).size}")
print(f"Number of NaN values in reservoir: {np.isnan(reservoir_np).sum()}")
print(f"Number of Inf values in reservoir: {np.isinf(reservoir_np).sum()}")

# Print first 20 reservoir values
print(f"\nFirst 20 reservoir values:")
for i in range(min(20, len(reservoir_np))):
    print(f"  [{i:3d}]: {reservoir_np[i]:.6f}")

# Print codebook values
codebook_np = quantizer.codebook.cpu().numpy()
print(f"\nCodebook shape: {codebook_np.shape}")
print(f"Codebook min: {codebook_np.min()}")
print(f"Codebook max: {codebook_np.max()}")
print(f"Codebook mean: {codebook_np.mean()}")
print(f"Codebook std: {codebook_np.std()}")
print(f"Number of unique values in codebook: {np.unique(codebook_np).size}")
print(f"Number of NaN values in codebook: {np.isnan(codebook_np).sum()}")
print(f"Number of Inf values in codebook: {np.isinf(codebook_np).sum()}")

# Print first 20 codebook values
print(f"\nFirst 20 codebook values:")
for i in range(min(20, len(codebook_np))):
    print(f"  [{i:3d}]: {codebook_np[i]:.6f}")

# Print last 20 codebook values
print(f"\nLast 20 codebook values:")
for i in range(max(0, len(codebook_np)-20), len(codebook_np)):
    print(f"  [{i:3d}]: {codebook_np[i]:.6f}")

# Check if there are any NaN ranges in the codebook
nan_indices = np.where(np.isnan(codebook_np))[0]
if len(nan_indices) > 0:
    print(f"\nNaN values found at indices: {nan_indices[:10]}...")  # Show first 10
    print(f"Total NaN indices: {len(nan_indices)}")

    # Show the range around some NaN values
    for i in range(min(3, len(nan_indices))):
        idx = nan_indices[i]
        start = max(0, idx-5)
        end = min(len(codebook_np), idx+6)
        print(f"\nRange around NaN at index {idx}:")
        for j in range(start, end):
            marker = " <-- NaN" if j == idx else ""
            print(f"  [{j:3d}]: {codebook_np[j]:.6f}{marker}")

print("\n" + "="*50) 