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
from aion.modalities import LegacySurveyImage, HSCMagG

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Model and CodecManager setup
model = AION.from_pretrained('polymathic-ai/aion-base').to(device).eval()
codec_manager = CodecManager(device=device)

# 2. Load test image data
img_data = torch.load('AION/tests/test_data/image_codec_input_batch.pt')
# img_data['image']['array'] shape: [batch, 8, 96, 96] (first 4 bands are WISE, last 4 are DES-G,R,I,Z)
# Use only DES-G, DES-R, DES-I, DES-Z bands (channels 4:8 or 5: as in test code)
image_tensor = img_data['image']['array'][:, 4:]  # shape: [batch, 4, 96, 96]
bands = ["DES-G", "DES-R", "DES-I", "DES-Z"]
image_tensor = image_tensor.float().to(device)

# Debug: Check the original image data and band selection
print(f"Original img_data shape: {img_data['image']['array'].shape}")
print(f"Selected image_tensor shape: {image_tensor.shape}")
print(f"Original data min/max: {img_data['image']['array'].min():.6f} to {img_data['image']['array'].max():.6f}")
print(f"Selected tensor min/max: {image_tensor.min():.6f} to {image_tensor.max():.6f}")

# Check each band separately
for i, band in enumerate(bands):
    band_data = image_tensor[:, i, :, :].cpu()
    print(f"{band} (index {i}): min={band_data.min():.6f}, max={band_data.max():.6f}, mean={band_data.mean():.6f}")

# Visualize the first image in the batch (DES-G, R, I, Z)
plt.figure(figsize=(12, 3))
for i, band in enumerate(bands):
    plt.subplot(1, 4, i + 1)
    plt.imshow(image_tensor[0, i].cpu(), cmap="gray")
    plt.axis("off")
    plt.title(band)
plt.suptitle("First Image in Batch (DES-G, R, I, Z)")
plt.tight_layout()

# Create output directory for PNGs
output_dir = 'pred_mag_from_image'
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'first_image_in_batch.png'))

batch_size = 32 

# 4. Predict magnitude in batches
predicted_mags = []
for i in range(0, len(image_tensor), batch_size):
    batch_img = image_tensor[i:i+batch_size]
    img_mod = LegacySurveyImage(flux=batch_img, bands=bands)
    tokens = codec_manager.encode(img_mod)
    # Target modality for predicting g-band magnitude
    with torch.no_grad():
        preds = model(tokens, target_modality=HSCMagG)
        # Decode predicted magnitude (returns a Scalar with .value shape [batch, 1, 1024])
        pred_mag = codec_manager.decode(preds, HSCMagG)
        # Squeeze to get 2D array of logits
        predicted_mags.append(pred_mag.value.squeeze().cpu().numpy())

predicted_mags = np.concatenate(predicted_mags)  # shape: (N, 1024)
print("predicted_mags shape:", predicted_mags.shape)
print("predicted_mags sample:", predicted_mags[:5])
print("Any NaN in predicted_mags?", np.isnan(predicted_mags).any())

# Convert logits/distribution to scalar magnitude using argmax and codebook
pred_indices = predicted_mags.argmax(axis=1)  # shape: (N,)
print("pred_indices shape:", pred_indices.shape)
print("pred_indices sample:", pred_indices[:10])
print("Any NaN in pred_indices?", np.isnan(pred_indices).any())
print("Max pred_indices:", pred_indices.max(), "Min pred_indices:", pred_indices.min())
codebook = codec_manager._load_codec(HSCMagG).quantizer.codebook.cpu().numpy()  # shape: (1024,)
print("codebook shape:", codebook.shape)
print("codebook sample:", codebook[:10])
print("Any NaN in codebook?", np.isnan(codebook).any())

pred_mag_scalar = codebook[pred_indices]  # shape: (N,)
print("pred_mag_scalar sample after codebook mapping:", pred_mag_scalar[:10])
print("Any NaN in pred_mag_scalar after mapping?", np.isnan(pred_mag_scalar).any())

# Ensure pred_indices are within valid range
max_valid_index = len(codebook) - 1
if pred_indices.max() > max_valid_index:
    print(f"WARNING: Some predictions exceed codebook size! Max index: {pred_indices.max()}, Codebook size: {len(codebook)}")
    # Clamp to valid range
    pred_indices = np.clip(pred_indices, 0, max_valid_index)
    pred_mag_scalar = codebook[pred_indices]  # Re-extract with clamped indices

all_bands_images = image_tensor.cpu().numpy()  # shape (N, 4, 96, 96)

# Average of the four center pixels [47,47], [47,48], [48,47], [48,48] per band
center_pixels_coords = [(47, 47), (47, 48), (48, 47), (48, 48)]
center_flux_per_band = np.zeros((all_bands_images.shape[0], 4))  # (N, 4) for 4 bands

for band_idx in range(4):
    # Extract the 4 center pixels for this band and average them
    center_values = []
    for y, x in center_pixels_coords:
        center_values.append(all_bands_images[:, band_idx, y, x])
    center_flux_per_band[:, band_idx] = np.mean(center_values, axis=0)

# Take the max across the 4 bands for each sample
center_pixel_flux_mean = np.max(center_flux_per_band, axis=1)  # (N,)

# Convert nanomaggies → erg s⁻¹ cm⁻² Hz⁻¹ (cgs flux–density units)
NANOMAGGIE_TO_FLUX = 3.631e-29
center_pixel_flux_cgs = center_pixel_flux_mean * NANOMAGGIE_TO_FLUX
center_pixel_logflux_cgs = np.log10(center_pixel_flux_cgs)

# Keep only samples whose predicted magnitudes are finite
valid_mask = np.isfinite(pred_mag_scalar)
center_pixel_flux_cgs = center_pixel_flux_cgs[valid_mask]
center_pixel_logflux_cgs = center_pixel_logflux_cgs[valid_mask]
pred_mag_valid = pred_mag_scalar[valid_mask]

# Scatter plot
plt.figure(figsize=(8, 6))
corr = np.corrcoef(center_pixel_logflux_cgs, pred_mag_valid)[0, 1]
plt.scatter(center_pixel_logflux_cgs, pred_mag_valid, alpha=0.7, s=30)
plt.xlabel('Center-pixels max log flux density across 4 bands')
plt.ylabel('Predicted g-band magnitude')
plt.title(f'Center-pixels logflux vs. predicted magnitude (Pearson r = {corr:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure (overwrite the old filename to keep downstream usage unchanged)
plt.savefig(os.path.join(output_dir, 'center_pixels_logflux_vs_magnitude.png'), dpi=150)
plt.show()

print(f'Pearson correlation (centre-pixel logflux vs. magnitude): {corr:.3f}')

