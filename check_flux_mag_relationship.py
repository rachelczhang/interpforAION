import torch
import numpy as np
import matplotlib.pyplot as plt

def load_test_data():
    """Load the test data files for flux and magnitude."""
    try:
        # Load flux data - fixed path to AION directory
        flux_g_data = torch.load('AION/tests/test_data/FLUX_G_codec_input_batch.pt')
        print(f"FLUX_G data loaded:")
        print(f"  Shape: {flux_g_data.shape}")
        print(f"  Type: {type(flux_g_data)}")
        print(f"  Units: nanomaggies (Legacy Survey g-band flux)")
        print(f"  Min: {flux_g_data.min():.6f} nanomaggies")
        print(f"  Max: {flux_g_data.max():.6f} nanomaggies")
        print(f"  Mean: {flux_g_data.mean():.6f} nanomaggies")
        print(f"  Sample values: {flux_g_data[:5]} nanomaggies")
        print()
        
        # Load magnitude data - fixed path to AION directory
        mag_g_data = torch.load('AION/tests/test_data/g_cmodel_mag_codec_input_batch.pt')
        print(f"MAG_G data loaded:")
        print(f"  Shape: {mag_g_data.shape}")
        print(f"  Type: {type(mag_g_data)}")
        print(f"  Units: AB magnitudes (HSC g-band cmodel magnitude)")
        print(f"  Min: {mag_g_data.min():.6f} mag")
        print(f"  Max: {mag_g_data.max():.6f} mag")
        print(f"  Mean: {mag_g_data.mean():.6f} mag")
        print(f"  Sample values: {mag_g_data[:5]} mag")
        print()
        
        return flux_g_data, mag_g_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def check_flux_magnitude_relationship(flux_data, mag_data):
    """
    Check if flux and magnitude follow the expected relationship:
    magnitude = -2.5 * log10(flux) + constant (AB system)
    """
    print("=== FLUX-MAGNITUDE RELATIONSHIP ANALYSIS ===")
    
    # Convert to numpy for easier analysis
    flux_np = flux_data.numpy()
    mag_np = mag_data.numpy()
    
    # Check if we have the same number of samples
    if len(flux_np) != len(mag_np):
        print(f"WARNING: Different number of samples!")
        print(f"  Flux samples: {len(flux_np)}")
        print(f"  Magnitude samples: {len(mag_np)}")
        print("  This suggests they are from different datasets/surveys.")
        print("  We'll analyze each dataset separately to understand their properties.")
        
        # Analyze flux data properties
        print(f"\n=== FLUX DATA ANALYSIS ===")
        print(f"Flux range: {flux_np.min():.6f} to {flux_np.max():.6f} nanomaggies")
        print(f"Flux mean: {flux_np.mean():.6f} nanomaggies")
        print(f"Flux std: {flux_np.std():.6f} nanomaggies")
        
        # Calculate expected magnitude from flux using AB system
        NANOMAGGIE_TO_FLUX = 3.631e-29
        
        # Filter out negative fluxes (they're likely noise/artifacts)
        positive_flux_mask = flux_np > 0
        positive_flux = flux_np[positive_flux_mask]
        
        print(f"Positive flux samples: {np.sum(positive_flux_mask)} out of {len(flux_np)}")
        print(f"Negative flux samples: {len(flux_np) - np.sum(positive_flux_mask)} (likely noise/artifacts)")
        
        if len(positive_flux) > 0:
            flux_density = positive_flux * NANOMAGGIE_TO_FLUX
            expected_mag_from_flux = -2.5 * np.log10(flux_density) - 48.60
            
            print(f"Expected magnitude range: {expected_mag_from_flux.min():.2f} to {expected_mag_from_flux.max():.2f} AB mag")
            print(f"Expected magnitude mean: {expected_mag_from_flux.mean():.2f} AB mag")
        else:
            print("No positive flux values found!")
            expected_mag_from_flux = np.array([])
        
        # Analyze magnitude data properties
        print(f"\n=== MAGNITUDE DATA ANALYSIS ===")
        print(f"Magnitude range: {mag_np.min():.2f} to {mag_np.max():.2f} AB mag")
        print(f"Magnitude mean: {mag_np.mean():.2f} AB mag")
        print(f"Magnitude std: {mag_np.std():.2f} AB mag")
        
        # Check if the ranges overlap (only if we have valid expected magnitudes)
        if len(expected_mag_from_flux) > 0:
            flux_mag_range = (expected_mag_from_flux.min(), expected_mag_from_flux.max())
            obs_mag_range = (mag_np.min(), mag_np.max())
            
            print(f"\n=== RANGE COMPARISON ===")
            print(f"Flux → Expected magnitude range: {flux_mag_range[0]:.2f} to {flux_mag_range[1]:.2f} AB mag")
            print(f"Observed magnitude range: {obs_mag_range[0]:.2f} to {obs_mag_range[1]:.2f} AB mag")
            
            if (flux_mag_range[0] <= obs_mag_range[1] and flux_mag_range[1] >= obs_mag_range[0]):
                print("✅ Ranges overlap - they could represent similar astronomical objects")
            else:
                print("❌ Ranges don't overlap - they likely represent different types of objects")
        else:
            print("\n=== RANGE COMPARISON ===")
            print("Cannot compare ranges due to invalid flux data")
        
        return flux_np, mag_np, expected_mag_from_flux, None
    
    print(f"Both datasets have {len(flux_np)} samples")
    
    # Calculate expected magnitude from flux using AB system
    # In AB system: m_AB = -2.5 * log10(f_ν) - 48.60
    # where f_ν is in erg/s/cm²/Hz
    # For nanomaggies: 1 nanomaggie = 3.631 × 10⁻⁶ Jy = 3.631 × 10⁻²⁹ erg/s/cm²/Hz
    
    # Convert nanomaggies to flux density in erg/s/cm²/Hz
    # 1 nanomaggie = 3.631e-29 erg/s/cm²/Hz
    NANOMAGGIE_TO_FLUX = 3.631e-29
    
    # Calculate expected magnitude from flux
    flux_density = flux_np * NANOMAGGIE_TO_FLUX
    expected_mag = -2.5 * np.log10(flux_density) - 48.60
    
    print("\n=== COMPARISON ===")
    print("Sample | Flux (nanomaggies) | Observed Mag | Expected Mag | Difference")
    print("-------|-------------------|--------------|--------------|------------")
    
    for i in range(min(10, len(flux_np))):
        flux_val = flux_np[i]
        obs_mag = mag_np[i]
        exp_mag = expected_mag[i]
        diff = obs_mag - exp_mag
        print(f"{i:6d} | {flux_val:17.6f} | {obs_mag:12.6f} | {exp_mag:12.6f} | {diff:10.6f}")
    
    # Calculate statistics
    differences = mag_np - expected_mag
    print(f"\n=== STATISTICS ===")
    print(f"Mean difference (obs - exp): {np.mean(differences):.6f}")
    print(f"Std deviation: {np.std(differences):.6f}")
    print(f"Min difference: {np.min(differences):.6f}")
    print(f"Max difference: {np.max(differences):.6f}")
    
    # Check if differences are consistent (should be roughly constant)
    print(f"\n=== CONSISTENCY CHECK ===")
    if np.std(differences) < 0.1:  # Arbitrary threshold
        print("✅ Differences are consistent - likely same objects with different zero points")
    else:
        print("❌ Differences are not consistent - may be different objects or different calibrations")
    
    return flux_np, mag_np, expected_mag, differences

def plot_relationship(flux_data, mag_data, expected_mag):
    """Create a plot showing the flux-magnitude relationship."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Filter out negative fluxes for plotting
        positive_flux_mask = flux_data > 0
        positive_flux = flux_data[positive_flux_mask]
        
        if len(positive_flux) > 0 and len(expected_mag) > 0:
            # Plot 1: Flux vs Magnitude (only positive fluxes)
            plt.subplot(2, 2, 1)
            plt.scatter(positive_flux, expected_mag, alpha=0.6, label='Expected from Flux')
            plt.xlabel('Flux (nanomaggies)')
            plt.ylabel('Magnitude (AB)')
            plt.title('Flux vs Expected Magnitude (Positive Flux Only)')
            plt.legend()
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Magnitude distribution comparison
            plt.subplot(2, 2, 2)
            plt.hist(expected_mag, bins=20, alpha=0.7, label='Expected from Flux', density=True)
            plt.hist(mag_data, bins=20, alpha=0.7, label='Observed Magnitudes', density=True)
            plt.xlabel('Magnitude (AB)')
            plt.ylabel('Density')
            plt.title('Magnitude Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Flux distribution
            plt.subplot(2, 2, 3)
            plt.hist(positive_flux, bins=20, alpha=0.7)
            plt.xlabel('Flux (nanomaggies)')
            plt.ylabel('Count')
            plt.title('Positive Flux Distribution')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Magnitude range comparison
            plt.subplot(2, 2, 4)
            plt.boxplot([expected_mag, mag_data], labels=['Expected', 'Observed'])
            plt.ylabel('Magnitude (AB)')
            plt.title('Magnitude Range Comparison')
            plt.grid(True, alpha=0.3)
            
        else:
            # If no valid data, show message
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, 'No valid flux data for plotting', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Valid Data')
        
        plt.tight_layout()
        plt.savefig('flux_magnitude_relationship.png', dpi=150, bbox_inches='tight')
        print("\n📊 Plot saved as 'flux_magnitude_relationship.png'")
        
    except Exception as e:
        print(f"Error creating plot: {e}")

def main():
    """Main function to run the analysis."""
    print("🔍 AION Flux-Magnitude Relationship Checker")
    print("=" * 50)
    
    # Load data
    flux_data, mag_data = load_test_data()
    
    if flux_data is None or mag_data is None:
        print("❌ Failed to load data")
        return
    
    # Analyze relationship
    flux_np, mag_np, expected_mag, differences = check_flux_magnitude_relationship(flux_data, mag_data)
    
    if flux_np is not None:
        # Create plot
        plot_relationship(flux_np, mag_np, expected_mag)
        
        print("\n" + "=" * 50)
        print("📋 SUMMARY:")
        print("• If differences are consistent (low std dev), they're likely the same objects")
        print("• If differences vary significantly, they may be different objects or calibrations")
        print("• The AB magnitude system is used: m_AB = -2.5 * log10(f_ν) - 48.60")
        print("• Flux is in nanomaggies: 1 nanomaggie = 3.631 × 10⁻²⁹ erg/s/cm²/Hz")

if __name__ == "__main__":
    main() 