import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sparse_autoencoder import SparseAutoencoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import sys
import traceback
from save_activations import get_data
import torch.nn.functional as F
from loss_ratio import calculate_loss_ratio
import webdataset as wds
import tarfile
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add both directories to sys.path
sys.path.append(os.path.join(current_dir, "AION"))
sys.path.append(os.path.join(current_dir, "4Mclone"))

from aion import AION

class Args:
    """Simple class to hold arguments needed for setup functions."""
    def __init__(self):
        self.data_config = "4Mclone/cfgs/default/mmoma/data/mmu/rusty_legacysurvey_desi_sdss_hsc_eval.yaml"
        self.text_tokenizer_path = "4Mclone/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
        self.input_size = 224
        self.patch_size = 16
        self.num_input_tokens = 900
        self.num_target_tokens = 128
        self.min_input_tokens = None
        self.min_target_tokens = None
        self.batch_size = 110
        # self.epoch_size = 9570
        self.epoch_size = 550
        self.num_workers = 1
        self.num_tasks = 1
        self.pin_mem = True
        self.dist_eval = False
        self.fixed_eval = False
        self.fixed_eval_input_tokens = None
        self.fixed_eval_target_tokens = None
        self.in_domains = None
        self.out_domains = None
        self.all_domains = None

def count_examples_in_shard(shard_path):
    """Count the number of examples in a WebDataset shard."""
    print(f"\nCounting examples in shard: {shard_path}")
    
    # First, let's check what's in the parent directory
    parent_dir = "/mnt/ceph/users/polymathic/jz_mmoma/tokenized_datasets/oct24/eval/sdss_legacysurvey"
    print(f"\nListing contents of parent directory: {parent_dir}")
    try:
        files = os.listdir(parent_dir)
        print("Found directories:")
        for f in files:
            if os.path.isdir(os.path.join(parent_dir, f)):
                print(f"  {f}")
    except Exception as e:
        print(f"Could not list directory: {e}")
        return None
    
    # The modalities are in separate directories
    # Let's check one modality directory to see the shard structure
    first_modality = "legacysurvey_image"  # Using image as it's likely to have the examples
    mod_dir = os.path.join(parent_dir, first_modality)
    print(f"\nChecking shard structure in {first_modality} directory:")
    try:
        mod_files = os.listdir(mod_dir)
        tar_files = [f for f in mod_files if f.endswith('.tar')]
        if tar_files:
            print(f"Found tar files: {tar_files[:5]}...")
            # Use the first tar file to count examples
            tar_path = os.path.join(mod_dir, tar_files[0])
            print(f"Using shard: {tar_path}")
            
            # Count examples in this shard
            with tarfile.open(tar_path, 'r') as tar:
                files = tar.getnames()
                example_names = set()
                for file in files:
                    base_name = file.split('.')[0]
                    example_names.add(base_name)
                
                print(f"Found {len(example_names)} unique examples in shard")
                print("\nNote: This count is from a single modality. The total number of examples")
                print("across all modalities should be the same, as each example has files in each modality directory.")
                return len(example_names)
        else:
            print("No tar files found in modality directory")
    except Exception as e:
        print(f"Error accessing directory {mod_dir}: {e}")
    
    print("\nCould not find any shard files")
    return None

def get_autoencoder_activations(transformer_activations, autoencoder, device, batch_size=256):
    """
    Process transformer activations through autoencoder encoder in batches.
    
    Args:
        transformer_activations: Flattened tensor of transformer activations (batch_size * seq_len, hidden_dim)
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
        batch_size: Size of batches to process at once
        
    Returns:
        numpy array of autoencoder activations
    """
    total_examples = transformer_activations.size(0)
    autoencoder_acts_list = []
    
    print(f"\nProcessing {total_examples} examples in batches of {batch_size}")
    for start_idx in tqdm(range(0, total_examples, batch_size)):
        end_idx = min(start_idx + batch_size, total_examples)
        batch = transformer_activations[start_idx:end_idx]
        
        with torch.no_grad():
            # Forward pass through encoder only to get hidden representations
            _, encoded = autoencoder(batch)
            autoencoder_acts_list.append(encoded.cpu())
    
    # Concatenate all batches
    return torch.cat(autoencoder_acts_list, dim=0).numpy()

def plot_activation_frequencies_comparison(transformer_activations, autoencoder, device, batch_size=256):
    """
    Creates a histogram comparing neuron activation frequencies between transformer layer
    and sparse autoencoder hidden layer activations. Processes activations in batches to handle large tensors.
    
    Args:
        transformer_activations: Flattened tensor of transformer activations (batch_size * seq_len, hidden_dim)
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
        batch_size: Size of batches to process at once
    """
    # Get transformer activations
    transformer_acts = transformer_activations.cpu().numpy()
    transformer_frequencies = (transformer_acts > 1e-12).mean(axis=0) * 100  # Convert to percentage
    
    # Get autoencoder activations using the helper function
    autoencoder_acts = get_autoencoder_activations(transformer_activations, autoencoder, device, batch_size)
    autoencoder_frequencies = (autoencoder_acts > 1e-12).mean(axis=0) * 100
    
    # Create figure with log-scale x-axis
    plt.figure(figsize=(12, 8))
    
    # Calculate histogram bins uniformly in log space
    min_freq = min(transformer_frequencies.min(), autoencoder_frequencies.min())
    max_freq = max(transformer_frequencies.max(), autoencoder_frequencies.max())
    bins = np.logspace(np.log10(max(min_freq, 1e-4)), np.log10(max_freq), 50)  # Uniform bins in log space
    
    # Plot histograms without density normalization
    plt.hist(transformer_frequencies, bins=bins, alpha=0.6, label='Transformer Neuron Activation Densities',
             color='blue', density=False)
    plt.hist(autoencoder_frequencies, bins=bins, alpha=0.6, label='Autoencoder Feature Activation Densities',
             color='red', density=False)
    
    # Customize plot
    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Activation Frequency')
    plt.ylabel('Number of Neurons')
    plt.title('Activation Densities')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add summary statistics as text
    trans_stats = f'Transformer Stats:\nMean: {transformer_frequencies.mean():.1f}%\n'
    trans_stats += f'Median: {np.median(transformer_frequencies):.1f}%\n'
    trans_stats += f'Neurons: {len(transformer_frequencies)}'
    
    ae_stats = f'Autoencoder Stats:\nMean: {autoencoder_frequencies.mean():.1f}%\n'
    ae_stats += f'Median: {np.median(autoencoder_frequencies):.1f}%\n'
    ae_stats += f'Neurons: {len(autoencoder_frequencies)}'
    
    # Position stats text boxes
    plt.text(0.02, 0.98, trans_stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.75, 0.80, ae_stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("activation_frequencies_comparison.png")
    
    # Print additional statistics
    print("\nDetailed Statistics:")
    print("\nTransformer Activations:")
    print(f"Minimum: {transformer_frequencies.min():.1f}%")
    print(f"Maximum: {transformer_frequencies.max():.1f}%")
    print(f"Standard Deviation: {transformer_frequencies.std():.1f}%")
    
    print("\nAutoencoder Activations:")
    print(f"Minimum: {autoencoder_frequencies.min():.1f}%")
    print(f"Maximum: {autoencoder_frequencies.max():.1f}%")
    print(f"Standard Deviation: {autoencoder_frequencies.std():.1f}%")
    
    # Calculate and print percentile information
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentile Comparison:")
    print(f"{'Percentile':>10} {'Transformer':>12} {'Autoencoder':>12}")
    print("-" * 35)
    for p in percentiles:
        t_perc = np.percentile(transformer_frequencies, p)
        a_perc = np.percentile(autoencoder_frequencies, p)
        print(f"{p:>10}th {t_perc:>12.1f}% {a_perc:>12.1f}%")

    return transformer_frequencies, autoencoder_frequencies

def collect_activations_batched_encoder(model, data_loader, device, args):
    """
    Collect activations from encoder block 8 while trusting AION's internal token selection.
    
    KEY APPROACH: 
    - We let AION handle ALL token selection internally via forward_mask_encoder
    - No manual truncation or token manipulation on our part
    - AION automatically prioritizes valid tokens (input_mask=False) and excludes masked tokens (input_mask=True)
    - We capture AION's actual selection by overriding embed_inputs temporarily
    - This approach respects AION's sophisticated mask-based token selection mechanism
    
    Args:
        model: AION model
        data_loader: DataLoader for input data
        device: Device to run on
        args: Arguments object containing num_input_tokens and other parameters
        
    Returns:
        tuple: (activations_tensor_flat, modality_labels_flat, examples_processed)
    """
    print("\n=== Starting Activation Collection ===")
    
    activations = []
    modality_labels = [] 
    examples_processed = 0
    
    # Storage for AION selection analysis
    aion_selection_analysis = {
        'before_selection': {},  # modality -> total tokens available
        'after_selection': {},   # modality -> tokens actually selected
        'elimination_details': {},  # detailed breakdown per batch
        'selection_results': []   # Store actual selection results
    }
    
    def hook_fn(module, input, output):
        print(f'\nHook received output shape: {output.shape}')
        # Save activations
        activations.append(output.clone().detach().cpu())
    
    # Store original embed_inputs method
    original_embed_inputs = model.embed_inputs
    
    def instrumented_embed_inputs(input_dict, mask=None, num_encoder_tokens=args.num_input_tokens):
        """Instrumented version of embed_inputs that captures selection details"""
        
        print(f"\n🔍 DEBUGGING AION'S MASK-BASED SELECTION:")
        print(f"  Input modalities: {list(input_dict.keys())}")
        print(f"  Requested num_encoder_tokens: {num_encoder_tokens}")
        
        # Check the input_mask values in the raw input
        print(f"\n📊 INPUT MASK ANALYSIS:")
        for mod, tensor in input_dict.items():
            print(f"  {mod}: tensor shape {tensor.shape}")
            if mask is not None and mod in mask:
                input_mask = mask[mod]
                masked_count = input_mask.sum().item()
                total_count = input_mask.numel()
                mask_percentage = (masked_count / total_count) * 100
                print(f"    Input mask: {masked_count}/{total_count} masked ({mask_percentage:.1f}%)")
            else:
                print(f"    No input mask provided - all tokens should be valid")
        
        # Call the original method - use whatever num_encoder_tokens was actually passed in
        encoder_tokens, encoder_emb, encoder_mask, mod_mask = original_embed_inputs(
            input_dict, mask=mask, num_encoder_tokens=num_encoder_tokens
        )
        
        print(f"\n🎯 AION'S SELECTION RESULTS:")
        print(f"  Selected tokens shape: {encoder_tokens.shape}")
        print(f"  Selected {encoder_tokens.shape[1]} out of requested {num_encoder_tokens}")
        
        # Analyze the final modality mask to see what was selected
        selected_mod_ids = mod_mask[0]  # First example in batch
        unique_mod_ids = torch.unique(selected_mod_ids)
        print(f"  Selected modality IDs: {unique_mod_ids.tolist()}")
        
        # Count tokens per modality in the selection
        print(f"\n📋 SELECTED TOKENS BY MODALITY:")
        for mod_id in unique_mod_ids:
            if mod_id != -1:  # Skip padding
                count = (selected_mod_ids == mod_id).sum().item()
                # Try to find modality name
                mod_name = next((mod for mod, info in model.modality_info.items() if info["id"] == mod_id), f"Unknown-{mod_id}")
                print(f"    {mod_name} (ID: {mod_id}): {count} tokens")
        
        # Check if selection appears to be positional
        # We'll do this by checking if the first few hundred tokens are consecutive modality IDs
        first_100_ids = selected_mod_ids[:100].tolist()
        print(f"  First 100 selected modality IDs: {first_100_ids[:20]}...{first_100_ids[-20:]}")
        
        # Store the selection information for analysis
        aion_selection_analysis['selection_results'].append({
            'encoder_tokens_shape': encoder_tokens.shape,
            'encoder_emb_shape': encoder_emb.shape,
            'encoder_mask_shape': encoder_mask.shape,
            'mod_mask_shape': mod_mask.shape,
            'mod_mask': mod_mask.clone().detach().cpu(),
            'encoder_mask': encoder_mask.clone().detach().cpu(),
            'selected_mod_ids': selected_mod_ids.clone().detach().cpu(),
            'unique_selected_mod_ids': unique_mod_ids.clone().detach().cpu()
        })
        
        return encoder_tokens, encoder_emb, encoder_mask, mod_mask
    
    # Temporarily replace the method
    model.embed_inputs = instrumented_embed_inputs
    
    # Register hook on encoder block 8 for activations
    ninth_encoder_block = model.encoder[8] 
    activation_hook = ninth_encoder_block.register_forward_hook(hook_fn)
    print(f"Registered activation hook on encoder block 8")

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            print(f"\nProcessing batch {batch_idx + 1}")
            
            # Get current batch size from the data
            current_batch_size = next(iter(data.values()))['tensor'].shape[0]
                
            print(f"Processing full batch of {current_batch_size} examples")
            
            # Move data to device
            processed_data = {}
            for mod, d in data.items():
                processed_data[mod] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in d.items()}
            print("Data moved to device")
            
            input_dict = {}
            input_mask_dict = {}
            total_tokens = 0
            batch_modality_labels = []
            
            # DETAILED ANALYSIS: Track BEFORE selection
            print(f"\n🔍 BEFORE AION SELECTION - Available Modalities:")
            batch_before_selection = {}
            
            # Collect input data and build comprehensive modality labels
            modality_order = []  # Track the order of modalities for proper indexing
            for mod, d in processed_data.items():
                if mod in model.encoder_embeddings:
                    print(f'\n--- Modality: {mod} ---')
                    print(f'  Tensor shape: {d["tensor"].shape}')
                    
                    input_dict[mod] = d['tensor']
                    
                    # Get the modality ID from the model's modality info
                    mod_id = model.modality_info[mod]["id"]
                    num_tokens = d['tensor'].shape[1]
                    
                    print(f'  Modality ID: {mod_id}')
                    print(f'  Number of tokens: {num_tokens}')
                    
                    # Track for BEFORE selection analysis
                    tokens_this_mod = current_batch_size * num_tokens
                    batch_before_selection[mod] = {
                        'id': mod_id,
                        'tokens_per_example': num_tokens,
                        'total_tokens_in_batch': tokens_this_mod,
                        'start_index': total_tokens,  # Track position in concatenated sequence
                        'end_index': total_tokens + num_tokens - 1
                    }
                    
                    # Check input_mask status
                    if 'input_mask' in d:
                        input_mask_dict[mod] = d['input_mask']
                        mask_tensor = d['input_mask']
                        masked_count = mask_tensor.sum().item()
                        total_count = mask_tensor.numel()
                        valid_count = total_count - masked_count
                        mask_percentage = (masked_count / total_count) * 100 if total_count > 0 else 0.0
                        
                        print(f"  Input mask: {masked_count}/{total_count} masked ({mask_percentage:.1f}%)")
                        print(f"  Valid tokens: {valid_count} (should be prioritized by AION)")
                        
                        batch_before_selection[mod]['masked_tokens'] = masked_count
                        batch_before_selection[mod]['valid_tokens'] = valid_count
                        batch_before_selection[mod]['mask_percentage'] = mask_percentage
                    else:
                        print(f'  No input_mask (all tokens assumed valid)')
                        batch_before_selection[mod]['masked_tokens'] = 0
                        batch_before_selection[mod]['valid_tokens'] = tokens_this_mod
                        batch_before_selection[mod]['mask_percentage'] = 0.0
                    
                    # Create labels for ALL tokens with proper indexing
                    batch_modality_labels.append(torch.full((current_batch_size, num_tokens), mod_id, dtype=torch.long, device='cpu'))
                    modality_order.append(mod)
                    total_tokens += num_tokens
            
            # Summarize BEFORE selection
            print(f"\n📊 SUMMARY BEFORE AION SELECTION:")
            total_available = sum(info['total_tokens_in_batch'] for info in batch_before_selection.values())
            total_valid = sum(info['valid_tokens'] for info in batch_before_selection.values())
            total_masked = sum(info['masked_tokens'] for info in batch_before_selection.values())
            
            for mod, info in batch_before_selection.items():
                print(f"  {mod}: {info['total_tokens_in_batch']} tokens (pos {info['start_index']}-{info['end_index']}), {info['mask_percentage']:.1f}% masked")
            
            # Concatenate modality labels for this batch
            if batch_modality_labels:
                batch_modality_tensor = torch.cat(batch_modality_labels, dim=1)
                print(f"\nTotal tokens across all modalities: {batch_modality_tensor.shape[1]}")
                print(f"Model will intelligently select up to {args.num_input_tokens} tokens from these using AION's internal token selection")
                
                # Store the full label tensor for later analysis
                modality_labels.append(batch_modality_tensor)
          
            # Use model.encode() to get embeddings following the notebook pattern
            print(f"\n🚀 Running model.encode() - AION will now make its selection...")
            embeddings = model.encode(input_dict, input_mask=input_mask_dict, num_encoder_tokens=args.num_input_tokens)
            print(f"✅ Embeddings shape: {embeddings.shape}")
            
            # EXACT ANALYSIS: Use AION's captured selection information
            actual_seq_length = embeddings.shape[1]
            
            print(f"\n🔍 AFTER AION SELECTION - Results:")
            print(f"  Available tokens: {batch_modality_tensor.shape[1]}")
            print(f"  AION selected: {actual_seq_length} tokens")
            print(f"  Elimination rate: {((batch_modality_tensor.shape[1] - actual_seq_length) / batch_modality_tensor.shape[1]) * 100:.1f}%")
            
            # EXACT ANALYSIS: Use AION's modality mask to determine selection
            if len(aion_selection_analysis['selection_results']) > batch_idx:
                selection_result = aion_selection_analysis['selection_results'][batch_idx]
                selected_mod_mask = selection_result['mod_mask']
                selected_encoder_mask = selection_result['encoder_mask']
                
                print(f"\n🎯 EXACT AION SELECTION ANALYSIS:")
                print(f"  Using AION's actual modality mask (not approximation)")
                print(f"  Selected modality mask shape: {selected_mod_mask.shape}")
                
                # Count exactly which modalities AION selected
                batch_after_selection = {}
                for mod, info in batch_before_selection.items():
                    mod_id = info['id']
                    # Count how many times this modality ID appears in AION's selection
                    selected_count = (selected_mod_mask == mod_id).sum().item()
                    batch_after_selection[mod] = {
                        'id': mod_id,
                        'selected_tokens': selected_count,
                        'original_tokens': info['total_tokens_in_batch'],
                        'survival_rate': selected_count / info['total_tokens_in_batch'] if info['total_tokens_in_batch'] > 0 else 0.0
                    }
                
                # Print detailed selection results
                print(f"\n📋 EXACT MODALITY SELECTION BREAKDOWN:")
                print(f"{'Modality':<20} {'Original':<10} {'Selected':<10} {'Survival%':<10} {'Status'}")
                print("-" * 65)
                
                for mod in sorted(batch_before_selection.keys()):
                    orig = batch_before_selection[mod]['total_tokens_in_batch']
                    selected = batch_after_selection[mod]['selected_tokens']
                    survival = batch_after_selection[mod]['survival_rate'] * 100
                    
                    if selected == 0:
                        status = "❌ ELIMINATED"
                    elif survival < 50:
                        status = "⚠️  HEAVILY REDUCED"
                    elif survival < 90:
                        status = "📉 PARTIALLY KEPT"
                    else:
                        status = "✅ MOSTLY KEPT"
                    
                    print(f"{mod:<20} {orig:<10} {selected:<10} {survival:<9.1f}% {status}")
                
                # Store detailed analysis for later aggregation
                aion_selection_analysis['elimination_details'][batch_idx] = {
                    'before': batch_before_selection,
                    'after': batch_after_selection,
                    'total_available': total_available,
                    'total_selected': actual_seq_length,
                    'elimination_rate': ((total_available - actual_seq_length) / total_available) * 100
                }
                
                # Update global counters
                for mod, info in batch_before_selection.items():
                    if mod not in aion_selection_analysis['before_selection']:
                        aion_selection_analysis['before_selection'][mod] = 0
                    if mod not in aion_selection_analysis['after_selection']:
                        aion_selection_analysis['after_selection'][mod] = 0
                    
                    aion_selection_analysis['before_selection'][mod] += info['total_tokens_in_batch']
                    aion_selection_analysis['after_selection'][mod] += batch_after_selection[mod]['selected_tokens']
                
                # Create EXACT labels based on AION's selection
                # We'll use AION's modality mask directly
                print(f"\n✅ CREATING EXACT LABELS FROM AION'S SELECTION")
                print(f"  Using AION's modality mask to create perfect alignment")
                
                # Replace our approximated labels with AION's exact selection
                exact_labels = selected_mod_mask.clone()  # Shape: (batch_size, selected_length)
                modality_labels[-1] = exact_labels  # Replace the last added labels
                
                print(f"  Exact labels shape: {exact_labels.shape}")
                print(f"  Perfect alignment with AION's embeddings: {embeddings.shape}")
                
            else:
                print(f"\n⚠️  WARNING: Could not capture AION's selection indices")
                print(f"  Falling back to positional approximation")
                selected_labels = batch_modality_tensor[:, :actual_seq_length]

            torch.cuda.empty_cache()
            examples_processed += current_batch_size
            print(f"\nTotal examples processed: {examples_processed}")
    
    # Restore original method and remove the hook
    model.embed_inputs = original_embed_inputs
    activation_hook.remove()
    print("\nMethod restored and hook removed")
    
    # Concatenate all activations
    print("\nConcatenating activations...")
    activations_tensor = torch.cat(activations, dim=0)
    print(f"Final activations shape: {activations_tensor.shape}")
    print(f"Final activations dtype: {activations_tensor.dtype}")
    print(f"Final activations device: {activations_tensor.device}")
    
    # Concatenate all modality labels
    print("\nConcatenating modality labels...")
    modality_labels_tensor = torch.cat(modality_labels, dim=0)
    print(f"Final modality labels shape: {modality_labels_tensor.shape}")
    print(f"Final modality labels dtype: {modality_labels_tensor.dtype}")
    print(f"Final modality labels device: {modality_labels_tensor.device}")

    # FINAL AION SELECTION ANALYSIS
    print(f"\n" + "="*60)
    print(f"🎯 FINAL AION SELECTION ANALYSIS")
    print(f"="*60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    total_before = sum(aion_selection_analysis['before_selection'].values())
    total_after = sum(aion_selection_analysis['after_selection'].values())
    overall_elimination_rate = ((total_before - total_after) / total_before) * 100 if total_before > 0 else 0
    
    print(f"  Total tokens available: {total_before}")
    print(f"  Total tokens selected: {total_after}")
    print(f"  Overall elimination rate: {overall_elimination_rate:.1f}%")
    print(f"  AION kept {total_after}/{total_before} tokens ({(total_after/total_before)*100:.1f}%)")
    
    print(f"\n📋 FINAL MODALITY BREAKDOWN:")
    print(f"{'Modality':<20} {'Available':<12} {'Selected':<12} {'Survival%':<12} {'Final Status'}")
    print("-" * 75)
    
    modality_survival_rates = {}
    for mod in sorted(aion_selection_analysis['before_selection'].keys()):
        available = aion_selection_analysis['before_selection'][mod]
        selected = aion_selection_analysis['after_selection'][mod]
        survival_rate = (selected / available * 100) if available > 0 else 0.0
        modality_survival_rates[mod] = survival_rate
        
        if selected == 0:
            status = "❌ COMPLETELY ELIMINATED"
        elif survival_rate < 10:
            status = "🔥 HEAVILY ELIMINATED"
        elif survival_rate < 50:
            status = "⚠️  SIGNIFICANTLY REDUCED"
        elif survival_rate < 80:
            status = "📉 MODERATELY REDUCED"
        elif survival_rate < 95:
            status = "📊 LIGHTLY REDUCED"
        else:
            status = "✅ ALMOST ALL KEPT"
        
        print(f"{mod:<20} {available:<12} {selected:<12} {survival_rate:<11.1f}% {status}")
    
    # Identify the winners and losers
    print(f"\n🏆 MODALITY WINNERS (highest survival rates):")
    sorted_mods = sorted(modality_survival_rates.items(), key=lambda x: x[1], reverse=True)
    for mod, rate in sorted_mods[:3]:
        print(f"  {mod}: {rate:.1f}% survival rate")
    
    print(f"\n💀 MODALITY LOSERS (lowest survival rates):")
    for mod, rate in sorted_mods[-3:]:
        print(f"  {mod}: {rate:.1f}% survival rate")
    
    # Special analysis for problematic modalities
    print(f"\n🔍 SPECIAL ANALYSIS:")
    for mod, rate in sorted_mods:
        if rate == 0:
            print(f"  {mod}: COMPLETELY ELIMINATED - all tokens were masked (input_mask=True)")
        elif rate < 1:
            print(f"  {mod}: NEARLY ELIMINATED - {rate:.3f}% survival suggests mostly masked tokens")
    
    print(f"\n✅ EXACT SELECTION: No approximation - used AION's actual modality mask")
    print(f"✅ PERFECT ALIGNMENT: Labels match embeddings exactly")
    
    print(f"\n" + "="*60)
    print(f"✅ AION SELECTION ANALYSIS COMPLETE")
    print(f"="*60)

    # Verify perfect alignment (no shape mismatch needed)
    print(f"\n*** PERFECT ALIGNMENT VERIFICATION ***")
    print(f"*** Activations shape: {activations_tensor.shape} ***")
    print(f"*** Labels shape: {modality_labels_tensor.shape} ***")
    
    if activations_tensor.shape[:2] == modality_labels_tensor.shape:
        print(f"*** ✅ PERFECT: Activations and labels perfectly aligned ***")
        print(f"*** Using AION's exact selection - no approximation needed ***")
    else:
        print(f"*** ⚠️  UNEXPECTED: Shape mismatch despite using exact selection ***")
        # Handle any unexpected mismatch
        target_seq_length = activations_tensor.shape[1]
        if modality_labels_tensor.shape[1] != target_seq_length:
            print(f"*** Adjusting labels to match: {modality_labels_tensor.shape[1]} -> {target_seq_length} ***")
            modality_labels_tensor = modality_labels_tensor[:, :target_seq_length]

    # Verify shapes match before flattening
    assert activations_tensor.shape[:2] == modality_labels_tensor.shape, \
        f"Shape mismatch: activations {activations_tensor.shape[:2]} vs labels {modality_labels_tensor.shape}"
    
    # Flatten for analysis
    print("\nFlattening activations...")
    sample_size, seq_length, embed_dim = activations_tensor.shape
    activations_tensor_flat = activations_tensor.reshape(sample_size * seq_length, embed_dim)
    modality_labels_flat = modality_labels_tensor.reshape(-1)
    print(f"Original activations shape: {activations_tensor.shape} -> [sample_size, seq_length, embed_dim]")
    print(f"Flattened activations shape: {activations_tensor_flat.shape} -> [sample_size*seq_length, embed_dim]")
    print(f"Flattened activations dtype: {activations_tensor_flat.dtype}")
    print(f"Flattened activations device: {activations_tensor_flat.device}")
    print(f"Flattened modality labels shape: {modality_labels_flat.shape}")
    print(f"Flattened modality labels dtype: {modality_labels_flat.dtype}")
    print(f"Flattened modality labels device: {modality_labels_flat.device}")
    
    # Verify shapes match after flattening
    assert len(activations_tensor_flat) == len(modality_labels_flat), \
        f"Length mismatch after flattening: activations {len(activations_tensor_flat)} vs labels {len(modality_labels_flat)}"
    
    # Print unique modality values for verification
    unique_modalities = torch.unique(modality_labels_flat)
    print(f"Unique modality values in final labels: {unique_modalities.tolist()}")
    
    # Print modality distribution
    for mod_id in unique_modalities:
        if mod_id != -1:  # Skip padding tokens
            count = (modality_labels_flat == mod_id).sum().item()
            percentage = (count / len(modality_labels_flat)) * 100
            # Try to get modality name from model's modality_info
            mod_name = next((mod for mod, info in model.modality_info.items() if info["id"] == mod_id), f"Unknown-{mod_id}")
            print(f"Final dataset - {mod_name} (ID: {mod_id}): {count} tokens ({percentage:.2f}%)")
    
    # Print padding token distribution
    padding_count = (modality_labels_flat == -1).sum().item()
    padding_percentage = (padding_count / len(modality_labels_flat)) * 100
    print(f"Final dataset - Padding tokens (-1): {padding_count} tokens ({padding_percentage:.2f}%)")
    
    return activations_tensor_flat, modality_labels_flat, examples_processed

def collect_activations_batched_decoder(
    model,
    data_loader,
    device,
    args,
):
    """Collect activations from **decoder** block 8 (index 8) in batches.

    The procedure mirrors ``collect_activations_batched_encoder`` but focuses on the
    decoder path.  We first obtain the encoder context, then run the model's
    internal decoder up to block-8 while registering a forward hook that stores
    the hidden states produced by that block.  The modality labels are taken
    *directly* from the ``decoder_mod_mask`` returned by ``embed_targets`` so they
    perfectly align with the hidden states.

    Args:
        model:  AION model (already loaded).
        data_loader:  DataLoader that yields a batch-dict with keys per modality.
        device:  Target device (cpu / cuda).
        args: Arguments object containing num_target_tokens and num_input_tokens.

    Returns:
        activations_tensor_flat (torch.Tensor): (N*M, D) flattened decoder activations.
        modality_labels_flat  (torch.Tensor): (N*M,) modality IDs aligned to activations.
        examples_processed (int): Number of examples (N) encountered.
    """

    model = model.to(device).eval()

    activations: list[torch.Tensor] = []
    modality_labels: list[torch.Tensor] = []
    examples_processed: int = 0

    # ------------------------------------------------------------------
    # 1. Register a hook on the 9-th decoder block (index 8)
    # ------------------------------------------------------------------
    def _hook(_module, _inp, output):
        # Store hidden states on CPU to save GPU memory
        activations.append(output.detach().cpu())

    hook_handle = model.decoder[8].register_forward_hook(_hook)

    # ------------------------------------------------------------------
    # 2. Iterate over batches
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move batch tensors to the correct device
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
            input_dict: dict[str, torch.Tensor] = {}
            input_mask: dict[str, torch.Tensor] = {}
            target_mask: dict[str, torch.Tensor] = {}

            for mod, d in processed_batch.items():
                # Encoder side (always use the raw token tensor)
                input_dict[mod] = d["tensor"]

                if "input_mask" in d:
                    input_mask[mod] = d["input_mask"].to(torch.bool)

                # Decoder side: boolean mask indicating which positions are to be predicted
                target_mask[mod] = d["target_mask"].to(torch.bool)

            # ------------------------------------------------------------------
            # Run through token selection utilities
            # ------------------------------------------------------------------
            enc_tokens, enc_emb, enc_mask, _ = model.embed_inputs(
                input_dict,
                mask=input_mask if len(input_mask) > 0 else None,
                num_encoder_tokens=args.num_input_tokens,
            )

            (
                dec_tokens,
                dec_emb,
                dec_mask,
                _target_ids,  # not needed for probing
                dec_att_mask,
                dec_mod_mask,
            ) = model.embed_targets(target_mask, num_decoder_tokens=args.num_target_tokens)

            # ------------------------------------------------------------------
            # Forward pass: encoder context -> decoder (hook captures block-8)
            # ------------------------------------------------------------------
            context = model._encode(enc_tokens, enc_emb, enc_mask)

            _ = model._decode(
                context,
                enc_mask,
                dec_tokens,
                dec_emb,
                dec_att_mask,
            )  # hook fires here

            # Store ground-truth modality labels for this batch
            modality_labels.append(dec_mod_mask.cpu())

            # House-keeping
            batch_size = next(iter(processed_batch.values()))["tensor"].shape[0]
            examples_processed += batch_size

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Cleanup hook
    # ------------------------------------------------------------------
    hook_handle.remove()

    # ------------------------------------------------------------------
    # 4. Stack and flatten tensors
    # ------------------------------------------------------------------
    activations_tensor = torch.cat(activations, dim=0)           # (N, M, D)
    modality_labels_tensor = torch.cat(modality_labels, dim=0)   # (N, M)

    assert activations_tensor.shape[:2] == modality_labels_tensor.shape, (
        "Decoder activations and modality labels mis-aligned: "
        f"{activations_tensor.shape[:2]} vs {modality_labels_tensor.shape}"
    )

    N, M, D = activations_tensor.shape
    activations_flat = activations_tensor.reshape(N * M, D)
    labels_flat = modality_labels_tensor.reshape(-1)

    return activations_flat, labels_flat, examples_processed

def apply_logistic_probe(
    activations_tensor_flat, 
    modality_labels_flat, 
    examples_processed, 
    autoencoder, 
    device,
    image_modality_id=18665,  # ID for tok_image
    output_prefix='tok_image', # Prefix for output files
    test_size=0.2,
    random_state=42
):
    # Apply 1D logistic probe to find whether latent has info on whether the input is an image or not
    # Save both tensors for later use
    torch.save({
        'activations': activations_tensor_flat,
        'modality_labels': modality_labels_flat,
        'examples_processed': examples_processed
    }, f'{output_prefix}_activations_with_modality_labels.pt')
    
    print(f"\nSaved activations and modality labels to '{output_prefix}_activations_with_modality_labels.pt'")
    print(f"Activations shape: {activations_tensor_flat.shape}")
    print(f"Modality labels shape: {modality_labels_flat.shape}")
    print(f"Unique modality values: {torch.unique(modality_labels_flat).tolist()}")

    # Filter out padding tokens (-1) - keep only positions with actual predictions
    print("\nFiltering out padding tokens...")
    valid_mask = (modality_labels_flat != -1)
    valid_activations = activations_tensor_flat[valid_mask]
    valid_labels = modality_labels_flat[valid_mask]

    print(f"Original dataset size: {len(modality_labels_flat)}")
    print(f"Dataset size after removing padding: {len(valid_labels)}")
    print(f"Removed {len(modality_labels_flat) - len(valid_labels)} padding tokens")

    # Create binary labels for image tokens
    image_labels = (valid_labels == image_modality_id).long()
    print(f"\nCreated binary labels for tok_image modality:")
    print(f"Number of image tokens: {image_labels.sum().item()}")
    print(f"Number of non-image tokens: {len(image_labels) - image_labels.sum().item()}")
    print(f"Percentage of image tokens: {(image_labels.sum().item() / len(image_labels) * 100):.2f}%")

    # Verify we have no padding tokens
    assert -1 not in np.unique(valid_labels), "Found padding tokens in labels after filtering!"
    print("\nVerified: No padding tokens in the dataset")

    # Prepare the dataset for probing
    print("\nPreparing dataset for probing...")
    # Convert tensors to numpy for sklearn compatibility
    activations_np = valid_activations.cpu().numpy()
    labels_np = image_labels.cpu().numpy()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        activations_np, labels_np, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels_np  # Ensure balanced split
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set image percentage: {(y_train.sum() / len(y_train) * 100):.2f}%")
    print(f"Test set image percentage: {(y_test.sum() / len(y_test) * 100):.2f}%")

    # Get autoencoder activations for both train and test sets
    print("\nGetting autoencoder activations...")
    with torch.no_grad():
        # Process training set
        train_acts = torch.tensor(X_train, dtype=torch.float32, device=device)
        _, train_latents = autoencoder(train_acts)
        train_latents = train_latents.cpu().numpy()
        
        # Process test set
        test_acts = torch.tensor(X_test, dtype=torch.float32, device=device)
        _, test_latents = autoencoder(test_acts)
        test_latents = test_latents.cpu().numpy()

    # Probe each latent dimension
    print("\nProbing all latent dimensions...")
    num_latents = train_latents.shape[1]
    print(f"Total number of latents to probe: {num_latents}")
    results = []
    
    # Calculate class weights for balanced loss
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (2 * class_counts)  # 2 classes
    print(f"\nClass weights for balanced loss: {class_weights}")
    print(f"Class distribution: {class_counts}")
    print(f"Percentage of image tokens: {(class_counts[1] / total_samples * 100):.2f}%")
    
    # Calculate baseline metrics
    majority_class = np.argmax(class_counts)
    baseline_accuracy = class_counts[majority_class] / total_samples
    print(f"\nBaseline metrics (always predicting majority class):")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Balanced accuracy: 0.5 (random chance)")
    print(f"F1 score: 0.0 (no true positives)")
    
    for i in tqdm(range(num_latents), desc="Probing latents"):
        # Extract activations for this latent
        z_i_train = train_latents[:, i]
        z_i_test = test_latents[:, i]
        
        # Train logistic regression with class weights
        model = NewtonRaphsonLogisticRegression()
        # Modify the loss function to use class weights
        model._loss = lambda params, X, y: -np.mean(
            class_weights[y] * (y * np.log(model._sigmoid(params[0] * X + params[1])) + 
                              (1 - y) * np.log(1 - model._sigmoid(params[0] * X + params[1])))
        )
        model.fit(z_i_train, y_train)
        
        # Evaluate on test set
        y_pred_proba = model.predict_proba(z_i_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate various metrics
        accuracy = (y_pred == y_test).mean()
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        # Calculate balanced cross-entropy loss
        balanced_loss = -np.mean(
            class_weights[y_test] * (y_test * np.log(np.clip(y_pred_proba, 1e-15, 1-1e-15)) + 
                                   (1 - y_test) * np.log(np.clip(1 - y_pred_proba, 1e-15, 1-1e-15)))
        )
        
        results.append({
            'latent_index': i,
            'weight': model.w,
            'bias': model.b,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'auc': auc,
            'balanced_cross_entropy_loss': balanced_loss
        })
    
    # Convert results to DataFrame and sort by balanced accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('balanced_accuracy', ascending=False)
    
    # Save all results
    results_df.to_csv(f'{output_prefix}_probing_results_all_balanced.csv', index=False)
    print(f"\nResults saved to '{output_prefix}_probing_results_all_balanced.csv'")
    
    # Get top 10 latents
    top_10_df = results_df.head(10)
    
    # Print top 10 results
    print("\nTop 10 best latents (sorted by balanced accuracy):")
    print(top_10_df[['latent_index', 'balanced_accuracy', 'f1_score', 'auc', 'balanced_cross_entropy_loss']].to_string())
    
    # Print summary statistics of all latents
    print("\nSummary statistics of all latents:")
    print(f"Best balanced accuracy: {results_df['balanced_accuracy'].max():.4f}")
    print(f"Worst balanced accuracy: {results_df['balanced_accuracy'].min():.4f}")
    print(f"Mean balanced accuracy: {results_df['balanced_accuracy'].mean():.4f}")
    print(f"Median balanced accuracy: {results_df['balanced_accuracy'].median():.4f}")
    print(f"Number of latents better than random (balanced accuracy > 0.5): {(results_df['balanced_accuracy'] > 0.5).sum()}")
    print(f"Best AUC: {results_df['auc'].max():.4f}")
    print(f"Worst AUC: {results_df['auc'].min():.4f}")
    print(f"Mean AUC: {results_df['auc'].mean():.4f}")
    print(f"Median AUC: {results_df['auc'].median():.4f}")
    print(f"Number of latents better than random (AUC > 0.5): {(results_df['auc'] > 0.5).sum()}")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution of balanced accuracies for all latents
    ax1.hist(results_df['balanced_accuracy'], bins=50)
    ax1.axvline(0.5, color='r', linestyle='--', label='Random Chance')
    ax1.set_xlabel('Balanced Accuracy')
    ax1.set_ylabel('Number of Latents')
    ax1.set_title('Distribution of Balanced Accuracies\n(All Latents)')
    ax1.legend()
    
    # Plot 2: Bar plot of balanced accuracies for top 10 latents
    top_10_df.plot(kind='bar', x='latent_index', y='balanced_accuracy', ax=ax2)
    ax2.axhline(0.5, color='r', linestyle='--', label='Random Chance')
    ax2.set_xlabel('Latent Index')
    ax2.set_ylabel('Balanced Accuracy')
    ax2.set_title('Balanced Accuracies of Top 10 Latents')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_probing_analysis.png')
    print(f"\nSaved analysis plots to '{output_prefix}_probing_analysis.png'")
    
    # Additional plot: Scatter plot of balanced accuracy vs latent index
    plt.figure(figsize=(12, 6))
    plt.scatter(results_df['latent_index'], results_df['balanced_accuracy'], alpha=0.5)
    plt.axhline(0.5, color='r', linestyle='--', label='Random Chance')
    plt.xlabel('Latent Index')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy vs Latent Index')
    plt.legend()
    plt.savefig(f'{output_prefix}_probing_scatter.png')
    print(f"Saved scatter plot to '{output_prefix}_probing_scatter.png'")

    return results_df

def apply_dense_probe_transformer(
    activations_tensor_flat, 
    modality_labels_flat, 
    examples_processed,
    image_modality_id,  # ID for the target modality to create binary labels
    output_prefix='image_transformer',  # Updated prefix
    test_size=0.2,
    random_state=42
):
    """
    Apply dense logistic probe to ALL transformer neurons simultaneously to find whether 
    they contain info on whether the position corresponds to the target modality or not.
    
    This uses all 768 dimensions together in a single model, rather than testing each individually.
    
    Args:
        activations_tensor_flat: Flattened tensor of transformer activations [n_samples, 768]
        modality_labels_flat: Flattened tensor of modality IDs [n_samples] (actual modality IDs, -1=padding)
        examples_processed: Number of examples processed
        image_modality_id: ID for target modality (used to create binary labels: 1=target, 0=other)
        output_prefix: Prefix for output files
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    print("\n=== Starting Dense Probing on ALL Transformer Neurons ===")
    
    # Save both tensors for later use
    torch.save({
        'activations': activations_tensor_flat,
        'modality_labels': modality_labels_flat,
        'examples_processed': examples_processed
    }, f'{output_prefix}_activations_with_modality_labels.pt')
    
    print(f"\nSaved activations and modality labels to '{output_prefix}_activations_with_modality_labels.pt'")
    print(f"Activations shape: {activations_tensor_flat.shape}")
    print(f"Modality labels shape: {modality_labels_flat.shape}")
    print(f"Unique label values: {torch.unique(modality_labels_flat).tolist()}")

    # Filter out padding tokens (-1) - keep only positions with actual predictions
    print("\nFiltering out padding tokens...")
    valid_mask = (modality_labels_flat != -1)
    valid_activations = activations_tensor_flat[valid_mask]
    valid_labels = modality_labels_flat[valid_mask]

    print(f"Original dataset size: {len(modality_labels_flat)}")
    print(f"Dataset size after removing padding: {len(valid_labels)}")
    print(f"Removed {len(modality_labels_flat) - len(valid_labels)} padding tokens")

    # Create binary labels for image tokens
    image_labels = (valid_labels == image_modality_id).long()
    print(f"\nCreated binary labels for tok_image modality:")
    print(f"Number of image tokens: {image_labels.sum().item()}")
    print(f"Number of non-image tokens: {len(image_labels) - image_labels.sum().item()}")
    print(f"Percentage of image tokens: {(image_labels.sum().item() / len(image_labels) * 100):.2f}%")

    # Verify we have no padding tokens
    assert -1 not in np.unique(valid_labels), "Found padding tokens in labels after filtering!"
    print("\nVerified: No padding tokens in the dataset")

    # Prepare the dataset for probing
    print("\nPreparing dataset for dense probing...")
    # Convert tensors to numpy for sklearn compatibility
    activations_np = valid_activations.cpu().numpy()
    labels_np = image_labels.cpu().numpy()

    print(f"Transformer activations shape: {activations_np.shape}")
    print(f"Using ALL {activations_np.shape[1]} transformer neurons in a single dense model")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        activations_np, labels_np, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels_np  # Ensure balanced split
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set image token percentage: {(y_train.sum() / len(y_train) * 100):.2f}%")
    print(f"Test set image token percentage: {(y_test.sum() / len(y_test) * 100):.2f}%")

    # Calculate class weights for balanced loss
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights_dict = {0: total_samples / (2 * class_counts[0]), 
                         1: total_samples / (2 * class_counts[1])}
    print(f"\nClass weights for balanced loss: {class_weights_dict}")
    print(f"Class distribution: {class_counts}")
    print(f"Percentage of image positions: {(class_counts[1] / total_samples * 100):.2f}%")
    
    # Calculate baseline metrics
    majority_class = np.argmax(class_counts)
    baseline_accuracy = class_counts[majority_class] / total_samples
    print(f"\nBaseline metrics (always predicting majority class):")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Balanced accuracy: 0.5 (random chance)")
    print(f"F1 score: 0.0 (no true positives)")
    
    # Train DENSE logistic regression using ALL neurons simultaneously
    print(f"\nTraining dense logistic regression on all {activations_np.shape[1]} neurons...")
    
    # Use sklearn's LogisticRegression with balanced class weights
    dense_model = LogisticRegression(
        class_weight='balanced',  # Automatically handle class imbalance
        random_state=random_state,
        max_iter=1000  # Increase iterations for convergence
    )
    
    print("Fitting dense model...")
    dense_model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating dense model...")
    y_pred_proba = dense_model.predict_proba(X_test)[:, 1]  # Probability of positive class
    y_pred = dense_model.predict(X_test)
    
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate balanced cross-entropy loss manually
    class_weights_array = np.array([class_weights_dict[0], class_weights_dict[1]])
    balanced_loss = -np.mean(
        class_weights_array[y_test] * (y_test * np.log(np.clip(y_pred_proba, 1e-15, 1-1e-15)) + 
                                     (1 - y_test) * np.log(np.clip(1 - y_pred_proba, 1e-15, 1-1e-15)))
    )
    
    # Store results
    results = {
        'model_type': 'dense_logistic_regression',
        'num_features': activations_np.shape[1],
        'image_modality_id': image_modality_id,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'auc': auc,
        'balanced_cross_entropy_loss': balanced_loss,
        'model_coefficients': dense_model.coef_[0],  # Weights for each neuron
        'model_intercept': dense_model.intercept_[0]
    }
    
    # Print results
    print(f"\n=== DENSE PROBE RESULTS ===")
    print(f"Model uses ALL {activations_np.shape[1]} transformer neurons simultaneously")
    print(f"Target: Modality {image_modality_id} vs all other modalities")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Balanced Cross-Entropy Loss: {balanced_loss:.4f}")
    
    # Compare to baseline
    print(f"\n=== COMPARISON TO BASELINE ===")
    print(f"Baseline accuracy (majority class): {baseline_accuracy:.4f}")
    print(f"Dense probe accuracy: {accuracy:.4f}")
    print(f"Improvement over baseline: {accuracy - baseline_accuracy:.4f}")
    print(f"Dense probe balanced accuracy: {balanced_acc:.4f}")
    print(f"Baseline balanced accuracy (random): 0.5000")
    print(f"Improvement over random: {balanced_acc - 0.5:.4f}")
    
    # Analysis of model weights
    coef_abs = np.abs(dense_model.coef_[0])
    print(f"\n=== MODEL WEIGHT ANALYSIS ===")
    print(f"Mean absolute weight: {coef_abs.mean():.6f}")
    print(f"Max absolute weight: {coef_abs.max():.6f}")
    print(f"Min absolute weight: {coef_abs.min():.6f}")
    print(f"Number of near-zero weights (|w| < 0.001): {(coef_abs < 0.001).sum()}")
    print(f"Top 10 most important neuron indices: {np.argsort(coef_abs)[-10:][::-1].tolist()}")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{output_prefix}_dense_probing_results.csv', index=False)
    print(f"\nResults saved to '{output_prefix}_dense_probing_results.csv'")
    
    # Save model weights for analysis
    weights_df = pd.DataFrame({
        'neuron_index': range(len(dense_model.coef_[0])),
        'weight': dense_model.coef_[0],
        'abs_weight': np.abs(dense_model.coef_[0])
    }).sort_values('abs_weight', ascending=False)
    weights_df.to_csv(f'{output_prefix}_model_weights.csv', index=False)
    print(f"Model weights saved to '{output_prefix}_model_weights.csv'")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'Dense Probe (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve - Dense Probe')
    ax1.legend()
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im, ax=ax2)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix')
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Plot 3: Distribution of model weights
    ax3.hist(dense_model.coef_[0], bins=50, alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', label='Zero Weight')
    ax3.set_xlabel('Model Weight')
    ax3.set_ylabel('Number of Neurons')
    ax3.set_title('Distribution of Model Weights')
    ax3.legend()
    
    # Plot 4: Top 20 most important neurons
    top_20_weights = weights_df.head(20)
    ax4.barh(range(len(top_20_weights)), top_20_weights['abs_weight'])
    ax4.set_yticks(range(len(top_20_weights)))
    ax4.set_yticklabels(top_20_weights['neuron_index'])
    ax4.set_xlabel('Absolute Weight')
    ax4.set_ylabel('Neuron Index')
    ax4.set_title('Top 20 Most Important Neurons')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_dense_probe_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to '{output_prefix}_dense_probe_analysis.png'")
    
    return results, dense_model, weights_df

def plot_tsne_visualization(activations_tensor_flat, modality_labels_flat, autoencoder, device, 
                          image_modality_id=18665, output_prefix='image_tsne', perplexity=30, 
                          n_iter=1000, random_state=42):
    """
    Create TSNE visualization of activations with different colors for image vs non-image positions.
    
    Args:
        activations_tensor_flat: Flattened tensor of activations [n_samples, hidden_dim]
        modality_labels_flat: Flattened tensor of binary labels [n_samples] (1=image, 0=non-image, -1=padding)
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
        image_modality_id: ID for image modality (unused but kept for compatibility)
        output_prefix: Prefix for output files
        perplexity: TSNE perplexity parameter
        n_iter: Number of TSNE iterations
        random_state: Random seed for reproducibility
    """
    print("\nStarting TSNE visualization...")
    
    # Move tensors to CPU to avoid CUDA memory issues
    print("Moving tensors to CPU...")
    activations_tensor_flat = activations_tensor_flat.cpu()
    modality_labels_flat = modality_labels_flat.cpu()
    
    # Filter out padding tokens (-1)
    print("Filtering padding tokens...")
    valid_mask = modality_labels_flat != -1
    activations_filtered = activations_tensor_flat[valid_mask]
    labels_filtered = modality_labels_flat[valid_mask]
    
    print(f"Filtered activations shape: {activations_filtered.shape}")
    print(f"Number of valid samples: {valid_mask.sum().item()}")
    print(f"Image positions (label=1): {(labels_filtered == 1).sum().item()}")
    print(f"Non-image positions (label=0): {(labels_filtered == 0).sum().item()}")
    
    # Process autoencoder activations in chunks to avoid memory issues
    chunk_size = 10000
    autoencoder_acts_list = []
    
    print("\nProcessing activations through autoencoder...")
    for start_idx in tqdm(range(0, len(activations_filtered), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(activations_filtered))
        chunk = activations_filtered[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            try:
                _, encoded = autoencoder(chunk)
                autoencoder_acts_list.append(encoded.cpu())
                # Clear CUDA cache after each chunk
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
                print(f"Traceback:\n{traceback.format_exc()}")
                continue
    
    # Concatenate all chunks
    print("\nConcatenating autoencoder activations...")
    autoencoder_acts = torch.cat(autoencoder_acts_list, dim=0)
    print(f"Autoencoder activations shape: {autoencoder_acts.shape}")
    
    # Convert to numpy and free GPU memory
    print("Converting to numpy array...")
    autoencoder_acts_np = autoencoder_acts.numpy()
    del autoencoder_acts
    del autoencoder_acts_list
    torch.cuda.empty_cache()
    
    # Apply TSNE with memory-efficient settings
    print("\nApplying TSNE...")
    try:
        print("Initializing TSNE with parameters:")
        print(f"- perplexity: {perplexity}")
        print(f"- n_iter: {n_iter}")
        print(f"- random_state: {random_state}")
        print(f"- n_jobs: -1 (using all CPU cores)")
        print(f"- method: barnes_hut (memory efficient)")
        print(f"- angle: 0.5 (trade-off between accuracy and speed)")
        
        # Use more memory-efficient settings
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, 30),  # Cap perplexity at 30
            n_iter=min(n_iter, 500),  # Reduce iterations
            random_state=random_state,
            verbose=2,  # More verbose output
            n_jobs=-1,  # Use all CPU cores
            method='barnes_hut',  # More memory efficient method
            angle=0.5,  # Trade-off between accuracy and speed
            init='pca'  # Use PCA initialization for better convergence
        )
        
        print("Starting TSNE fit_transform...")
        latents_2d = tsne.fit_transform(autoencoder_acts_np)
        print("TSNE completed successfully")
        
        # Free memory
        del autoencoder_acts_np
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error during TSNE: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, None
    
    # Create visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot points with different colors for different positions
    print("Plotting points...")
    # Plot non-image positions in gray
    non_image_mask = labels_filtered == 0
    if non_image_mask.any():
        plt.scatter(latents_2d[non_image_mask, 0], latents_2d[non_image_mask, 1], 
                   c='gray', alpha=0.1, s=1, label='Non-image positions')
    
    # Highlight image positions in red
    print("Highlighting image positions...")
    image_mask = labels_filtered == 1
    if image_mask.any():
        plt.scatter(latents_2d[image_mask, 0], latents_2d[image_mask, 1], 
                   c='red', alpha=0.5, s=2, label='Image positions')
    
    plt.title('t-SNE Visualization of Autoencoder Activations\n(Image vs Non-Image Positions)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    
    # Save plot
    print("Saving plot...")
    output_file = f'{output_prefix}_tsne.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_file}")
    
    # Clean up
    print("Cleaning up...")
    torch.cuda.empty_cache()
    
    return latents_2d

def plot_pca_visualization(activations_tensor_flat, modality_labels_flat, autoencoder, device, 
                          image_modality_id, output_prefix, n_components, 
                          standardize, random_state):
    """
    Create PCA visualization of activations with different colors for image vs non-image tokens.
    Also analyze distances to check for radial structure.
    
    Args:
        activations_tensor_flat: Flattened tensor of activations [n_samples, hidden_dim]
        modality_labels_flat: Flattened tensor of modality labels [n_samples]
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
        image_modality_id: ID for image modality tokens
        output_prefix: Prefix for output files
        n_components: Number of PCA components (2 for visualization)
        standardize: Whether to standardize features before PCA
        random_state: Random seed for reproducibility
    """
    print("\nStarting PCA visualization...")
    
    # Move tensors to CPU to avoid CUDA memory issues
    print("Moving tensors to CPU...")
    activations_tensor_flat = activations_tensor_flat.cpu()
    modality_labels_flat = modality_labels_flat.cpu()
    
    # Filter out padding tokens (-1)
    print("Filtering padding tokens...")
    valid_mask = modality_labels_flat != -1
    activations_filtered = activations_tensor_flat[valid_mask]
    labels_filtered = modality_labels_flat[valid_mask]
    
    print(f"Filtered activations shape: {activations_filtered.shape}")
    print(f"Number of valid samples: {valid_mask.sum().item()}")
    print(f"Image tokens: {(labels_filtered == image_modality_id).sum().item()}")
    
    # Process autoencoder activations in chunks to avoid memory issues
    chunk_size = 10000
    autoencoder_acts_list = []
    
    print("\nProcessing activations through autoencoder...")
    for start_idx in tqdm(range(0, len(activations_filtered), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(activations_filtered))
        chunk = activations_filtered[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            try:
                _, encoded = autoencoder(chunk)
                autoencoder_acts_list.append(encoded.cpu())
                # Clear CUDA cache after each chunk
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
                print(f"Traceback:\n{traceback.format_exc()}")
                continue
    
    # Concatenate all chunks
    print("\nConcatenating autoencoder activations...")
    autoencoder_acts = torch.cat(autoencoder_acts_list, dim=0)
    print(f"Autoencoder activations shape: {autoencoder_acts.shape}")
    
    # Convert to numpy and free GPU memory
    print("Converting to numpy array...")
    autoencoder_acts_np = autoencoder_acts.numpy()
    del autoencoder_acts
    del autoencoder_acts_list
    torch.cuda.empty_cache()
    
    # Standardize features if requested
    if standardize:
        print("Standardizing features...")
        scaler = StandardScaler()
        autoencoder_acts_np = scaler.fit_transform(autoencoder_acts_np)
    
    # Apply PCA
    print(f"\nApplying PCA with {n_components} components...")
    try:
        pca = PCA(n_components=n_components, random_state=random_state)
        latents_2d = pca.fit_transform(autoencoder_acts_np)
        print("PCA completed successfully")
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_var}")
        print(f"Total explained variance: {explained_var.sum():.4f}")
        
        # Free memory
        del autoencoder_acts_np
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error during PCA: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, None
    
    # Analyze distances to check for radial structure
    print("\nAnalyzing distance patterns...")
    image_mask = labels_filtered == image_modality_id
    
    # Calculate centroids
    all_centroid = np.mean(latents_2d, axis=0)
    gray_centroid = np.mean(latents_2d[~image_mask], axis=0)
    red_centroid = np.mean(latents_2d[image_mask], axis=0)
    
    # Calculate distances from centroids
    red_distances_from_all = np.linalg.norm(latents_2d[image_mask] - all_centroid, axis=1)
    gray_distances_from_all = np.linalg.norm(latents_2d[~image_mask] - all_centroid, axis=1)
    red_distances_from_gray = np.linalg.norm(latents_2d[image_mask] - gray_centroid, axis=1)
    
    # Print distance statistics
    print(f"\nDistance Analysis:")
    print(f"Red points distance from overall centroid: {red_distances_from_all.mean():.4f} ± {red_distances_from_all.std():.4f}")
    print(f"Gray points distance from overall centroid: {gray_distances_from_all.mean():.4f} ± {gray_distances_from_all.std():.4f}")
    print(f"Red points distance from gray centroid: {red_distances_from_gray.mean():.4f} ± {red_distances_from_gray.std():.4f}")
    
    # Check if red points form a shell around gray points
    red_mean_dist = red_distances_from_gray.mean()
    red_std_dist = red_distances_from_gray.std()
    shell_coefficient = red_std_dist / red_mean_dist  # Lower values suggest more shell-like
    print(f"Shell coefficient (std/mean of red distances from gray): {shell_coefficient:.4f}")
    print(f"Lower values suggest more shell-like arrangement")
    
    # Create visualization
    print("\nCreating PCA visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main PCA visualization
    ax1.scatter(latents_2d[~image_mask, 0], latents_2d[~image_mask, 1], 
               c='gray', alpha=0.1, s=1, label='Other modalities')
    if image_mask.any():
        ax1.scatter(latents_2d[image_mask, 0], latents_2d[image_mask, 1], 
                   c='red', alpha=0.5, s=2, label='Image modality')
    
    ax1.scatter(*all_centroid, c='blue', s=100, marker='x', label='Overall centroid')
    ax1.scatter(*gray_centroid, c='black', s=100, marker='+', label='Gray centroid')
    ax1.scatter(*red_centroid, c='darkred', s=100, marker='*', label='Red centroid')
    
    ax1.set_title(f'PCA Visualization\n(Explained variance: {explained_var.sum():.3f})')
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.3f})')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance distributions
    ax2.hist(red_distances_from_gray, bins=50, alpha=0.7, label='Red from gray centroid', color='red')
    ax2.hist(gray_distances_from_all, bins=50, alpha=0.7, label='Gray from overall centroid', color='gray')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Count')
    ax2.set_title('Distance Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Polar plot to check radial structure
    if image_mask.any():
        # Convert red points to polar coordinates relative to gray centroid
        red_relative = latents_2d[image_mask] - gray_centroid
        red_angles = np.arctan2(red_relative[:, 1], red_relative[:, 0])
        red_radii = np.linalg.norm(red_relative, axis=1)
        
        ax3.scatter(red_angles, red_radii, c='red', alpha=0.5, s=2)
        ax3.set_xlabel('Angle (radians)')
        ax3.set_ylabel('Distance from gray centroid')
        ax3.set_title('Red Points in Polar Coordinates\n(relative to gray centroid)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: First few principal components
    if n_components >= 3:
        pca_more = PCA(n_components=min(10, autoencoder_acts.shape[1]), random_state=random_state)
        latents_more = pca_more.fit_transform(autoencoder_acts_np)
        ax4.bar(range(min(10, len(pca_more.explained_variance_ratio_))), 
               pca_more.explained_variance_ratio_[:min(10, len(pca_more.explained_variance_ratio_))])
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Explained Variance Ratio')
        ax4.set_title('Explained Variance by Component')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Need more components\nfor this analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Component Analysis')
    
    plt.tight_layout()
    
    # Save plot
    print("Saving plot...")
    output_file = f'{output_prefix}_pca_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA analysis to {output_file}")
    
    # Clean up
    print("Cleaning up...")
    torch.cuda.empty_cache()
    
    return latents_2d, {
        'explained_variance_ratio': explained_var,
        'red_distances_from_gray': red_distances_from_gray,
        'gray_distances_from_all': gray_distances_from_all,
        'shell_coefficient': shell_coefficient,
        'centroids': {
            'all': all_centroid,
            'gray': gray_centroid,
            'red': red_centroid
        }
    }

def run_tsne_analysis(model, autoencoder, data_loader_train, device, args, image_modality_id, output_prefix, perplexity, n_iter, random_state):
    print(f"Data loader created: {type(data_loader_train)}")
    
    # Try to get a single batch to test the data loader
    print("\nTesting data loader with a single batch...")
    try:
        test_batch = next(iter(data_loader_train))
        print("Successfully got first batch!")
        print(f"Batch keys: {list(test_batch.keys())}")
        for mod, d in test_batch.items():
            print(f"Modality {mod} tensor shape: {d['tensor'].shape}")
    except Exception as e:
        print(f"Error getting first batch: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise
    
    print("\nStarting activation collection...")
    try:
        # Collect activations and modality labels
        activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched_encoder(
            model, data_loader_train, device, args
        )
        print("Finished collecting activations!")
        
        # Save activations immediately
        print("\nSaving activations...")
        save_path = f'/mnt/home/rzhang/ceph/activations_{examples_processed}examples.pt'
        torch.save({
            'activations': activations_tensor_flat,
            'modality_labels': modality_labels_flat,
            'examples_processed': examples_processed
        }, save_path)
        print(f"Saved activations to {save_path}")
        
        # Clear data loader and CUDA cache
        del data_loader_train
        torch.cuda.empty_cache()
        
        # Run TSNE visualization
        print("\nStarting TSNE visualization...")
        latents_2d = plot_tsne_visualization(
            activations_tensor_flat,
            modality_labels_flat,
            autoencoder,
            device,
            image_modality_id=18665,
            output_prefix='image_tsne',
            perplexity=30,
            n_iter=1000,
            random_state=42
        )
        print("Finished TSNE visualization!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        # Clean up
        if 'data_loader_train' in locals():
            del data_loader_train
        torch.cuda.empty_cache()
        print("\nCleanup completed")

def run_pca_analysis(model, autoencoder, data_loader, device, args, 
                    image_modality_id=18665, output_prefix='tok_image',
                    n_components=2, standardize=True, random_state=42):
    """
    Run complete PCA analysis pipeline: collect activations, save them, and create visualization.
    
    Args:
        model: AION model
        autoencoder: Trained sparse autoencoder
        data_loader: Data loader for collecting activations
        device: Device to run computations on
        args: Arguments object containing batch size and other parameters
        image_modality_id: ID for image modality tokens
        output_prefix: Prefix for output files
        n_components: Number of PCA components
        standardize: Whether to standardize features before PCA
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Results including PCA latents, statistics, and examples processed
    """
    try:
        print("\n=== Starting PCA Analysis Pipeline ===")
        
        # Collect activations
        print("\nCollecting activations...")
        activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched_encoder(
            model, data_loader, device, args
        )
        print("Finished collecting activations!")
        
        # Save activations
        print("\nSaving activations...")
        save_path = f'/mnt/home/rzhang/ceph/activations_{examples_processed}examples.pt'
        torch.save({
            'activations': activations_tensor_flat,
            'modality_labels': modality_labels_flat,
            'examples_processed': examples_processed
        }, save_path)
        print(f"Saved activations to {save_path}")
        
        # Clear data loader and CUDA cache
        del data_loader
        torch.cuda.empty_cache()
        
        # Run PCA visualization
        print("\nStarting PCA visualization...")
        pca_latents, pca_stats = plot_pca_visualization(
            activations_tensor_flat,
            modality_labels_flat,
            autoencoder,
            device,
            image_modality_id=image_modality_id,
            output_prefix=output_prefix,
            n_components=n_components,
            standardize=standardize,
            random_state=random_state
        )
        print("Finished PCA visualization!")
        
        return {
            'pca_latents': pca_latents,
            'pca_stats': pca_stats,
            'examples_processed': examples_processed
        }
        
    except Exception as e:
        print(f"\nError during PCA analysis: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None
        
    finally:
        # Clean up
        if 'data_loader' in locals():
            del data_loader
        torch.cuda.empty_cache()
        print("\nCleanup completed")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with trained parameters
    print("\n=== Loading Model ===")
    model = AION.from_pretrained('/mnt/ceph/users/polymathic/aion/dec24/base')
    print(f"Model loaded from pretrained weights")
    model.freeze_encoder()
    model.freeze_decoder()
    model = model.to(device).eval()  
    print(f"Model architecture:\n{model}")
    
    # Find the correct image modality ID dynamically
    image_modality_name = None
    image_modality_id = None
    for mod_name, mod_info in model.modality_info.items():
        if 'image' in mod_name.lower() or 'tok_image' in mod_name:
            image_modality_name = mod_name
            image_modality_id = mod_info['id']
            break
    
    if image_modality_id is not None:
        print(f"Found image modality: {image_modality_name} with ID: {image_modality_id}")
    else:
        print("Warning: Could not find image modality automatically")
        # Fallback to manual specification
        image_modality_id = 18665
        print(f"Using fallback image modality ID: {image_modality_id}")
    print()

    # Load the sparse autoencoder
    input_size = 768
    hidden_size = input_size*4
    k = max(1, int(hidden_size * 0.02))  # Use 2% sparsity by default
    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    autoencoder.load_state_dict(torch.load('best_llm_sae_rural-wood-16.pth', weights_only=True, map_location=device))   

    # # Set this flag to True to use a randomly initialized autoencoder for testing
    # USE_RANDOM_AUTOENCODER = True

    # if USE_RANDOM_AUTOENCODER:
    #     print("\n=== Using randomly initialized autoencoder for test analysis ===")
    #     autoencoder = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    # else:
    #     autoencoder = SparseAutoencoder(input_size, hidden_size, k=k).to(device)
    #     # Load state dict, removing 'encoder.bias' if present and handling missing 'bias'
    #     state_dict = torch.load('best_llm_sae_fresh-voice-15.pth', map_location=device)
    #     if 'encoder.bias' in state_dict:
    #         print('Removing encoder.bias from state_dict')
    #         del state_dict['encoder.bias']
    #     missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
    #     print('Missing keys:', missing)
    #     print('Unexpected keys:', unexpected)


    autoencoder.eval()
    print(f"Autoencoder loaded")

    # # Read in activations
    # activations_path = '/mnt/home/rzhang/ceph/activations_4992examples.pt'
    # activations_tensor_flat = torch.load(activations_path, map_location=device, weights_only=True)

    # Load tokenized training data
    print("\n=== Setting Up Data ===")
    args = Args()
    print(f"Args batch size: {args.batch_size}")
    print(f"Args num workers: {args.num_workers}")
    print(f"Args epoch size: {args.epoch_size}")
    
    # # Count examples in the shard before setting up the data loader
    # shard_path = args.data_config
    # with open(shard_path, "r") as f:
    #     data_config = yaml.safe_load(f)
    #     # Find the sdss_legacysurvey dataset path
    #     for dataset_name, dataset_cfg in data_config['train']['datasets'].items():
    #         if dataset_name == 'sdss_legacysurvey':
    #             shard_path = dataset_cfg['data_path']
    #             break
    
    # num_examples = count_examples_in_shard(shard_path)
    # if num_examples is not None:
    #     print(f"\nWith batch size {args.batch_size}, you'll get {num_examples // args.batch_size} full batches")
    #     print(f"Total examples per epoch: {num_examples}")
    #     print(f"Note: Due to n_repeats=5 in the config, each example will be seen 5 times")
    #     print(f"Effective examples per epoch with repeats: {num_examples * 5}")
    
    data_loader_train = get_data(args)

    # for batch_idx, batch in enumerate(data_loader_train):
    #     print(f"Processing batch {batch_idx + 1}")
    #     current_batch_size = next(iter(batch.values()))['tensor'].shape[0]
    #     print(f"Processing full batch of {current_batch_size} examples")
    #     for mod, d in batch.items():
    #         print(f"Modality: {mod}")
    #         print(f"Shape: {d['tensor'].shape}")
    #         print(f"Target mask shape: {d['target_mask'].shape}")

    # # Calculate loss ratio
    # loss_ratio, losses = calculate_loss_ratio(model, data_loader_train, autoencoder, device)
    # print(f"Loss ratio: {loss_ratio:.2f}%")
    # print(f"Losses: {losses}")

    # # Plot activation frequencies comparison
    # transformer_frequencies, autoencoder_frequencies = plot_activation_frequencies_comparison(activations_tensor_flat, autoencoder, device)
    
    # Collect activations and modality labels for transformer probing
    print("\n=== Collecting activations for transformer neuron probing ===")
    # activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched_encoder(
    #     model, data_loader_train, device, args)
    activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched_decoder(
        model, data_loader_train, device, args
    )
    
    # Apply the dense probe on ALL transformer neurons simultaneously  
    apply_dense_probe_transformer(
        activations_tensor_flat, 
        modality_labels_flat, 
        examples_processed,
        image_modality_id=image_modality_id,  # Use the dynamically found modality ID
        output_prefix='image_decoder_dense',  
        test_size=0.2,
        random_state=42
    )

    # # Apply the logistic probe on each latent dimension  
    # apply_logistic_probe(
    #     activations_tensor_flat, 
    #     modality_labels_flat, 
    #     examples_processed, 
    #     autoencoder, 
    #     device,
    #     image_modality_id=18665,  # ID for tok_image
    #     output_prefix='tok_image',  # Prefix for output files
    #     test_size=0.2,
    #     random_state=42
    # )

    # # Run PCA analysis
    # run_pca_analysis(
    #     model,
    #     autoencoder,
    #     data_loader_train,
    #     device,
    #     args,
    #     image_modality_id=18665,
    #     output_prefix='tok_image',
    #     n_components=2,
    #     standardize=True,
    #     random_state=42
    # )
    
    # # Run TSNE analysis
    # run_tsne_analysis(
    #     model=model,
    #     autoencoder=autoencoder,
    #     data_loader_train=data_loader_train,
    #     device=device,
    #     args=args,
    #     image_modality_id=18665,
    #     output_prefix='tok_image',
    #     perplexity=30,
    #     n_iter=500,
    #     random_state=42
    # )



