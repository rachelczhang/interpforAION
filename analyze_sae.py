import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sparse_autoencoder import SparseAutoencoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import sys
from save_activations import get_data
import torch.nn.functional as F
from loss_ratio import calculate_loss_ratio
import webdataset as wds
import tarfile
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, balanced_accuracy_score, f1_score
from scipy.optimize import minimize
import pandas as pd
from sklearn.manifold import TSNE
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
        self.num_input_tokens = 128
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

def collect_activations_batched(model, data_loader, device, num_target_tokens):
    print("\n=== Starting Activation Collection ===")
    print(f"Using num_target_tokens: {num_target_tokens}")
    activations = []
    modality_labels = []  # List to store modality labels
    examples_processed = 0
    
    def hook_fn(module, input, output):
        print(f'\nHook received output shape: {output.shape}')
        # Save activations
        activations.append(output.clone().detach().cpu())
    
    # register hook on the spatial attention layer
    ninth_decoder_block = model.decoder[8]
    hook = ninth_decoder_block.register_forward_hook(hook_fn)
    print(f"Registered hook on decoder block 8")

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
            
            # Prepare input dictionary and target mask
            input_dict = {}
            target_mask = {}
            
            # Track which modality each token comes from
            batch_modality_labels = []
            total_tokens = 0
            
            for mod, d in processed_data.items():
                if mod in model.encoder_embeddings:
                    input_dict[mod] = d['tensor']

                if mod in model.decoder_embeddings:
                    target_mask[mod] = d['target_mask']
                    # Get the modality ID from the model's modality info
                    mod_id = model.modality_info[mod]["id"]
                    # Get the number of tokens for this modality
                    num_tokens = d['tensor'].shape[1]
                    # Only create labels for tokens up to num_target_tokens
                    num_tokens = min(num_tokens, num_target_tokens)
                    # Create labels for these tokens
                    batch_modality_labels.append(torch.full((current_batch_size, num_tokens), mod_id, dtype=torch.long, device='cpu'))
                    total_tokens += num_tokens
            
            # Concatenate modality labels for this batch
            if batch_modality_labels:
                batch_modality_tensor = torch.cat(batch_modality_labels, dim=1)
                # Ensure we only have num_target_tokens tokens
                if batch_modality_tensor.shape[1] > num_target_tokens:
                    batch_modality_tensor = batch_modality_tensor[:, :num_target_tokens]
                # Pad to match the model's sequence length (256)
                if batch_modality_tensor.shape[1] < 256:
                    padding = torch.full((current_batch_size, 256 - batch_modality_tensor.shape[1]), 
                                       -1, dtype=torch.long, device='cpu')
                    batch_modality_tensor = torch.cat([batch_modality_tensor, padding], dim=1)
                modality_labels.append(batch_modality_tensor)
            else:
                print("Warning: No modality labels created for this batch")
                modality_labels.append(torch.full((current_batch_size, 256), -1, dtype=torch.long, device='cpu'))

            # Run the batch through the model
            print("\nRunning forward pass...")
            output = model(input_dict, target_mask)

            torch.cuda.empty_cache()

            examples_processed += current_batch_size
            print(f"Total examples processed: {examples_processed}")
            
    # Remove the hook
    hook.remove()
    print("\nHook removed")
    
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
    
    # Verify shapes match before flattening
    assert activations_tensor.shape[:2] == modality_labels_tensor.shape, \
        f"Shape mismatch: activations {activations_tensor.shape[:2]} vs labels {modality_labels_tensor.shape}"
    
    # Flatten activations
    print("\nFlattening activations...")
    # Get original shape dimensions
    sample_size, seq_length, embed_dim = activations_tensor.shape
    # Reshape to [sample_size*seq_length, embed_dim] instead of [sample_size, seq_length*embed_dim]
    activations_tensor_flat = activations_tensor.reshape(sample_size * seq_length, embed_dim)
    print(f"Original activations shape: {activations_tensor.shape} -> [sample_size, seq_length, embed_dim]")
    print(f"Flattened activations shape: {activations_tensor_flat.shape} -> [sample_size*seq_length, embed_dim]")
    print(f"Flattened activations dtype: {activations_tensor_flat.dtype}")
    print(f"Flattened activations device: {activations_tensor_flat.device}")
    
    # Flatten modality labels to match the flattened activations
    modality_labels_flat = modality_labels_tensor.reshape(-1)
    print(f"Flattened modality labels shape: {modality_labels_flat.shape}")
    print(f"Flattened modality labels dtype: {modality_labels_flat.dtype}")
    print(f"Flattened modality labels device: {modality_labels_flat.device}")
    
    # Verify shapes match after flattening
    assert len(activations_tensor_flat) == len(modality_labels_flat), \
        f"Length mismatch after flattening: activations {len(activations_tensor_flat)} vs labels {len(modality_labels_flat)}"
    
    # Print unique modality values for verification
    unique_modalities = torch.unique(modality_labels_flat)
    print(f"Unique modality values in labels: {unique_modalities.tolist()}")
    
    # Print modality distribution
    for mod_id in unique_modalities:
        if mod_id != -1:  # Skip padding tokens
            count = (modality_labels_flat == mod_id).sum().item()
            percentage = (count / len(modality_labels_flat)) * 100
            # Try to get modality name from model's modality_info
            mod_name = next((mod for mod, info in model.modality_info.items() if info["id"] == mod_id), f"Unknown-{mod_id}")
            print(f"Modality {mod_name} (ID: {mod_id}): {count} tokens ({percentage:.2f}%)")
    
    # Print padding token distribution
    padding_count = (modality_labels_flat == -1).sum().item()
    padding_percentage = (padding_count / len(modality_labels_flat)) * 100
    print(f"Padding tokens (-1): {padding_count} tokens ({padding_percentage:.2f}%)")
    
    return activations_tensor_flat, modality_labels_flat, examples_processed

class NewtonRaphsonLogisticRegression:
    """1D logistic regression using Newton-Raphson optimization"""
    def __init__(self, max_iter=100, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # Clip to avoid overflow
    
    def _loss(self, params, X, y):
        """Binary cross-entropy loss"""
        w, b = params
        z = w * X + b
        y_pred = self._sigmoid(z)
        # Avoid log(0) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def _gradient(self, params, X, y):
        """Gradient of loss with respect to parameters"""
        w, b = params
        z = w * X + b
        y_pred = self._sigmoid(z)
        error = y_pred - y
        grad_w = np.mean(error * X)
        grad_b = np.mean(error)
        return np.array([grad_w, grad_b])
    
    def _hessian(self, params, X, y):
        """Hessian matrix (second derivatives)"""
        w, b = params
        z = w * X + b
        y_pred = self._sigmoid(z)
        diag = y_pred * (1 - y_pred)
        H_ww = np.mean(diag * X * X)
        H_wb = np.mean(diag * X)
        H_bb = np.mean(diag)
        return np.array([[H_ww, H_wb], [H_wb, H_bb]])
    
    def fit(self, X, y):
        """Fit the model using Newton-Raphson method"""
        # Initialize parameters
        params = np.zeros(2)  # [w, b]
        
        for _ in range(self.max_iter):
            # Calculate gradient and Hessian
            gradient = self._gradient(params, X, y)
            hessian = self._hessian(params, X, y)
            
            # Newton update: params = params - H^(-1) * gradient
            try:
                update = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                # If Hessian is singular, add small value to diagonal
                hessian = hessian + np.eye(2) * 1e-6
                update = np.linalg.solve(hessian, gradient)
            
            # Update parameters
            params = params - update
            
            # Check convergence
            if np.linalg.norm(update) < self.tol:
                break
        
        self.w, self.b = params
        return self
    
    def predict_proba(self, X):
        """Predict probability of class 1"""
        z = self.w * X + self.b
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)

def apply_logistic_probe(
    activations_tensor_flat, 
    modality_labels_flat, 
    examples_processed, 
    autoencoder, 
    device,
    image_modality_id=18665,  # ID for tok_image
    output_prefix='tok_image',  # Prefix for output files
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

    # Filter out padding tokens (-1)
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
    print(f"Training set image token percentage: {(y_train.sum() / len(y_train) * 100):.2f}%")
    print(f"Test set image token percentage: {(y_test.sum() / len(y_test) * 100):.2f}%")

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
    # num_latents = train_latents.shape[1]
    num_latents = 10
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
    print(top_10_df[['latent_index', 'balanced_accuracy', 'f1_score', 'balanced_cross_entropy_loss']].to_string())
    
    # Print summary statistics of all latents
    print("\nSummary statistics of all latents:")
    print(f"Best balanced accuracy: {results_df['balanced_accuracy'].max():.4f}")
    print(f"Worst balanced accuracy: {results_df['balanced_accuracy'].min():.4f}")
    print(f"Mean balanced accuracy: {results_df['balanced_accuracy'].mean():.4f}")
    print(f"Median balanced accuracy: {results_df['balanced_accuracy'].median():.4f}")
    print(f"Number of latents better than random (balanced accuracy > 0.5): {(results_df['balanced_accuracy'] > 0.5).sum()}")
    
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

def plot_tsne_visualization(activations_tensor_flat, modality_labels_flat, autoencoder, device, 
                          image_modality_id=18665, output_prefix='tok_image', perplexity=30, 
                          n_iter=1000, random_state=42):
    """
    Create TSNE visualization of activations with different colors for image vs non-image tokens.
    
    Args:
        activations_tensor_flat: Flattened tensor of activations [n_samples, hidden_dim]
        modality_labels_flat: Flattened tensor of modality labels [n_samples]
        autoencoder: Trained sparse autoencoder model
        device: Device to run computations on
        image_modality_id: ID for image modality tokens
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
    
    # Take a subset of the data for TSNE (20% of the data)
    print("\nTaking subset of data for TSNE...")
    subset_size = len(activations_filtered) // 5
    indices = torch.randperm(len(activations_filtered))[:subset_size]
    activations_subset = activations_filtered[indices]
    labels_subset = labels_filtered[indices]
    
    print(f"Subset size: {subset_size} samples")
    print(f"Image tokens in subset: {(labels_subset == image_modality_id).sum().item()}")
    
    # Process autoencoder activations in chunks to avoid memory issues
    chunk_size = 10000
    autoencoder_acts_list = []
    
    print("\nProcessing activations through autoencoder...")
    for start_idx in tqdm(range(0, len(activations_subset), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(activations_subset))
        chunk = activations_subset[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            try:
                _, encoded = autoencoder(chunk)
                autoencoder_acts_list.append(encoded.cpu())
                # Clear CUDA cache after each chunk
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
                import traceback
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
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, None
    
    # Create visualization
    print("\nCreating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    print("Plotting points...")
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='gray', alpha=0.1, s=1, label='Other modalities')
    
    # Highlight image modality
    print("Highlighting image modality...")
    image_mask = labels_subset == image_modality_id
    if image_mask.any():
        plt.scatter(latents_2d[image_mask, 0], latents_2d[image_mask, 1], 
                   c='red', alpha=0.5, s=2, label='Image modality')
    
    plt.title('t-SNE Visualization of Autoencoder Activations (20% Subset)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    
    # Save plot
    print("Saving plot...")
    output_file = f'{output_prefix}_tsne_subset.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_file}")
    
    # Clean up
    print("Cleaning up...")
    torch.cuda.empty_cache()
    
    return latents_2d, None

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

    # Collect activations and modality labels
    activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched(
        model, data_loader_train, device, num_target_tokens=args.num_target_tokens
    )
    # Apply the logistic probe on each latent dimension  
    apply_logistic_probe(
        activations_tensor_flat, 
        modality_labels_flat, 
        examples_processed, 
        autoencoder, 
        device,
        image_modality_id=18665,  # ID for tok_image
        output_prefix='tok_image',  # Prefix for output files
        test_size=0.2,
        random_state=42
    )
    
    # print(f"Data loader created: {type(data_loader_train)}")
    
    # # Try to get a single batch to test the data loader
    # print("\nTesting data loader with a single batch...")
    # try:
    #     test_batch = next(iter(data_loader_train))
    #     print("Successfully got first batch!")
    #     print(f"Batch keys: {list(test_batch.keys())}")
    #     for mod, d in test_batch.items():
    #         print(f"Modality {mod} tensor shape: {d['tensor'].shape}")
    # except Exception as e:
    #     print(f"Error getting first batch: {str(e)}")
    #     import traceback
    #     print(f"Traceback:\n{traceback.format_exc()}")
    #     raise
    
    # print("\nStarting activation collection...")
    # try:
    #     # Collect activations and modality labels
    #     activations_tensor_flat, modality_labels_flat, examples_processed = collect_activations_batched(
    #         model, data_loader_train, device, num_target_tokens=args.num_target_tokens
    #     )
    #     print("Finished collecting activations!")
        
    #     # Save activations immediately
    #     print("\nSaving activations...")
    #     save_path = f'/mnt/home/rzhang/ceph/activations_{examples_processed}examples.pt'
    #     torch.save({
    #         'activations': activations_tensor_flat,
    #         'modality_labels': modality_labels_flat,
    #         'examples_processed': examples_processed
    #     }, save_path)
    #     print(f"Saved activations to {save_path}")
        
    #     # Clear data loader and CUDA cache
    #     del data_loader_train
    #     torch.cuda.empty_cache()
        
    #     # Run TSNE visualization
    #     print("\nStarting TSNE visualization...")
    #     latents_2d, latents_2d_alt = plot_tsne_visualization(
    #         activations_tensor_flat,
    #         modality_labels_flat,
    #         autoencoder,
    #         device,
    #         image_modality_id=18665,
    #         output_prefix='tok_image',
    #         perplexity=30,
    #         n_iter=1000,
    #         random_state=42
    #     )
    #     print("Finished TSNE visualization!")
        
    # except Exception as e:
    #     print(f"Error during processing: {str(e)}")
    #     import traceback
    #     print(f"Traceback:\n{traceback.format_exc()}")
    # finally:
    #     # Clean up
    #     if 'data_loader_train' in locals():
    #         del data_loader_train
    #     torch.cuda.empty_cache()
    #     print("\nCleanup completed")

    # # After collecting activations, add TSNE visualization
    # latents_2d, latents_2d_alt = plot_tsne_visualization(
    #     activations_tensor_flat,
    #     modality_labels_flat,
    #     autoencoder,
    #     device,
    #     image_modality_id=18665,
    #     output_prefix='tok_image',
    #     perplexity=30,
    #     n_iter=1000,
    #     random_state=42
    # )
    

