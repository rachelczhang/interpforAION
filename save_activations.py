import torch
import yaml
import os
from collections import defaultdict
import argparse
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add both directories to sys.path
sys.path.append(os.path.join(current_dir, "AION"))
sys.path.append(os.path.join(current_dir, "4Mclone"))

from aion import AION
from run_training_4m_fsdp_single_epoch import setup_data

class Args:
    """Simple class to hold arguments needed for setup functions."""
    def __init__(self):
        self.data_config = "4Mclone/cfgs/default/mmoma/data/mmu/rusty_legacysurvey_desi_sdss_hsc.yaml"
        self.text_tokenizer_path = "4Mclone/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
        self.input_size = 224
        self.patch_size = 16
        self.num_input_tokens = 128
        self.num_target_tokens = 128
        self.min_input_tokens = None
        self.min_target_tokens = None
        self.batch_size = 128
        self.epoch_size = 5000  
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

def get_data(args):
    # Load data config
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)

    modality_info, data_loader_train, num_train_steps, data_loaders_val, _ = setup_data(args)

    # Print detailed modality information
    print("\n=== Modality Information ===")
    print(f"Available modalities: {list(modality_info.keys())}")
    for mod_name, mod_info in modality_info.items():
        print(f"\nModality: {mod_name}")
        # Print all available keys in the modality info
        print("  Available keys:", list(mod_info.keys()))
        # Print specific information if available
        if 'input_size' in mod_info:
            print(f"  Input size: {mod_info['input_size']}")
        if 'patch_size' in mod_info:
            print(f"  Patch size: {mod_info['patch_size']}")
        if 'num_input_tokens' in mod_info:
            print(f"  Num input tokens: {mod_info['num_input_tokens']}")
        if 'num_target_tokens' in mod_info:
            print(f"  Num target tokens: {mod_info['num_target_tokens']}")
        # Print any other relevant information
        for key, value in mod_info.items():
            if key not in ['input_size', 'patch_size', 'num_input_tokens', 'num_target_tokens']:
                print(f"  {key}: {value}")

    # Print data loader information
    print("\n=== Data Loader Information ===")
    print(f"Number of training steps per epoch: {num_train_steps}")

    # Check the structure of a batch
    print("\n=== Sample Batch Structure ===")
    sample_batch = next(iter(data_loader_train))
    if isinstance(sample_batch, dict):
        print(f"Batch keys: {list(sample_batch.keys())}")
        for mod_name, mod_data in sample_batch.items():
            print(f"\nModality: {mod_name}")
            for key, value in mod_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key} shape: {value.shape}")
                    print(f"  {key} dtype: {value.dtype}")
                    print(f"  {key} device: {value.device}")
                    if key == 'tensor':
                        if value.numel() > 0:  # Check if tensor is not empty
                            print(f"  {key} min value: {value.min().item()}")
                            print(f"  {key} max value: {value.max().item()}")
                            print(f"  {key} mean value: {value.float().mean().item()}")
                        else:
                            print(f"  {key} is empty - skipping min/max/mean calculations")
    return data_loader_train

def collect_activations_batched(model, data_loader, device):
    print("\n=== Starting Activation Collection ===")
    activations = []
    # modality_samples = {} 
    examples_processed = 0
    
    def hook_fn(module, input, output):
        print(f'\nHook received output shape: {output.shape}')
        activations.append(output.clone().detach().cpu())
    
    # register hook on the spatial attention layer
    ninth_decoder_block = model.decoder[8]
    hook = ninth_decoder_block.register_forward_hook(hook_fn)
    print(f"Registered hook on decoder block 8")

    # # Process a single batch first for detailed analysis
    # first_batch = next(iter(data_loader))
    # print("\n=== DETAILED TENSOR ANALYSIS OF FIRST BATCH ===")
    
    # # Examine each modality's tensor in detail
    # for mod, d in first_batch.items():
    #     tensor = d.get('tensor', None)
    #     if tensor is not None:
    #         # Check if tensor contains meaningful data
    #         if tensor.numel() > 0:
    #             is_all_zeros = (tensor == 0).all().item()
    #             is_all_same = (tensor == tensor[0]).all().item()
    #             has_nan = torch.isnan(tensor).any().item()
                
    #             # Stats about tensor
    #             print(f"\nModality: {mod}")
    #             print(f"  Shape: {tensor.shape}")
    #             print(f"  All zeros? {is_all_zeros}")
    #             print(f"  All same value? {is_all_same}")
    #             print(f"  Has NaN? {has_nan}")
                
    #             # If not all zeros/same, examine a bit more
    #             if not is_all_zeros and not is_all_same and not has_nan:
    #                 print(f"  Min: {tensor.min().item()}")
    #                 print(f"  Max: {tensor.max().item()}")
    #                 print(f"  Mean: {tensor.float().mean().item()}")
    #                 print(f"  Std: {tensor.float().std().item()}")
                    
    #                 # Count non-zero samples
    #                 non_zero_count = (tensor.abs().sum(dim=tuple(range(1, tensor.dim()))) > 0).sum().item()
    #                 print(f"  Non-zero samples: {non_zero_count} / {tensor.shape[0]}")
                    
    #                 # This is a modality with real data
    #                 if non_zero_count > 0:
    #                     modality_samples[mod] = tensor[0].clone().detach().cpu()  # Save first sample
    
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
            
            for mod, d in processed_data.items():
                if mod in model.encoder_embeddings:
                    input_dict[mod] = d['tensor']

                if mod in model.decoder_embeddings:
                    target_mask[mod] = d['target_mask']

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
    
    return activations_tensor_flat, examples_processed

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

    args = Args()
    print(f"Data config path: {args.data_config}")
    print(f"Text tokenizer path: {args.text_tokenizer_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epoch size: {args.epoch_size}")

    data_loader_train = get_data(args)

    # Collect activations from a subset of the data
    print("\n=== Starting Main Collection Process ===")
    print(f"Will collect activations from all available examples in the data loader")
    activations, total_examples = collect_activations_batched(model, data_loader_train, device=device)

    # Save activations to file
    print("\n=== Saving Activations ===")
    output_file = f'/mnt/home/rzhang/ceph/activations_{total_examples}examples.pt'
    torch.save(activations, output_file)
    print(f"Saved {total_examples} examples of activations to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
