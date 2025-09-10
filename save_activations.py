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

    # # Check the structure of a batch
    # print("\n=== Sample Batch Structure ===")
    # sample_batch = next(iter(data_loader_train))
    # if isinstance(sample_batch, dict):
    #     print(f"Batch keys: {list(sample_batch.keys())}")
    #     for mod_name, mod_data in sample_batch.items():
    #         print(f"\nModality: {mod_name}")
    #         for key, value in mod_data.items():
    #             if isinstance(value, torch.Tensor):
    #                 print(f"  {key} shape: {value.shape}")
    #                 print(f"  {key} dtype: {value.dtype}")
    #                 print(f"  {key} device: {value.device}")
    #                 if key == 'tensor':
    #                     if value.numel() > 0:  # Check if tensor is not empty
    #                         print(f"  {key} min value: {value.min().item()}")
    #                         print(f"  {key} max value: {value.max().item()}")
    #                         print(f"  {key} mean value: {value.float().mean().item()}")
    #                     else:
    #                         print(f"  {key} is empty - skipping min/max/mean calculations")
    return data_loader_train

def collect_activations_batched(model, data_loader, device, max_examples_per_file=5000):
    print("\n=== Starting Activation Collection ===")
    activations = []
    examples_processed = 0
    file_count = 0
    saved_files = []
    
    def hook_fn(module, input, output):
        print(f'\nHook received output shape: {output.shape}')
        activations.append(output.clone().detach().cpu())
    
    # register hook on the spatial attention layer
    ninth_decoder_block = model.decoder[8]
    hook = ninth_decoder_block.register_forward_hook(hook_fn)
    print(f"Registered hook on decoder block 8")

    def save_current_activations():
        nonlocal activations, file_count, saved_files
        if not activations:
            return
        
        # Concatenate current activations
        print(f"\nSaving chunk {file_count + 1}...")
        activations_tensor = torch.cat(activations, dim=0)
        print(f"Chunk activations shape: {activations_tensor.shape}")
        
        # Flatten activations
        sample_size, seq_length, embed_dim = activations_tensor.shape
        activations_tensor_flat = activations_tensor.reshape(sample_size * seq_length, embed_dim)
        print(f"Flattened chunk shape: {activations_tensor_flat.shape}")
        
        # Save to file
        output_file = f'/mnt/home/rzhang/ceph/activations_{activations_tensor_flat.shape[0]}examples_chunk{file_count + 1}.pt'
        torch.save(activations_tensor_flat, output_file)
        saved_files.append(output_file)
        print(f"Saved chunk to {output_file}")
        print(f"Chunk file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Clear activations and increment counter
        activations = []
        file_count += 1

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
            
            # Check if we should save current chunk
            if examples_processed >= (file_count + 1) * max_examples_per_file:
                save_current_activations()
            
    # Save any remaining activations
    if activations:
        save_current_activations()
    
    # Remove the hook
    hook.remove()
    print("\nHook removed")
    
    print(f"\n=== Summary ===")
    print(f"Total examples processed: {examples_processed}")
    print(f"Number of files created: {len(saved_files)}")
    for i, file_path in enumerate(saved_files):
        print(f"  File {i+1}: {file_path}")
    
    return saved_files, examples_processed

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
    print(f"Will collect activations and save them in chunks of 5000 examples each")
    saved_files, total_examples = collect_activations_batched(model, data_loader_train, device=device)

    # Print summary of saved files
    print("\n=== Activation Collection Complete ===")
    print(f"Total examples processed: {total_examples}")
    print(f"Number of files created: {len(saved_files)}")
    total_size_mb = 0
    for i, file_path in enumerate(saved_files):
        file_size_mb = os.path.getsize(file_path) / (1024*1024)
        total_size_mb += file_size_mb
        print(f"  File {i+1}: {file_path} ({file_size_mb:.2f} MB)")
    print(f"Total storage used: {total_size_mb:.2f} MB")
