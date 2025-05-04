import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
# Import your Data_handling class and necessary functions/classes
from src.dataset import Data_handling # Adjust import path if needed

print("Preparing Clothing1M metadata...")

# --- Configuration ---
metadata_base_dir = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/'
output_metadata_file = 'clothing1m_metadata.pkl' # Where to save the metadata

# --- Instantiate Data_handling to get basic info ---
# We don't need the full Datasets here, just the label info it loads
print("Loading label information via Data_handling...")
# Temporarily modify Data_handling or create a helper function
# to ONLY load labels and keys, not instantiate Datasets yet.
# OR, let it create Datasets, we just need the label info it gathers.
# Let's assume Data_handling loads necessary info into attributes like
# self.num_classes, self.train_labels_int, etc. (as in previous refactoring)

class MinimalDataHandler:
    # Simplified version to just get labels and class count
    # Adapt based on your actual Data_handling structure
    def __init__(self, dataset='clothing1m'):
        if dataset == 'clothing1m':
            metadata_dir = metadata_base_dir
            clean_label_kv_path = os.path.join(metadata_dir, 'clean_label_kv.txt')
            noisy_label_kv_path = os.path.join(metadata_dir, 'noisy_label_kv.txt')
            noisy_train_key_list_path = os.path.join(metadata_dir, 'noisy_train_key_list.txt') # Or use noisy_label_kv keys directly
            category_names_eng_path = os.path.join(metadata_dir, 'category_names_eng.txt')

            def load_labels(filepath):
                labels = {}
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            labels[os.path.normpath(parts[0])] = int(parts[1])
                return labels

            category_names = []
            if os.path.exists(category_names_eng_path):
                with open(category_names_eng_path, 'r') as f:
                    category_names = [line.strip() for line in f if line.strip()]
            self.num_classes = len(category_names)
            c = self.num_classes

            noisy_labels_map = load_labels(noisy_label_kv_path)
            # Assuming training uses all keys from noisy_label_kv.txt
            train_keys = list(noisy_labels_map.keys())
            self.train_labels_int = [noisy_labels_map[k] for k in train_keys] # List of noisy int labels
            self.weak_labels_tensor = torch.tensor(self.train_labels_int, dtype=torch.long)


# Instantiate the minimal handler (or your full one)
data_handler_info = MinimalDataHandler(dataset='clothing1m')
num_classes = data_handler_info.num_classes
noisy_labels_tensor = data_handler_info.weak_labels_tensor # Integer noisy labels

# --- Create the 'Weak' metadata dictionary ---
Weak_clothing1m = {}
Weak_clothing1m['c'] = num_classes
Weak_clothing1m['d'] = num_classes # Assuming weak labels dimension is num_classes

# Create the full one-hot encoded noisy label matrix Y (as numpy array)
# This IS feasible memory-wise (e.g., 1M x 14 floats)
print(f"Creating one-hot noisy label matrix Y ({len(noisy_labels_tensor)} x {num_classes})...")
Y_one_hot_np = F.one_hot(noisy_labels_tensor, num_classes=num_classes).float().numpy()
Weak_clothing1m['Y'] = Y_one_hot_np
# Your 'Backward' loss expects Weak.Y, so we provide it here.

# Provide the noisy integer labels as well (might correspond to Weak.z or Weak.w)
Weak_clothing1m['z'] = noisy_labels_tensor.numpy() # Use key 'z' or 'w' based on your convention

# Estimate/Load Corruption Matrix M (Crucial Step!)
# This is highly dependent on your methodology.
# Option A: Estimate M using validation data (if available and appropriate)
# Option B: Load a pre-computed M specific to Clothing1M
# Option C: Assume a structure or use an identity matrix if 'Forward' loss isn't used initially
# Placeholder: Needs actual implementation
print("Estimating/Loading Corruption Matrix M (Placeholder)...")
# M_estimated = estimate_M(...) # Replace with your actual estimation/loading
# For example purposes, let's use identity, but THIS IS LIKELY WRONG for Forward loss.
M_estimated = np.eye(num_classes)
Weak_clothing1m['M'] = M_estimated
print("WARNING: Using placeholder Identity Matrix for M. Replace with actual estimation/loading.")

# Compute/Load other specific matrices/vectors if needed by other loss types
# Weak_clothing1m['Y_opt'] = ...
# Weak_clothing1m['Y_conv'] = ...
# Weak_clothing1m['Ml'] = ...
# Weak_clothing1m['Mr'] = ...
# Weak_clothing1m['pest'] = ... # Priors might be estimated from noisy labels

# --- Save the metadata ---
print(f"Saving Clothing1M metadata to {output_metadata_file}...")
with open(output_metadata_file, 'wb') as f:
    pickle.dump(Weak_clothing1m, f)

print("Clothing1M metadata preparation complete.")