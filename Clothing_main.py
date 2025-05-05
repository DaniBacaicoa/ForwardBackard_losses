# clothing_main.py (Revised V2 - with Matrix Factorization)
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy.linalg # Added for pinv

# Import from your provided files (assuming they are in the same directory or src)
from src.dataset import Clothing1MDataset # Using the efficient version from dataset.py
from src.weakener import Weakener         # From weakener.py
from src.model import ResNet50            # From model.py
import utils.losses as loss_lib             # From losses.py
from utils.train_test_loop_clothing import train_and_evaluate_clothing


# --- Helper Functions for Loading Clothing1M Metadata ---
# (Same as before - load_labels, load_key_list)
def load_labels(filepath):
    """Loads image path -> label mapping from a file."""
    labels = {}
    count = 0
    malformed = 0
    if not os.path.exists(filepath):
        print(f"ERROR: Label file not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_path = os.path.normpath(parts[0])
                labels[image_path] = int(parts[1])
                count += 1
            else:
                malformed += 1
    if malformed > 0:
        print(f"Warning: Skipped {malformed} malformed lines in {filepath}")
    print(f"Loaded {count} labels from {os.path.basename(filepath)}")
    return labels

def load_key_list(filepath):
    """Loads a list of image paths (keys) from a file."""
    keys = []
    count = 0
    if not os.path.exists(filepath):
        print(f"ERROR: Key list file not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        for line in f:
            image_path = os.path.normpath(line.strip())
            if image_path:
                keys.append(image_path)
                count += 1
    print(f"Loaded {count} keys from {os.path.basename(filepath)}")
    return keys

# --- Helper Function for Matrix Estimation ---
# (Same as before)
def estimate_transition_matrix(clean_labels_path, noisy_labels_path, num_classes):
    """ Estimates M = P(noisy | clean). """
    print("\n--- Estimating Transition Matrix M ---")
    print(f"Using clean labels: {clean_labels_path}")
    print(f"Using noisy labels: {noisy_labels_path}")
    clean_labels_map = load_labels(clean_labels_path)
    noisy_labels_map = load_labels(noisy_labels_path)
    if clean_labels_map is None or noisy_labels_map is None or not clean_labels_map or not noisy_labels_map:
        print("ERROR: Failed to load one or both label files/maps. Cannot estimate matrix.")
        return None
    common_keys = set(clean_labels_map.keys()) & set(noisy_labels_map.keys())
    print(f"Found {len(common_keys)} images common to both clean and noisy label sets.")
    if not common_keys:
        print("ERROR: No common images found. Cannot estimate matrix.")
        return None
    count_matrix = np.zeros((num_classes, num_classes), dtype=float)
    valid_pairs = 0
    for key in common_keys:
        clean_label = clean_labels_map[key]
        noisy_label = noisy_labels_map[key]
        if 0 <= clean_label < num_classes and 0 <= noisy_label < num_classes:
            # Note: Original weakener.py had count_matrix[noisy, clean]
            # Let's stick to M[clean, noisy] = P(noisy|clean) convention here
            count_matrix[noisy_label,clean_label] += 1.0
            valid_pairs += 1
    if valid_pairs == 0:
        print("ERROR: No valid label pairs found for matrix estimation.")
        return None
    print(f"Constructed count matrix based on {valid_pairs} valid common pairs.")
    
    transition_matrix = count_matrix / np.sum(count_matrix, axis=0, keepdims=True)
    print("--- Transition Matrix M Estimation Complete ---")
    return transition_matrix

# --- Helper Function for Matrix Factorization ---
def factorize_matrix_alternating_sgd(M_target, num_classes, max_iter=10000, tol=1e-3, lr=1e-2, eps=1e-8):
    """
    Factorizes M_target into M_l @ M_r using alternating projected SGD.
    Assumes M_target[clean, noisy] = P(noisy|clean).
    We want M_target ≈ M_l @ M_r where Ml, Mr are column stochastic.
    (Adapting the logic from the user's weakener.py)

    Args:
        M_target (np.ndarray): The target matrix (num_classes x num_classes).
        num_classes (int): Number of classes.
        max_iter (int): Maximum iterations for optimization.
        tol (float): Tolerance for Frobenius norm difference to stop early.
        lr (float): Learning rate for gradient descent.
        eps (float): Small value for clipping and projection denominator.

    Returns:
        tuple (np.ndarray, np.ndarray) or (None, None): Factors (M_l, M_r) or None if failed.
    """
    print("\n--- Factorizing Matrix M -> Ml * Mr ---")
    n = num_classes
    if M_target is None or M_target.shape != (n, n):
        print("ERROR: Invalid target matrix M for factorization.")
        return None, None

    # Initialize M_l, M_r >= 0 and column-stochastic
    rng = np.random.default_rng(42)
    M_l = rng.uniform(eps, 1-eps, size=(n, n))
    M_l /= (M_l.sum(axis=0, keepdims=True) + eps) # Add eps for stability
    M_r = rng.uniform(eps, 1-eps, size=(n, n))
    M_r /= (M_r.sum(axis=0, keepdims=True) + eps)

    last_err = float('inf')
    it=0
    while last_err > 3e-2:
        it += 1
        # Update M_l (fix M_r)
        grad_Ml = (M_l @ M_r - M_target) @ M_r.T
        M_l -= lr * grad_Ml
        # Project M_l: clip to [eps, 1-eps] and normalize columns
        M_l = np.clip(M_l, eps, 1-eps)
        M_l /= M_l.sum(axis=0, keepdims=True)

        # Update M_r (fix M_l)
        grad_Mr = M_l.T @ (M_l @ M_r - M_target)
        M_r -= lr * grad_Mr
        # Project M_r: clip to [eps, 1-eps] and normalize columns
        M_r = np.maximum(M_r, 0)
        M_r /= (M_r.sum(axis=0, keepdims=True) + eps)
        current_err = np.linalg.norm(M_l @ M_r - M_target, 'fro')
        last_err = current_err

    print("--- Matrix Factorization Complete with error:", last_err)
    return M_l, M_r
# --------------------------------------------------------

def main(args):
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Provided dataset path: {args.dataset_path}")
    if not os.path.isdir(args.dataset_path):
         print(f"ERROR: Dataset path does not exist or is not a directory: {args.dataset_path}")
         return
    num_classes = 14

    # --- Estimate M ---
    clean_label_kv_path = os.path.join(args.dataset_path, 'clean_label_kv.txt')
    noisy_label_kv_path = os.path.join(args.dataset_path, 'noisy_label_kv.txt')
    M = estimate_transition_matrix(clean_label_kv_path, noisy_label_kv_path, num_classes)
    if M is None: return

    # --- Factorize M into Ml and Mr (needed for FB_decomposed) ---
    M_l, M_r = None, None
    if args.loss_type == 'FB_decomposed':
        M_l, M_r = factorize_matrix_alternating_sgd(M, num_classes)
        if M_l is None or M_r is None:
            print("ERROR: Matrix factorization failed. Cannot use FB_decomposed loss. Exiting.")
            return

    # --- Data Loading ---
    print("\n--- Setting up Dataset and DataLoaders ---")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    noisy_train_key_list_path = os.path.join(args.dataset_path, 'noisy_train_key_list.txt')
    clean_test_key_list_path = os.path.join(args.dataset_path, 'clean_test_key_list.txt')
    noisy_labels_map = load_labels(noisy_label_kv_path)
    clean_labels_map = load_labels(clean_label_kv_path)
    train_keys = load_key_list(noisy_train_key_list_path)
    test_keys = load_key_list(clean_test_key_list_path)
    if train_keys is None or test_keys is None or noisy_labels_map is None or clean_labels_map is None: return

    train_keys_filtered = [k for k in train_keys if k in noisy_labels_map]
    train_labels = [noisy_labels_map[k] for k in train_keys_filtered]
    print(f"Using {len(train_keys_filtered)} training samples (with noisy labels).")
    test_keys_filtered = [k for k in test_keys if k in clean_labels_map]
    test_labels = [clean_labels_map[k] for k in test_keys_filtered]
    print(f"Using {len(test_keys_filtered)} testing samples (with clean labels).")
    if not train_keys_filtered or not test_keys_filtered:
         print("ERROR: No valid training or testing samples found. Check paths/files.")
         return

    train_dataset = Clothing1MDataset(root_dir=args.dataset_path, image_keys=train_keys_filtered, labels=train_labels, transform=train_transform)
    test_dataset = Clothing1MDataset(root_dir=args.dataset_path, image_keys=test_keys_filtered, labels=test_labels, transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("--- DataLoaders Ready ---")

    # --- Model Definition ---
    print(f"\n--- Initializing ResNet50 model (fine_tune_all={args.fine_tune}) ---")
    model = ResNet50(num_classes=num_classes, fine_tune_all=args.fine_tune).to(device)

    # --- Loss Function ---
    print(f"\n--- Setting up Loss Function: {args.loss_type} ---")
    if args.loss_type == 'Forward':
        # FwdLoss expects M = P(noisy|clean)
        # FwdBwdLoss(B=I, F=M) is equivalent if FwdLoss isn't separate
        # Using FwdBwdLoss form for consistency: B=Identity, F=M
        loss_fn = loss_lib.FwdBwdLoss(B=np.eye(num_classes), F=M)
        # If you have a separate FwdLoss(M) class, use: loss_fn = loss_lib.FwdLoss(M=M)
    elif args.loss_type == 'Backward':
        # Backward loss needs Y = pinv(M)
        # Uses FwdBwdLoss form: B=Y, F=Identity
        print("Calculating pseudo-inverse of M for Backward loss...")
        try:
            # Ensure M is P(noisy|clean) before inverting
            # If M was P(clean|noisy), you'd use M directly as B
            Y = np.linalg.pinv(M)
        except np.linalg.LinAlgError:
            print("ERROR: Could not compute pseudo-inverse of M. Using Identity fallback.")
            Y = np.eye(num_classes)
        loss_fn = loss_lib.FwdBwdLoss(B=Y, F=np.eye(num_classes))
    elif args.loss_type == 'FB_decomposed':
        # Uses FwdBwdLoss form: B=pinv(Ml), F=Mr
        print("Calculating pseudo-inverse of Ml for FB_decomposed loss...")
        B_inv_Ml = np.linalg.pinv(M_l)
        loss_fn = loss_lib.FwdBwdLoss(B=B_inv_Ml, F=M_r)
    else:
        # Ensure the loss type is added to argparse choices if new
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    print(f"Loss function '{args.loss_type}' initialized.")

    # --- Optimizer ---
    print(f"\n--- Setting up Optimizer: Adam (lr={args.lr}) ---")
    if args.fine_tune:
        print("Optimizing all model parameters.")
        parameters_to_optimize = model.parameters()
    else:
        print("Optimizing only the final classifier layer (resnet50.fc).")
        try:
            # Ensure model structure matches access 'resnet50.fc'
            parameters_to_optimize = model.resnet50.fc.parameters()
            if not list(parameters_to_optimize):
                 print("Warning: No parameters found in model.resnet50.fc. Optimizing all parameters instead.")
                 parameters_to_optimize = model.parameters()
            else: parameters_to_optimize = model.resnet50.fc.parameters() # Reassign
        except AttributeError:
             print("Warning: Could not access model.resnet50.fc. Optimizing all parameters instead.")
             parameters_to_optimize = model.parameters()
    optimizer = optim.Adam(parameters_to_optimize, lr=args.lr)

    # --- Training & Evaluation ---
    print(f"\n--- Starting Training for {args.epochs} epochs ---")
    initial_lr = optimizer.param_groups[0]['lr']
    model, results_df = train_and_evaluate_clothing(
        model=model, trainloader=train_loader, testloader=test_loader, optimizer=optimizer,
        loss_fn=loss_fn, num_epochs=args.epochs, num_classes=num_classes,
        rep=args.rep_id, sound=1, loss_type=args.loss_type, initial_lr=initial_lr)
    print("\n--- Training Finished ---")

    # --- Save Results ---
    results_dir = f"Results/Clothing1M_{args.loss_type}_FT{args.fine_tune}_LR{args.lr}"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_rep{args.rep_id}_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    model_filename = f"model_rep{args.rep_id}_{timestamp}.pth"
    model_path = os.path.join(results_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model state saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing1M Training with ResNet50 (Revised V2)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root directory of the Clothing1M dataset.")
    # Added FB_decomposed back to choices
    parser.add_argument("--loss_type", type=str, default='Forward', choices=['Forward', 'Backward', 'FB_decomposed', 'CrossEntropy'], help="Type of loss function.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--fine_tune", action='store_true', help="Fine-tune all ResNet50 layers instead of just the classifier.")
    parser.add_argument("--rep_id", type=int, default=0, help="Repetition ID for saving results.")

    args = parser.parse_args()
    main(args)

#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type Forward --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type Backward --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type FB_decomposed --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 

#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type Forward --lr 1e-4 --epochs 10 --batch_size 32 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type Backward --lr 1e-4 --epochs 10 --batch_size 32 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type FB_decomposed --lr 1e-4 --epochs 10 --batch_size 32  