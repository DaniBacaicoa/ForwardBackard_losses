# clothing_main.py
import os
import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
from torch.utils.data import DataLoader

# Import from your provided files (assuming they are in a 'src' or similar directory)
# Adjust paths if necessary
from src.dataset import Clothing1MDataset # Using the efficient version from dataset.py
from src.weakener import Weakener         # From weakener.py
from src.model import ResNet50            # From model.py
import utils.losses as loss_lib             # From losses.py
from utils.train_test_loop_clothing import train_and_evaluate_clothing
#from train_test_loop import train_and_evaluate # From train_test_loop.py
from torchvision import transforms

def main(args):
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Setting up dataset...")
    # Define transformations (consistent with ResNet50 requirements)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224), # Use RandomCrop for training augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), # Use CenterCrop for evaluation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Helper function to load keys and labels (adapt from dataset.py or weakener.py)
    def load_labels(filepath):
        labels = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_path = os.path.normpath(parts[0])
                    labels[image_path] = int(parts[1])
        return labels

    def load_key_list(filepath):
        keys = []
        with open(filepath, 'r') as f:
            for line in f:
                image_path = os.path.normpath(line.strip())
                if image_path:
                    keys.append(image_path)
        return keys

    # Load necessary keys and labels
    noisy_label_kv_path = os.path.join(args.dataset_path, 'noisy_label_kv.txt')
    clean_label_kv_path = os.path.join(args.dataset_path, 'clean_label_kv.txt')
    noisy_train_key_list_path = os.path.join(args.dataset_path, 'noisy_train_key_list.txt')
    clean_test_key_list_path = os.path.join(args.dataset_path, 'clean_test_key_list.txt')
    category_names_eng_path = os.path.join(args.dataset_path, 'category_names_eng.txt')

    noisy_labels_map = load_labels(noisy_label_kv_path)
    clean_labels_map = load_labels(clean_label_kv_path) # Needed for test set and potentially M estimation

    # Use all noisy keys for training, or adjust logic if you need clean + noisy split
    train_keys = load_key_list(noisy_train_key_list_path)
    # Filter train keys to only those present in the noisy map
    train_keys = [k for k in train_keys if k in noisy_labels_map]
    train_labels = [noisy_labels_map[k] for k in train_keys]
    print(f"Loaded {len(train_keys)} training samples (using noisy labels).")

    test_keys = load_key_list(clean_test_key_list_path)
    # Filter test keys for those present in the clean map
    test_keys = [k for k in test_keys if k in clean_labels_map]
    test_labels = [clean_labels_map[k] for k in test_keys]
    print(f"Loaded {len(test_keys)} testing samples (using clean labels).")

    num_classes = 14 # Clothing1M specific

    # Create Datasets using the efficient class from dataset.py
    train_dataset = Clothing1MDataset(root_dir=args.dataset_path,
                                        image_keys=train_keys,
                                        labels=train_labels,
                                        transform=train_transform)
    test_dataset = Clothing1MDataset(root_dir=args.dataset_path,
                                       image_keys=test_keys,
                                       labels=test_labels,
                                       transform=eval_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Weakening Matrix Estimation ---
    print("Estimating/Loading weakening matrix...")
    # Use the Weakener class to estimate M from clean/noisy labels
    # Note: This estimation requires BOTH clean and noisy labels for the *same* images.
    # The weakener.py function compares noisy_label_kv and clean_label_kv.
    weakener = Weakener(true_classes=num_classes)
    try:
        # Assuming weakener.py has the estimate_transition_matrix adapted or integrated
        weakener.estimate_transition_matrix(clean_label_kv_path, noisy_label_kv_path, num_classes)
        M = weakener.M # Extract the estimated matrix
        Mr = weakener.Mr # Extract the right factor if needed
        Ml = weakener.Ml # Extract the left factor if needed
        Y = np.linalg.inv(M) # Compute the inverse if needed


    except Exception as e:
         print(f"Error during matrix estimation/factorization: {e}")
         print("Proceeding with Identity Matrix as fallback M.")
         # Fallback to identity if estimation fails or is not desired
         M = np.eye(num_classes)
         # weakener.Ml = np.eye(num_classes)
         # weakener.Mr = np.eye(num_classes)
         # weakener.Y = np.eye(num_classes)

    # --- Model Definition ---
    print(f"Initializing ResNet50 model (fine_tune_all={args.fine_tune})...")
    model = ResNet50(num_classes=num_classes, fine_tune_all=args.fine_tune).to(device)

    # --- Loss Function ---
    print(f"Using loss function: {args.loss_type}")
    if args.loss_type == 'Forward':
        loss_fn = loss_lib.FwdBwdLoss(np.eye(num_classes), M)
    elif args.loss_type == 'Backward':
        Y = np.linalg.pinv(M) # Calculate pseudo-inverse
        loss_fn = loss_lib.FwdBwdLoss(B=Y, F=np.eye(num_classes))
    elif args.loss_type == 'FB_decomposed':
        B = np.linalg.inv(Ml)
        loss_fn = loss_lib.FwdBwdLoss(B, Mr)
    # Add other loss types from losses.py as needed
    # elif args.loss_type == 'EM':
    #     loss_fn = loss_lib.EMLoss(M=M)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    # --- Optimizer ---
    print(f"Setting up optimizer: Adam (lr={args.lr})")
    if args.fine_tune:
        print("Optimizing all parameters.")
        parameters_to_optimize = model.parameters()
    else:
        print("Optimizing only the final classifier layer.")
        # Ensure the model class correctly exposes the final layer parameters
        # For ResNet50 in model.py, it's model.resnet50.fc.parameters()
        parameters_to_optimize = model.resnet50.fc.parameters()

    optimizer = optim.Adam(parameters_to_optimize, lr=args.lr)

    # --- Training & Evaluation ---
    print(f"Starting training for {args.epochs} epochs...")
    # Adapt train_and_evaluate if needed for Clothing1M's data format (img, label) vs (img, weak, true)
    # The current train_and_evaluate expects (inputs, vl, targets)
    # We might need to adjust it or the dataloader.
    # For now, assuming train_and_evaluate can handle (img, label) from dataloader
    # and uses the correct labels for loss calculation vs accuracy calculation.
    # A modification might be needed in train_test_loop.py:
    # E.g., if loss needs noisy label (vl) but accuracy needs true label (targets)

    # --- Placeholder for adapting train_and_evaluate ---
    # Modify train_and_evaluate in train_test_loop.py:
    # 1. Change the loop to: for inputs, labels in trainloader:
    # 2. Pass the correct label to the loss: loss = loss_fn(outputs, labels) # or maybe noisy labels depending on loss type
    # 3. Calculate accuracy against clean labels (requires clean labels available or test set only)

    # Since train_and_evaluate expects (inputs, vl, targets), let's pass labels as both vl and targets
    # Modify train_and_evaluate loop to handle this:
    # In train loop: inputs, labels = inputs.to(device), labels.to(device)
    #                loss = loss_fn(outputs, labels) # If loss uses noisy labels
    #                _, true_labels = torch.max(labels, dim=1) # If labels are one-hot, otherwise use directly
    #                correct_train += torch.sum(preds == true_labels)
    # In test loop: inputs, targets = inputs.to(device), targets.to(device)
    #               ... accuracy calculation based on test 'targets'

    print("NOTE: Ensure 'train_and_evaluate' in 'train_test_loop.py' correctly handles labels for loss calculation and accuracy metrics.")
    # Assuming train_and_evaluate is adapted or works with (img, noisy_label) for training loss
    # and (img, clean_label) for test accuracy.
    # For simplicity here, we'll run it, but be mindful of label usage in the loop.

    # We need a way to pass the "true" labels if the training loop expects them for accuracy calculation.
    # The efficient Clothing1MDataset currently returns (image, label) where label is noisy for train, clean for test.
    # Modifying train_and_evaluate is the cleanest way. For now, let's proceed assuming it's adapted.

    # --- Training & Evaluation ---
    print(f"Starting training for {args.epochs} epochs...")

    # Get initial LR for logging
    initial_lr = optimizer.param_groups[0]['lr']

    model, results_df = train_and_evaluate_clothing(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        num_classes=num_classes, # Pass num_classes=14
        rep=args.rep_id,
        sound=1, # Print log every epoch
        loss_type=args.loss_type,
        initial_lr=initial_lr
    )

    print("--- Training Finished ---")

    # --- Save Results ---
    results_dir = f"Results/Clothing1M_{args.loss_type}_FT{args.fine_tune}" # Add fine-tune status
    os.makedirs(results_dir, exist_ok=True)
    # timestamp = time.strftime("%Y%m%d-%H%M%S") # Use time from the function or generate new one
    # Get current time for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_filename = f"results_rep{args.rep_id}_lr{args.lr}_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Save the final model state
    model_filename = f"model_rep{args.rep_id}_lr{args.lr}_{timestamp}.pth"
    model_path = os.path.join(results_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model state saved to {model_path}")
'''
    # --- Mock Training Loop (if train_and_evaluate is not adapted) ---
    print("\n--- Running Mock Training Loop (Adapt train_test_loop.py for full run) ---")
    results = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        # Mock training step
        for i, (inputs, noisy_labels) in enumerate(train_loader):
            inputs, noisy_labels = inputs.to(device), noisy_labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Loss calculation depends on loss_fn's expectation (noisy labels assumed here)
            if args.loss_type == 'CrossEntropy':
                 loss = loss_fn(outputs, noisy_labels) # Standard CE uses target labels
            elif hasattr(loss_fn, 'M') or hasattr(loss_fn,'B'): # Our custom losses might need index or one-hot
                 noisy_labels_one_hot = torch.nn.functional.one_hot(noisy_labels, num_classes=num_classes).float()
                 loss = loss_fn(outputs, noisy_labels_one_hot) # Assuming losses take one-hot weak labels (z)
            else:
                 loss = loss_fn(outputs, noisy_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            if i % 100 == 99: # Print progress
                 print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Mock evaluation step
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, clean_labels in test_loader:
                inputs, clean_labels = inputs.to(device), clean_labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += clean_labels.size(0)
                correct_test += (predicted == clean_labels).sum().item()

        epoch_test_acc = 100 * correct_test / total_test
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {epoch_train_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%")

        # Store mock results
        results.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'test_acc': epoch_test_acc / 100.0, # Store as fraction
            'loss_fn': args.loss_type,
            'lr': args.lr,
            'rep': args.rep_id
        })
    print("--- Mock Training Finished ---")
    results_df = pd.DataFrame(results) # Use mock results

    # --- Save Results ---
    results_dir = f"Results/Clothing1M_{args.loss_type}"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = f"results_rep{args.rep_id}_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Save the final model state
    model_filename = f"model_rep{args.rep_id}_{timestamp}.pth"
    model_path = os.path.join(results_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model state saved to {model_path}")
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing1M Training with ResNet50")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root of the Clothing1M dataset.")
    parser.add_argument("--loss_type", type=str, default='Forward', choices=['Forward', 'Backward', 'FB_decomposed', 'CrossEntropy'], help="Type of loss function.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--fine_tune", action='store_true', help="Fine-tune all ResNet50 layers instead of just the classifier.")
    parser.add_argument("--rep_id", type=int, default=0, help="Repetition ID for saving results.")
    # Add other necessary arguments (e.g., corruption parameters if not using estimated M)

    args = parser.parse_args()
    main(args)
    
    
    
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M --loss_type Forward --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/datasets/Clothing1M --loss_type Backward --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 
#python Clothing_main.py --dataset_path /export/usuarios_ml4ds/danibacaicoa/datasets/Clothing1M --loss_type FB_decomposed --lr 1e-4 --epochs 10 --batch_size 32 --fine_tune 