# train_test_loop_clothing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import time
import os




def train_and_evaluate_clothing(model, trainloader, testloader, optimizer, loss_fn, num_epochs, num_classes, rep=None, sound=1, loss_type=None, initial_lr=None):
    """
    Training and evaluation loop specifically adapted for Clothing1M.

    Args:
        model (nn.Module): The neural network model.
        trainloader (DataLoader): DataLoader for the training set (yields image, noisy_label).
        testloader (DataLoader): DataLoader for the test set (yields image, clean_label).
        optimizer (Optimizer): The optimizer for training.
        loss_fn (nn.Module): The loss function.
        num_epochs (int): Total number of epochs to train.
        num_classes (int): Number of classes in the dataset (14 for Clothing1M).
        rep (int, optional): Repetition ID for logging. Defaults to None.
        sound (int, optional): Print logs every `sound` epochs. Defaults to 1.
        loss_type (str, optional): String identifier for the loss function type (used for logging and label formatting). Defaults to None.
        initial_lr (float, optional): Initial learning rate for logging. If None, taken from optimizer.
    """
    # Seed setting (optional, but good practice)
    seed = 42 + (rep if rep is not None else 0) # Vary seed per repetition
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if initial_lr is None:
         initial_lr = optimizer.param_groups[0]['lr']

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n--- Starting Repetition {rep if rep is not None else 'N/A'} ---")
    print(f"Model: {model.__class__.__name__}, Loss: {loss_type}, Epochs: {num_epochs}, Initial LR: {initial_lr}")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # --- Training Phase ---
        for i, (inputs, noisy_labels) in enumerate(trainloader):
            inputs, noisy_labels = inputs.to(device), noisy_labels.to(device)
            #noisy_labels = torch.nn.functional.one_hot(noisy_labels, num_classes=num_classes).float()  # One-hot encoding

            optimizer.zero_grad()
            outputs = model(inputs)


            loss = loss_fn(outputs, noisy_labels)
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Un-normalize loss

            # Optional: Print batch loss periodically
            # if i % 200 == 199:
            #     print(f"\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], Batch Loss: {loss.item():.4f}", end="")

        # Normalize train loss by dataset size
        train_loss = running_loss / len(trainloader.dataset)

        # --- Evaluation Phase ---
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, clean_labels in testloader:
                inputs, clean_labels = inputs.to(device), clean_labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += clean_labels.size(0)
                correct_test += (predicted == clean_labels).sum().item()

        test_acc = correct_test / total_test

        # --- Logging ---
        actual_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'optimizer': type(optimizer).__name__,
            'loss_fn': loss_type,
            'repetition': rep,
            'initial_lr': initial_lr,
            'actual_lr': actual_lr,
            'epoch_time_sec': epoch_time,
        }
        results.append(epoch_data)

        if (epoch + 1) % sound == 0:
            print(f"\rEpoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | LR: {actual_lr:.6f} | Time: {epoch_time:.2f}s")

    # Convert results to DataFrame at the end
    results_df = pd.DataFrame(results)
    print(f"--- Finished Repetition {rep if rep is not None else 'N/A'} ---")

    return model, results_df