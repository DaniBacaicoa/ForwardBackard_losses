import os
import torch
import torch.nn as nn
import numpy as np
import argparse

import datasets.datasets as dtset
import utils.losses as losses
from utils.weakener import Weakener
from models.model import MLP
from utils.train_test_loop import train_and_evaluate
from Dataset_generation import generate_dataset,generate_iris_dataset
import pickle
import json

import os
import torch
import numpy as np
import pickle
import argparse
from src.dataset import Data_handling
from src.weakener import Weakener
from src.model import MLP
from utils.datasets_generation import generate_dataset
import utils.losses as losses
from utils.train_test_loop import train_and_evaluate

def main(args):
    reps = args.reps
    dataset_base_path = args.dataset_base_path
    dataset = args.dataset
    corruption = args.corruption
    corr_p = args.corr_p
    corr_n = args.corr_n
    loss_type = args.loss_typeç
    
    for i in range(reps):
        base_dir = dataset_base_path
        if corr_n is not None:
            folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p_+{corr_p}p_-{corr_n}')
        else:
            folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p{corr_p}')
        
        # Load the dataset
        with open(folder_path + f'/Dataset_{i}.pkl', 'rb') as f:
            Data, Weak = pickle.load(f)

        # Select the appropriate loss function
        if loss_type == 'Backward':
            loss_fn = losses.FwdBwdLoss(Weak.Y, np.eye(Weak.c))
        elif loss_type == 'Forward':
            loss_fn = losses.FwdBwdLoss(np.eye(Weak.d), Weak.M)
        elif loss_type == 'EM':
            loss_fn = losses.EMLoss(Weak.M)
        elif loss_type == 'LBL':
            loss_fn = losses.LBLoss()
        elif loss_type == 'Backward_opt':
            loss_fn = losses.FwdBwdLoss(Weak.Y_opt, np.eye(Weak.c))
        elif loss_type == 'Backward_conv':
            loss_fn = losses.FwdBwdLoss(Weak.Y_conv, np.eye(Weak.c))
        elif loss_type == 'Backward_opt_conv':
            loss_fn = losses.FwdBwdLoss(Weak.Y_opt_conv, np.eye(Weak.c))
        elif loss_type == 'OSL':
            loss_fn = losses.OSLCELoss()

        # Include weak labels based on loss type
        if loss_type == 'OSL':
            Data.include_weak(Weak.w)
        else:
            Data.include_weak(Weak.z)

        # Prepare data loaders
        trainloader, testloader = Data.get_dataloader(weak_labels='weak')

        # Initialize the model
        mlp = MLP(Data.num_features, [Data.num_features], Weak.c, dropout_p=0.3, bn=True, activation='relu')
        lr = MLP(Data.num_features, [], Weak.c, dropout_p=0. bn=False, activation='id')
        
        # Initialize the optimizer
        optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        
        # Train and evaluate the model
        mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                          loss_fn=loss_fn, corr_p=corr_p, num_epochs=100, 
                                          sound=10, rep=i)
        
        res_dir = f"Results/{dataset}_{corruption}"
        os.makedirs(folder_path, exist_ok=True)
        if corr_n is not None:
            file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}.csv'
        else:
            file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}.csv'
        file_path = os.path.join(res_dir, file_name)
        results.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for dataset handling, model training and evaluation.")
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions.")
    parser.add_argument("--dataset_base_path", type=str, default='Datasets/weak_datasets', help="Base path for datasets.")
    parser.add_argument("--dataset", type=str, default='image', help="Dataset name.")
    parser.add_argument("--corruption", type=str, default='pll', help="Corruption type.")
    parser.add_argument("--corr_p", type=float, default=0.5, help="Positive corruption probability.")
    parser.add_argument("--corr_n", type=float, default=None, help="Negative corruption probability.")
    parser.add_argument("--loss_type", type=str, default='Forward', help="Type of loss function to use.")
    
    args = parser.parse_args()
    main(args)
