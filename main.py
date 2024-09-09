import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
from ucimlrepo import fetch_ucirepo 

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
    loss_type = args.loss_type
    epochs = args.epochs
    model = args.model



    for i in range(reps):
        generate_dataset(dataset=dataset,corruption=corruption,corr_p=corr_p,repetitions=i)


    for i in range(reps):
        
        base_dir = dataset_base_path
        if corr_n is not None:
            folder_path = os.path.join(base_dir, f'{dataset}_{corruption}_p{corr_p}')
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
        elif loss_type == 'Forward_opt':
            pest = torch.from_numpy(Weak.generate_wl_priors())
            tm = torch.from_numpy(Weak.M)
            B = tm @ torch.inverse(tm.T @ torch.inverse(torch.diag(pest)) @ tm) @ tm.T @ torch.inverse(torch.diag(pest))
            loss_fn = losses.FwdBwdLoss(B, Weak.M)
        elif loss_type == 'EM':
            loss_fn = losses.EMLoss(Weak.M)
        elif loss_type == 'LBL':
            loss_fn = losses.FwdBwdLoss(Weak.Y_conv, np.eye(Weak.c),k=1,beta=1.5)
            #loss_fn = losses.LBLoss()
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

        if model == 'lr':
            lr = MLP(Data.num_features, [], Weak.c, dropout_p=0, bn=False, activation='id')
            optim = torch.optim.Adam(lr.parameters(), lr=1e-3)
            lr, results = train_and_evaluate(lr, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            if dataset == 'gmm':
                results_dict = {'overall_models': lr}
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                file_path = os.path.join(res_dir, file_name)
                pickle_path = os.path.join(res_dir, pickle_name)
                results.to_csv(file_path, index=False)
                with open(pickle_path, "wb") as f:
                    pickle.dump(results_dict, f)
            else:
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                file_path = os.path.join(res_dir, file_name)
                results.to_csv(file_path, index=False)
        else:
            mlp = MLP(Data.num_features, [500], Weak.c, dropout_p=0.0, bn=False, activation='relu')
            optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, 
                                            loss_fn=loss_fn, corr_p=corr_p, num_epochs=epochs, 
                                            sound=10, rep=i, loss_type=loss_type)
            if dataset == 'gmm':
                results_dict = {'overall_models': mlp}
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                    pickle_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.pkl'
                file_path = os.path.join(res_dir, file_name)
                pickle_path = os.path.join(res_dir, pickle_name)
                results.to_csv(file_path, index=False)
                with open(pickle_path, "wb") as f:
                    pickle.dump(results_dict, f)
            else:
                res_dir = f"Results/{dataset}_{corruption}"
                os.makedirs(res_dir, exist_ok=True)
                if corr_n is not None:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
                else:
                    file_name = f'{loss_type}_p_+{corr_p}p_-{corr_n}_{i}.csv'
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
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--model", type=str, default='lr', help="Whether to use an MLP or a LR" )
    
    args = parser.parse_args()
    main(args)


# BINARY
## Noisy
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.2 --corr_n 0.2 --epochs 50
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.3 --corr_n 0.1 --epochs 50
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.4 --corr_n 0.4 --epochs 50

# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.2 --corr_n 0.2 --epochs 50
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.3 --corr_n 0.1 --epochs 50
# python main.py --reps 10 --dataset banknote-authentication --model lr --corruption Noisy_Natarajan --loss_type Forward --corr_p 0.4 --corr_n 0.4 --epochs 50




# MNIST
## Noisy
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Backward_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption Noisy_Patrini_MNIST --loss_type Forward_opt --corr_p 0.8 --epochs 50


# MNIST
## Noisy
# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Backward_conv --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption Complementary --loss_type Forward_opt --corr_p 0.2 --epochs 50


# MNIST
## pll
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Backward_opt_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset mnist --model mlp --corruption pll --loss_type Forward_opt --corr_p 0.8 --epochs 50


# GMM
## Noisy
# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Forward --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Backward --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Backward_conv --corr_p 0.2 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption Complementary --loss_type Forward_opt --corr_p 0.2 --epochs 50


# GMM
## pll
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_opt_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_conv --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_conv --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Backward_conv --corr_p 0.8 --epochs 50

# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward_opt --corr_p 0.2 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward_opt --corr_p 0.5 --epochs 50
# python main.py --reps 10 --dataset gmm --model lr --corruption pll --loss_type Forward_opt --corr_p 0.8 --epochs 50


