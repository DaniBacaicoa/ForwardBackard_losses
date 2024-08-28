"""" Load Datasets for classification problems
    Authors: Daniel Bacaicoa-Barber
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""

#importing libraries
import numpy as np

import openml

import sklearn

import sklearn.datasets
import sklearn.mixture
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

import pandas as pd

from ucimlrepo import fetch_ucirepo 

class Benchmark_Data_Handling(Dataset):
    def __init__(self, dataset, train_size, test_size = None, valid_size = None, batch_size = 64, shuffling = False, splitting_seed = None):
        self.dataset = dataset
        self.dataset_source = None

        self.tr_size = train_size
        self.val_size = valid_size
        self.test_size = test_size

        self.weak_labels = None
        self.virtual_labels = None

        self.batch_size = batch_size

        self.shuffle = shuffling

        self.splitting_seed = splitting_seed

        le = sklearn.preprocessing.LabelEncoder()

        if self.dataset == 'Noisy1':
            breast_cancer = fetch_ucirepo(id=14)
            X = breast_cancer.data.features 
            y = breast_cancer.data.targets
        elif self.dataset == 'Noisy2':
            statlog_german_credit_data = fetch_ucirepo(id=144) 
            X = statlog_german_credit_data.data.features 
            y = statlog_german_credit_data.data.targets
        elif self.dataset == 'Noisy3':
            heart_disease = fetch_ucirepo(id=45) 
            X = heart_disease.data.features
            y = heart_disease.data.targets 
        elif self.dataset == 'Noisy4':
            image_segmentation = fetch_ucirepo(id=50) 
            X = image_segmentation.data.features 
            y = image_segmentation.data.targets 