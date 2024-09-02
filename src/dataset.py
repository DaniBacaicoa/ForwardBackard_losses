"""" Load Datasets for classification problems
    Authors: Daniel Bacaicoa-Barber
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""

#importing libraries
import numpy as np

import openml
from ucimlrepo import fetch_ucirepo

import sklearn

import sklearn.datasets
import sklearn.mixture
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

import pandas as pd


class Data_handling(Dataset):
    '''
    The dataloader returns a pytorch dataset
    inputs:
        dataset: [str/int] refers to the selected dataset
            Synthetic datasets
            - 'hypercube'
            - 'blobs'
            - 'blobs2'
            Sklearn's datasets
            - 'iris'
            - 'digits'
            - 'covtype'
            Openml edited datasets
            - 'glass_2' (glass dataset but with 3 classes)
            Openml datasets
            -   'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
                'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
                'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
                'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
                'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
                'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
                'GesturePhaseSegmentationProcessed': 4538,
                'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
                'page-blocks': 30, 'yeast': 181, 'flags': 285,
                'visualizing_livestock': 685, 'diggle_table_a2': 694,
                'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477,
                'blood-transfusion': 1464, ' banknote-authentication': 1462
            UCI dtasets
            - TBD
    '''

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

        openml_ids = {
            'iris': 61,  # 150 x 5 - 3 (n_samples x n_feat - n_classes)
            'pendigits': 32,  # 10992 x 17 - 10
            'glass': 41,  # 214 x 10 - 7 (really 6)
            'segment': 36,  # 2310 x 19 - 7
            'vehicle': 54,  # 846 x 19 - 4
            'vowel': 307,  # 990 x 13 - 11 <- This has cat feat to deal with
            'wine': 187,  # 178 x 14 - 3
            'abalone': 1557,  # 4177 x 9 - 3 <- Firts cat feat
            'balance-scale': 11,  # 625 x 5 - 3
            'car': 21,  # 1728 x 7 - 4 <- All categoric
            'ecoli': 39,  # 336 x 8 - 8
            'satimage': 182,  # 6435 x 37 - 6
            'collins': 478,  # 500 x 5 - 6 <- The last feat is cat (but is number as str)
            'cardiotocography': 1466,  # 2126 x 35 - 10
            'JapaneseVowels': 375,  # 9961 x 14 - 9
            'autoUniv-au6-1000': 1555,  # 1000 x 300 - 4 <- This has cat feat to deal with
            'autoUniv-au6-750': 1549,  # 750 x 300 - 4  <- This has cat feat to deal with
            'analcatdata_dmft': 469,  # 797 x 4 - 6  <- This has cat feat to deal with
            'autoUniv-au7-1100': 1552,  # 1100 x 12 - 5 <- This has cat feat to deal with
            'GesturePhaseSegmentationProcessed': 4538,  # 9873 x 32 - 5
            'autoUniv-au7-500': 1554,  # 500 x 300 - 4  <- This has cat feat to deal with
            'mfeat-zernike': 22,  # 2000 x 48 - 10
            'zoo': 62,  # 101 x 16 - 7 <- This has dichotomus feat to deal with
            'page-blocks': 30,  # 5473 x 10 - 5
            'yeast': 181,  # 1484 x 8 - 10
            'flags': 285,  # 194 x 29 - 8  <- This has cat feat to deal with
            'visualizing_livestock': 685,  # 280 x 8 - 3 <- This has cat feat to deal with
            'diggle_table_a2': 694,  # 310 x 8 - 10
            'prnn_fglass': 952,  # 214 x 9 - 6
            'confidence': 468,  # 72 x 3 - 6
            'fl2000': 477,  # 67 x 15 - 5
            'blood-transfusion': 1464,  # 748 x 4 - 2
            'banknote-authentication': 1462,  # 1372 x 4 - 2
            'cifar10': 40927, # 60000 x (32x32x3=3072) - 10
            'breast-tissue': 15,  # 699 x 9 - 2
            'cholesterol': 141,  # 10^6 x 18 - 4  <- This has cat feat to deal with
            'liver-disorders': 145,  # 10^6 x 12 - 11  <- This has cat feat to deal with
            'pasture': 294,  # 6435 x 36 - 6
            'eucalyptus': 180,  # 110393 x 54 - 7 <- This has cat feat to deal with
            'dermatology': 35,  # 366 x 34 - 6 #X = X.astype(np.float64).values should be used
            'optdigits': 28,  # 5620 x 64 - 10
            'cmc': 23,  # 1473 x 9 - 3
            }
        uci_ids = {#'breast-cancer-wisconsin': 699,  # 699 x 9 - 2
            'diabetes': 768,  # 768 x 8 - 2 #'heart-disease': 303,  # 303 x 13 - 5
            'ionosphere': 351,  # 351 x 34 - 2
            'sonar': 208,  # 208 x 60 - 2
            'parkinsons': 195,  # 195 x 22 - 2
            'seeds': 210,  # 210 x 7 - 3
            'seismic-bumps': 2584,  # 2584 x 18 - 2
            'spam': 4601,  # 4601 x 57 - 2
            'letter-recognition': 20000,  # 20000 x 16 - 26
            'bupa': 345,  # 345 x 6 - 2 (liver disorders)
            'lung-cancer': 32,  # 32 x 56 - 3
            'primary-tumor': 339,  # 339 x 17 - 21
            'mushroom': 8124,  # 8124 x 22 - 2
            'breast-cancer': 14,
            'german': 144,
            'heart': 45,
            'image':50,
            }
        
        le = sklearn.preprocessing.LabelEncoder()
        if self.dataset in self.dataset in ['mnist','kmnist','fmnist']:
            self.dataset = self.dataset.upper()
            # The standard train/test partition of this datasets will be considered 
            # but we will add the random partition
            if self.dataset == 'MNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
            elif self.dataset == 'KMNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1904,), (0.3475,))
                    ])
            elif self.dataset == 'FMNIST':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                    ])
            self.train_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=True, 
                transform=self.transform, 
                download=True)
            self.test_dataset = datasets.__dict__[self.dataset](
                root='Datasets/raw_datasets', 
                train=False, 
                transform=self.transform, 
                download=True)
            # full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
            # for the full with a random partition transforms mut be changed
            
            self.num_classes = len(np.unique(self.train_dataset.targets))
            self.num_features = self.train_dataset.data.shape[1]

            self.train_num_samples = self.train_dataset.data.shape[0]
            self.test_num_samples = self.test_dataset.data.shape[0]
            
            self.train_dataset.data = self.train_dataset.data.to(torch.float32).view((self.train_num_samples,-1))
            self.test_dataset.data = self.test_dataset.data.to(torch.float32).view((self.test_num_samples,-1))
            
            self.train_dataset.targets = self.test_dataset.data.to(torch.long)
            self.train_dataset.targets = self.test_dataset.data.to(torch.long)

        else: 
            if self.dataset in openml_ids:
                data = openml.datasets.get_dataset(openml_ids[self.dataset])
                X, y, categorical, feature_names = data.get_data(target=data.default_target_attribute)
                if any(categorical):
                    raise ValueError("TBD. For now, we don't handle categorical variables.")
                X = X.values
                y = le.fit_transform(y) #Tis encodes labels into classes [0,1,2,...,n_classes-1]
                X, y = sklearn.utils.shuffle(X, y, random_state = self.splitting_seed)
            elif self.dataset in uci_ids:
                data = fetch_ucirepo(id = uci_ids[self.dataset])
                if np.any(data.variables.type[1:]=='Categorical'):
                    raise ValueError("TBD. For now, we don't handle categorical variables.")
                X = data.data.features 
                y = data.data.targets
                X = X.values
                y = le.fit_transform(y)
                X, y = sklearn.utils.shuffle(X, y, random_state = self.splitting_seed)
            elif self.dataset == 'gmm':
                num_samples = 4000
                n_components = 4
                n_features = 3

                # Means and covariances
                means = np.array([
                    [0, 0, 0],
                    [3, 3, 3],
                    [0, -3, -3],
                    [3, 0, 0]
                    ])
                covariances = np.array([
                    3*np.eye(3),
                    1.5*np.eye(3),
                    3*np.eye(3),
                    4*np.eye(3)
                    ])

                # Mixture weights
                self.weights = np.array([0.1, 0.3, 0.5, 0.1])

                gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='full')
                gmm.weights_ = self.weights
                gmm.means_ = means
                gmm.covariances_ = covariances
                gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
                
                X, y = gmm.sample(n_samples=num_samples)

            elif self.dataset == 'hypercube':
                X, y = sklearn.datasets.make_classification(
                    n_samples=400, n_features=40, n_informative=40,
                    n_redundant=0, n_repeated=0, n_classes=4,
                    n_clusters_per_class=2,
                    weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
                    shift=0.0, scale=1.0, shuffle=True, random_state=None)
            
            elif self.dataset == 'blobs':
                X, y = sklearn.datasets.make_blobs(
                    n_samples=400, n_features=2, centers=20, cluster_std=2,
                    center_box=(-10.0, 10.0), shuffle=True, random_state=None)
            elif self.dataset == 'blobs2':
                X, y = sklearn.datasets.make_blobs(
                    n_samples=400, n_features=4, centers=10, cluster_std=1,
                    center_box=(-10.0, 10.0), shuffle=True, random_state=None)
            elif self.dataset in uci_ids:
                raise ValueError("TBD. We still dont support UCI datasets.") 

            self.num_classes = len(np.unique(y))
            self.num_features = X.shape[1]

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size = self.tr_size, random_state = self.splitting_seed)

            self.train_num_samples = X_train.shape[0]
            self.test_num_samples = X_test.shape[0]
            
            X_train = torch.from_numpy(X_train).to(torch.float32)
            X_test = torch.from_numpy(X_test).to(torch.float32)
            y_train = torch.from_numpy(y_train).to(torch.long)
            y_test = torch.from_numpy(y_test).to(torch.long)

            self.train_dataset = TensorDataset(X_train, y_train)
            self.test_dataset = TensorDataset(X_test, y_test)

            # This is done to mantain coherence between de datset classes
            self.train_dataset.data = self.train_dataset.tensors[0]
            self.train_dataset.targets = self.train_dataset.tensors[1]
            self.test_dataset.data = self.test_dataset.tensors[0]
            self.test_dataset.targets = self.test_dataset.tensors[1]

        #One hot encoding of the labels
        self.train_dataset.targets = torch.eye(self.num_classes)[self.train_dataset.targets]
        self.test_dataset.targets = torch.eye(self.num_classes)[self.test_dataset.targets]

    def __getitem__(self, index):
        if self.weak_labels is None:
            x = self.train_dataset.data[index]
            y = self.train_dataset.targets[index]
            return x, y
        else:
            x = self.train_dataset.data[index]
            w = self.weak_labels[index]
            y = self.train_dataset.targets[index]
            return x, w, y
        
    def get_dataloader(self, indices = None, weak_labels = None):
        '''
        weak_labels(str): 'weak', 'virtual' or None
        '''
        #Not sure ifindices is necessary. It works this way
        if indices is None:
            indices = torch.Tensor(list(range(len(self.train_dataset)))).to(torch.long)
        if weak_labels is None: 
        #(self.weak_labels is None) & (self.virtual_labels is None):
            tr_dataset = TensorDataset(self.train_dataset.data[indices],
                                    self.train_dataset.targets[indices])
        elif weak_labels == 'virtual':
            if self.virtual_labels is None:
                print('you must provide virtual labels via include_virtual()')
                self.train_loader = None
            else:
                tr_dataset = TensorDataset(self.train_dataset.data[indices], 
                                    self.virtual_labels[indices],
                                    self.train_dataset.targets[indices])
        elif weak_labels == 'weak':
            if self.weak_labels is None:
                print('you must provide weak labels via include_weak()')
                self.train_loader = None
            else:
                tr_dataset = TensorDataset(self.train_dataset.data[indices], 
                                    self.weak_labels[indices],
                                    self.train_dataset.targets[indices])

        self.train_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                        num_workers=0)
        self.test_loader = DataLoader(TensorDataset(
            self.test_dataset.data, self.test_dataset.targets
        ), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        return self.train_loader, self.test_loader
    
    def get_data(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        test_x = self.test_dataset.data
        test_y = self.test_dataset.targets

        return train_x, train_y, test_x, test_y
    
    def include_weak(self, z):
        if torch.is_tensor(z):
            self.weak_labels = z
        else:
            self.weak_labels = torch.from_numpy(z)
            
    def include_virtual(self, vy):
        if torch.is_tensor(vy):
            self.virtual_labels = vy
        else:
            self.virtual_labels = torch.from_numpy(vy)