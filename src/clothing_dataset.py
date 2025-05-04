import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

metadata_dir = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/'
clean_label_inst = os.path.join(metadata_dir, 'clean_label_kv.txt') # - Path, label for clean images 
noisy_label_inst = os.path.join(metadata_dir, 'noisy_label_kv.txt') # - Path, label for noisy images
clean_train_paths = os.path.join(metadata_dir, 'clean_train_key_list.txt') # - Path to the list of clean training images
noisy_train_paths = os.path.join(metadata_dir, 'noisy_train_key_list.txt') # - Path to the list of noisy training images
#clean_val_instances = os.path.join(metadata_dir, 'clean_val_key_list.txt')
clean_test_paths = os.path.join(metadata_dir, 'clean_test_key_list.txt') # - Path to the list of clean test images
category_names_eng = os.path.join(metadata_dir, 'category_names_eng.txt') # Cathegory names in English

def load_instances(filepath):
    '''Load labels from a file.
    Args:
        filepath (str): Path to the label file.
    Returns:
        dict: A dictionary mapping image paths to labels.
    '''
    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_path = os.path.normpath(parts[0])
            labels[image_path] = int(parts[1])

    return labels

def load_paths(filepath):
    '''Load key list from a file.
    Args:
        filepath (str): Path to the key list file.
    Returns:
        list: A list of image paths.
    '''
    with open(filepath, 'r') as f:
        image_paths = [os.path.normpath(line.strip()) for line in f]
    return image_paths

def load_category_names(filepath):
    '''Load category names from a file.
    Args:
        filepath (str): Path to the category names file.
    Returns:
        list: A list of category names.
    '''
    with open(filepath, 'r') as f:
        category_names = [line.strip() for line in f]
    print(category_names)
    return category_names

class ClothingDataset(Dataset):
    def __init__(self, direction, train = "True", transform=None):
        self.direction = direction

        self.train = train
        self.transform = transform
        self.N = None

        self.c = len(load_category_names(category_names_eng))

        self.samples = []

        clean_instances = load_instances(clean_label_inst)
        noisy_instances = load_instances(noisy_label_inst)
        print(f"Loaded {len(clean_instances)} clean instances and {len(noisy_instances)} noisy instances.")

        num_clean = 0
        if self.train == "True":
            self.N = np.zeros((self.c, self.c))
            clean_train = load_paths(clean_train_paths)
            noisy_train = load_paths(noisy_train_paths)
            for key in noisy_train:
                if key in noisy_instances:
                    self.samples.append((key, noisy_instances[key]))
                    if key in clean_instances:
                        self.N[noisy_instances[key], clean_instances[key]] += 1
                        del clean_train[key]
            print(f"Loaded {len(noisy_train)} noisy training instances")
            for key in clean_train:
                self.samples.append((key, clean_instances[key]))
                num_clean += 1
            print(f"Loaded {num_clean} clean training instances")
            

        else:
            clean_test = load_paths(clean_test_paths)
            for key in clean_test:
                self.samples.append((key, clean_instances[key]))
            print(f"Loaded {len(clean_test)} clean test instances")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        relative_img_path, label = self.samples[idx]
        img_full_path = os.path.join(self.direction, relative_img_path)
        image = Image.open(img_full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

img_size = 224
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

clothing1m_root = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/' 

train_dataset = ClothingDataset(clothing1m_root, train = "True", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataset = ClothingDataset(clothing1m_root, train = "False", transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)   