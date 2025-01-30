import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




class Clothing1MDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            mode (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Set the annotation file based on the mode
        if mode == 'train':
            self.annotations_file = os.path.join(root_dir, 'clean_train_key_list.txt')
            self.noisy_annotations_file = os.path.join(root_dir, 'noisy_train_key_list.txt')
        elif mode == 'val':
            self.annotations_file = os.path.join(root_dir, 'clean_val_key_list.txt')
            self.noisy_annotations_file = None  # No noisy labels for validation
        elif mode == 'test':
            self.annotations_file = os.path.join(root_dir, 'clean_test_key_list.txt')
            self.noisy_annotations_file = None  # No noisy labels for testing
        else:
            raise ValueError("Mode must be one of 'train', 'val', or 'test'.")

        # Load the annotations
        self.image_paths, self.labels, self.noisy_flags = self._load_annotations()

    def _load_annotations(self):
        """
        Load image paths, labels, and noisy flags from the annotation files.
        """
        image_paths = []
        labels = []
        noisy_flags = []

        # Load clean labels
        clean_label_kv_file = os.path.join(self.root_dir, 'clean_label_kv.txt')
        with open(clean_label_kv_file, 'r') as f:
            clean_label_kv = {line.split()[0]: int(line.split()[1]) for line in f.read().splitlines()}

        # Load noisy labels (if available)
        noisy_label_kv_file = os.path.join(self.root_dir, 'noisy_label_kv.txt')
        if os.path.exists(noisy_label_kv_file):
            with open(noisy_label_kv_file, 'r') as f:
                noisy_label_kv = {line.split()[0]: int(line.split()[1]) for line in f.read().splitlines()}
        else:
            noisy_label_kv = {}  # If noisy_label_kv.txt doesn't exist, use an empty dictionary

        # Load clean samples
        with open(self.annotations_file, 'r') as f:
            for line in f.read().splitlines():
                image_path = os.path.join(self.root_dir, 'images', line)
                if os.path.exists(image_path):  # Check if the image file exists
                    image_paths.append(image_path)
                    labels.append(clean_label_kv[line])
                    noisy_flags.append(0)  # 0 indicates a clean label
                else:
                    print(f"Warning: Image file {image_path} not found. Skipping.")

        # Load noisy samples (only for training)
        if self.noisy_annotations_file and os.path.exists(self.noisy_annotations_file):
            with open(self.noisy_annotations_file, 'r') as f:
                for line in f.read().splitlines():
                    image_path = os.path.join(self.root_dir, 'images', line)
                    if os.path.exists(image_path):  # Check if the image file exists
                        image_paths.append(image_path)
                        labels.append(noisy_label_kv.get(line, clean_label_kv.get(line, -1)))  # Fallback to clean label if noisy label is missing
                        noisy_flags.append(1)  # 1 indicates a noisy label
                    else:
                        print(f"Warning: Image file {image_path} not found. Skipping.")

        return image_paths, labels, noisy_flags

'''

class Clothing1MDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, use_noisy_labels=False):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            mode (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_noisy_labels (bool): If True, use noisy labels instead of clean labels.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.use_noisy_labels = use_noisy_labels

        # Set the annotation file based on the mode
        if mode == 'train':
            self.annotations_file = os.path.join(root_dir, 'clean_train_key_list.txt')
        elif mode == 'val':
            self.annotations_file = os.path.join(root_dir, 'clean_val_key_list.txt')
        elif mode == 'test':
            self.annotations_file = os.path.join(root_dir, 'clean_test_key_list.txt')
        else:
            raise ValueError("Mode must be one of 'train', 'val', or 'test'.")

        # Load the annotations
        self.image_paths, self.clean_labels, self.noisy_labels = self._load_annotations()

    def _load_annotations(self):
        """
        Load image paths, clean labels, and noisy labels from the annotation files.
        """
        image_paths = []
        clean_labels = []
        noisy_labels = []

        # Load clean labels
        clean_label_kv_file = os.path.join(self.root_dir, 'clean_label_kv.txt')
        with open(clean_label_kv_file, 'r') as f:
            clean_label_kv = {line.split()[0]: int(line.split()[1]) for line in f.read().splitlines()}

        # Load noisy labels
        noisy_label_kv_file = os.path.join(self.root_dir, 'noisy_label_kv.txt')
        with open(noisy_label_kv_file, 'r') as f:
            noisy_label_kv = {line.split()[0]: int(line.split()[1]) for line in f.read().splitlines()}

        # Load the list of image filenames
        with open(self.annotations_file, 'r') as f:
            for line in f.read().splitlines():
                image_path = os.path.join(self.root_dir, 'images', line)
                if os.path.exists(image_path):  # Check if the image file exists
                    image_paths.append(image_path)
                    clean_labels.append(clean_label_kv[line])
                    noisy_labels.append(noisy_label_kv[line])
                else:
                    print(f"Warning: Image file {image_path} not found. Skipping.")

        return image_paths, clean_labels, noisy_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        clean_label = self.clean_labels[idx]
        noisy_label = self.noisy_labels[idx]

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return clean or noisy label based on the flag
        if self.use_noisy_labels:
            return image, noisy_label
        else:
            return image, clean_label




# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # Create datasets
train_dataset = Clothing1MDataset(data_root, mode='train', transform=transform, use_noisy_labels=False)
val_dataset = Clothing1MDataset(data_root, mode='val', transform=transform, use_noisy_labels=False)
test_dataset = Clothing1MDataset(data_root, mode='test', transform=transform, use_noisy_labels=False)

    # Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Verify data loading
for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")  # Should be [batch_size, 3, 256, 256]
    print(f"Batch of labels: {labels}")             # Should be a tensor of labels
    break
'''