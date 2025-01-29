import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Clothing1MDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the Clothing1M dataset.
            mode (str): Dataset mode. Options: 'train', 'val', 'test'.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.annotations_file = self._get_annotations_file()
        self.image_paths, self.labels = self._load_annotations()

    def _get_annotations_file(self):
        """Get the path to the annotations file based on the mode."""
        if self.mode == 'train':
            return os.path.join(self.root_dir, 'annotations', 'noisy_train.txt')
        elif self.mode == 'val':
            return os.path.join(self.root_dir, 'annotations', 'clean_val.txt')
        elif self.mode == 'test':
            return os.path.join(self.root_dir, 'annotations', 'clean_test.txt')
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'val', 'test'.")

    def _load_annotations(self):
        """Load image paths and labels from the annotations file."""
        image_paths = []
        labels = []
        with open(self.annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path = os.path.join(self.root_dir, 'images', parts[0])
                    label = int(parts[1])
                    image_paths.append(img_path)
                    labels.append(label)
        return image_paths, labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its label by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            image (PIL.Image): The image.
            label (int): The label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and data loader
root_dir = './data/Clothing1M'
train_dataset = Clothing1MDataset(root_dir, mode='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)