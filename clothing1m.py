from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


train_data_list = []
for element in clean_label_kv:
    path, label = element.split()
    if path in clean_train_key_list:
        # Assign a noisy flag as needed (example: 0)
        noisy_flag = 0
        data_list.append((path, int(label), noisy_flag))
for element in noisy_label_kv:
    path, label = element.split()
    if path in noisy_train_key_list:
        # Assign a noisy flag as needed (example: 1)
        noisy_flag = 1
        data_list.append((path, int(label), noisy_flag))

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None, train = True):
        self.data_list = data_list
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label, noisy_flag = self.data_list[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, noisy_flag

#This are the standatd transformations for imagenet
transform = transforms.Compose([
    transforms.Resize(256),            # Resize so the shorter side is 256
    transforms.CenterCrop(224),        # Crop the center 224×224 region
    transforms.ToTensor(),             # Convert PIL image to PyTorch tensor
    transforms.Normalize(              # Normalize channels to ImageNet mean/std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = CustomDataset(train_data_list, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for imgs, labels, noisy_flags in train_dataloader:
    print("Images:", imgs.shape)
    print("Labels:", labels)
    print("Noisy flags:", noisy_flags)
    break