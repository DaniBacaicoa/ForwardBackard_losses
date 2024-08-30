import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers here, e.g., self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        # Define forward pass here
        return x

# Define a function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy

# Define a function to test the model
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy

# Hyperparameters
num_epochs = 10
learning_rate = 0.001
repetitions = 3

# Assuming you have train_loader and test_loader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create a device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DataFrame to store results
columns = ['epoch', 'repetition', 'lr', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']
results_df = pd.DataFrame(columns=columns)

# Training loop
for rep in range(repetitions):
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test_model(model, test_loader, criterion, device)

        # Store results in DataFrame
        results_df = results_df.append({
            'epoch': epoch + 1,
            'repetition': rep + 1,
            'lr': learning_rate,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, ignore_index=True)

# Save DataFrame to CSV
results_df.to_csv('training_results.csv', index=False)
