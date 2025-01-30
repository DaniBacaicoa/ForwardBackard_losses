import numpy as np
from torch import nn, optim
from torchvision.models import resnet50

def estimate_transition_matrix(model, data_loader, num_classes):
    """
    Estimate the transition matrix T using the trained model.
    
    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): Data loader for the dataset.
        num_classes (int): Number of classes.
    
    Returns:
        T (np.ndarray): Estimated transition matrix of shape (num_classes, num_classes).
    """
    model.eval()
    T = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for pred, label in zip(preds, labels.cpu().numpy()):
                T[pred, label] += 1
    # Normalize the transition matrix
    T = T / T.sum(axis=1, keepdims=True)
    return T

'''
# Initialize model
num_classes = 14  # Clothing1M has 14 classes
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.cuda()

# Train the model (simplified training loop)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

for epoch in range(10):  # Train for 10 epochs
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Estimate the transition matrix
T = estimate_transition_matrix(model, train_loader, num_classes)
print("Estimated Transition Matrix T:\n", T)
'''