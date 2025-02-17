#train ResNet-18 on the MNIST dataset while applying data augmentation techniques.
#The augmentations include:
#Random Rotation (Small angle variations)
#Random Translation (Shifts the image slightly)
#Random Resized Crop (Scaling variations)
#Random Horizontal Flip (Though MNIST digits aren't usually flipped, this is optional)
#Normalization (Standardize the pixel values)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
    transforms.RandomRotation(10),  # Rotate images slightly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),  # Random scaling
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize like during training
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data_augmentations', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data_augmentations', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# Modify ResNet-18 for MNIST
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust for MNIST input size
model.fc = nn.Linear(model.fc.in_features, 10)  # Modify final layer for 10 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Apply label smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

# Training function with augmentation
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  # Update learning rate

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Train and evaluate
train(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)
evaluate(model, test_loader)

# Save the trained model
save_model(model, "./model/resnet18_mnist_augmentations.pth")