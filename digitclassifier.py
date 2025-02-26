# -*- coding: utf-8 -*-
#Early Stopping (Stops training if accuracy doesn’t improve for 3 consecutive epochs)
#Option: data augmentation of training data (Yes or No)
#Optimized Training Strategy (AdamW, StepLR for stable learning rate decay)
#Computes Classification Accuracy, TPR, and TNR per Class
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

# Training function
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=20, early_stop_patience=3):
    model.train()
    best_accuracy = 0.0
    patience_counter = 0

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

        # Early stopping condition
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} with best accuracy: {best_accuracy:.2f}%")
            break

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Compute overall accuracy
    accuracy = 100 * correct / total

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Compute TPR and TNR for each class
    tpr = np.diag(cm) / np.sum(cm, axis=1)  # True Positive Rate
    tnr = (np.sum(cm) - np.sum(cm, axis=1) - np.sum(cm, axis=0) + np.diag(cm)) / (np.sum(cm) - np.sum(cm, axis=1))

    print(f"Test Accuracy: {accuracy:.2f}%\n")
    for i in range(10):
        print(f"Class {i}: TPR = {tpr[i]:.4f}, TNR = {tnr[i]:.4f}")

    return accuracy, tpr, tnr

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Define transformations (NO augmentation)
#transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
#    transforms.ToTensor(),
#    transforms.Normalize((0.5,), (0.5,))
#])
def transform_data(data_augmentation):
    if data_augmentation =='Yes':
      # Data augmentation and transformations
      train_transform = transforms.Compose([
          transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
          transforms.RandomRotation(10),  # Rotate images slightly
          transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
          transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),  # Random scaling
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))  # Normalize like during training
      ])
    else:#NO augmentation
      train_transform = transforms.Compose([
          transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))  # Normalize like during training
      ])
     
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return train_transform, test_transform

#data_augmentation='Yes'
data_augmentation='No'

print(F"data augmentation: '{data_augmentation}'\n")

train_transform, test_transform = transform_data(data_augmentation)

if data_augmentation=='Yes':
   model_path="./model/resnet18_mnist_data_augmentation.pth"
else:
   model_path="./model/resnet18_mnist.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# Modify ResNet-18 for MNIST
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust for MNIST input size
model.fc = nn.Linear(model.fc.in_features, 10)  # Modify final layer for 10 classes
model = model.to(device)

# Freeze all layers except the last few layers (optional)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the last few layers
for param in model.layer4.parameters():  # Last convolutional block
    param.requires_grad = True

for param in model.fc.parameters():  # Fully connected layer
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train and evaluate
train(model, train_loader, criterion, optimizer, scheduler, num_epochs=20)
evaluate(model, test_loader)

# Save the trained model

save_model(model,model_path)
