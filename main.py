import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from datetime import timedelta
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations with augmentation
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize data (optional)
])

# Load datasets with transformations
train_data = datasets.ImageFolder(root=r"C:/Users/Dhanush/OneDrive/Desktop/anamoly/Train_ana", transform=data_transform)
val_data = datasets.ImageFolder(root=r"C:/Users/Dhanush/OneDrive/Desktop/anamoly/Val_ana", transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize data (optional)
]))
test_data = datasets.ImageFolder(root=r"C:/Users/Dhanush/OneDrive/Desktop/anamoly/Test_ana", transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize data (optional)
]))

# Data loaders
train_dataloader = DataLoader(dataset=train_data, batch_size=8, pin_memory=True, drop_last=True, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=8, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=8, shuffle=False)

# Define the CAE model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the model and send it to the appropriate device
model = CAE().to(device)
# Function to load model state dict with filtering
def load_filtered_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

# Load the checkpoint (if any)
checkpoint_path = None  # Replace with your checkpoint path if any
if checkpoint_path:
    try:
        checkpoint = torch.load(checkpoint_path)
        load_filtered_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded model state dict from checkpoint.')
    except FileNotFoundError:
        print('Checkpoint file not found. Starting training from scratch.')

# Define loss function and optimizer with weight decay for regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # You can adjust learning rate and weight decay

# Set number of epochs
epochs = 20
# Training loop
metrics = {'train_loss': [], 'val_loss': []}
model.train()
start = time.time()
for epoch in range(epochs):
    ep_start = time.time()
    running_train_loss = 0.0
    running_val_loss = 0.0

    # Training phase
    for images, _ in train_dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed_images = model(images)
        loss = criterion(images, reconstructed_images)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # Compute average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_dataloader)
    metrics['train_loss'].append(epoch_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, _ in val_dataloader:
            images = images.to(device)
            reconstructed_images = model(images)
            val_loss = criterion(images, reconstructed_images)
            running_val_loss += val_loss.item()

    # Compute average validation loss for the epoch
    epoch_val_loss = running_val_loss / len(val_dataloader)
    metrics['val_loss'].append(epoch_val_loss)

    # Print epoch statistics
    print(f"[EPOCH] {epoch + 1}/{epochs}")
    print(f"[TRAIN LOSS] {epoch_train_loss:.4f} [VAL LOSS] {epoch_val_loss:.4f}")

    model.train()
    ep_end = time.time()
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end - ep_start)))

end = time.time()
print()
print('[System Complete: {}]'.format(timedelta(seconds=end - start)))

# Save the model
torch.save(model.state_dict(), 'cae_model.pth')

# Evaluation loop
model.eval()
test_loss = 0.0
reconstructions = []
original_images = []

with torch.no_grad():
    for images, _ in test_dataloader:
        images = images.to(device)
        reconstructed = model(images)
        test_loss += criterion(images, reconstructed).item()  # Accumulate test loss

        # Convert tensors to numpy arrays for visualization (optional)
        original_images.append(images.cpu().numpy())
        reconstructions.append(reconstructed.cpu().numpy())

# Calculate average test loss
test_loss /= len(test_dataloader)
print(f'Test Loss: {test_loss:.4f}')

print('Evaluation Complete!')
