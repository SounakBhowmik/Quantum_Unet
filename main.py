#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:52:59 2024

@author: sounakbhowmik
"""

from get_data import *
from unet_model import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

def standardize_data(X):
    X_standardized = (X-np.min(X))/(np.max(X) - np.min(X))
    
    return X_standardized
#%%
# Assuming X and y are your dataset's features and labels, respectively
X, y = get_data()


# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create TensorDatasets for both training and validation
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create DataLoaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Initialize the model
model = UNet(3,1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Train on batches
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model

            running_loss += loss.item() * inputs.size(0)

        # Calculate and print average loss over the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        # Calculate and print the validation loss
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}')

# Run the training function
train_model(model, train_loader, val_loader, criterion, optimizer)
