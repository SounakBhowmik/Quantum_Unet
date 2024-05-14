#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 23:52:59 2024

@author: sounakbhowmik
"""
#%% Imports
from get_data import get_data
from Utils import standardize_data, visualize
from training_script import train_model
from unet_model import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#%% Define variables, data, optimizers for training
# Assuming X and y are your dataset's features and labels, respectively
X, y = get_data()

X = standardize_data(X)
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

# Define number of epochs
num_epochs = 1

# Run the training function
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

#%% Visualize the results and original image
visualize(model(X_val[20:21])[0])
visualize(X_val[20])



