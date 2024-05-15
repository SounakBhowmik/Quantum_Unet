# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:47:11 2024

@author: sbhowmi2
"""
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    val_losses = []
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
        val_losses.append(epoch_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}')
    return val_losses