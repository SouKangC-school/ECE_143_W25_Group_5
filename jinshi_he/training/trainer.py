"""
Training utilities for neural network models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from ..utils.metrics import calculate_metrics

def train_model_optimized(model, train_loader, val_loader=None, optimizer=None, criterion=None,
                         num_epochs=20, patience=5, device=torch.device('cpu'), learning_rate=0.001):
    """
    Train a PyTorch model with early stopping and mixed precision training if available.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        optimizer: Optimizer to use (default: Adam)
        criterion: Loss function to use (default: MSE)
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        device: Device to train on (cpu or cuda)
        learning_rate: Learning rate for optimizer (if optimizer is not provided)
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Move model to device
    model.to(device)
    
    # Set up optimizer and criterion if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Check if mixed precision training is available
    use_amp = device.type == 'cuda'
    if use_amp:
        print("Enabling mixed precision training")
        scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_epochs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update running loss
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase - Only run every 2 epochs to save time or if val_loader is provided
        if (val_loader is not None) and (epoch % 2 == 0 or epoch == num_epochs - 1):
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Update running loss
                    val_loss += loss.item() * inputs.size(0)
            
            # Calculate average validation loss
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            val_epochs.append(epoch)
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            # Just print training loss
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {"train_losses": train_losses, "val_losses": val_losses, "val_epochs": val_epochs}

def evaluate_model(model, data_loader, criterion=None, device=torch.device('cpu')):
    """
    Evaluate a PyTorch model on a dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        criterion: Loss function to use (default: MSE)
        device: Device to evaluate on (cpu or cuda)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
            # Store targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    # Calculate metrics
    all_targets = np.array(all_targets).flatten()
    all_predictions = np.array(all_predictions).flatten()
    
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = running_loss / len(data_loader.dataset)
    metrics['targets'] = all_targets
    metrics['predictions'] = all_predictions
    
    return metrics

def quick_evaluate(model, data_loader, criterion=None, max_batches=50, device=torch.device('cpu')):
    """
    Quickly evaluate a PyTorch model on a subset of data.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        criterion: Loss function to use (default: MSE)
        max_batches: Maximum number of batches to evaluate
        device: Device to evaluate on (cpu or cuda)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= max_batches:
                break
            
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
            # Store targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    # Calculate metrics
    all_targets = np.array(all_targets).flatten()
    all_predictions = np.array(all_predictions).flatten()
    
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = running_loss / len(all_targets) if len(all_targets) > 0 else 0
    metrics['targets'] = all_targets
    metrics['predictions'] = all_predictions
    
    return metrics

def train_evaluate_plot(model, model_name, train_loader, val_loader=None, learning_rate=0.001,
                       num_epochs=100, patience=10, device=torch.device('cpu'),
                       visualize_fn=None):
    """
    Train a model, evaluate it, and plot the results.
    
    Args:
        model: PyTorch model to train
        model_name: Name of the model for display
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        learning_rate: Learning rate for optimizer
        num_epochs: Maximum number of epochs to train
        patience: Early stopping patience
        device: Device to train on (cpu or cuda)
        visualize_fn: Function to visualize results (optional)
        
    Returns:
        Tuple of (trained_model, val_results)
    """
    print(f"\n===== Training {model_name} Model =====")
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train the model
    trained_model, history = train_model_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        patience=patience,
        device=device
    )
    
    # Evaluate on validation set
    if val_loader is not None:
        print(f"Evaluating {model_name} on validation set...")
        val_results = evaluate_model(trained_model, val_loader, criterion, device=device)
        
        print(f"{model_name} Validation Results:")
        print(f"  RMSE: {val_results['rmse']:.4f}")
        print(f"  MAE: {val_results['mae']:.4f}")
        print(f"  R²: {val_results['r2']:.4f}")
    else:
        val_results = None
    
    # Visualize results if function is provided
    if visualize_fn is not None and callable(visualize_fn):
        visualize_fn(model_name, history, val_results)
    
    return trained_model, val_results
