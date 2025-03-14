"""
Dataset classes for model training.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class HeartRateDataset(Dataset):
    """
    PyTorch Dataset for heart rate prediction.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X: Input features (shape: [n_samples, seq_length, n_features])
            y: Target values (shape: [n_samples])
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # reshape to [batch_size, 1]

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, target)
        """
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, batch_size=128):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        X_test: Test features (optional)
        y_test: Test targets (optional)
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary of DataLoaders
    """
    loaders = {}
    
    # Create training DataLoader
    train_dataset = HeartRateDataset(X_train, y_train)
    loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation DataLoader if validation data is provided
    if X_val is not None and y_val is not None:
        val_dataset = HeartRateDataset(X_val, y_val)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create test DataLoader if test data is provided
    if X_test is not None and y_test is not None:
        test_dataset = HeartRateDataset(X_test, y_test)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size)
    
    return loaders
