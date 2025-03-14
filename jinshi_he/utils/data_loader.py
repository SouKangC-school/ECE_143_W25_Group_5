"""
Data loading utilities for workout data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load workout data from a numpy file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded workout data
    """
    data = np.load(file_path, allow_pickle=True)
    print(f"Dataset loaded with {len(data)} workouts")
    return data

def convert_to_dataframe(all_workouts):
    """
    Convert processed workouts to a pandas DataFrame.
    
    Args:
        all_workouts: List of processed workout dictionaries
        
    Returns:
        pandas DataFrame with workout data
    """
    workouts_df = pd.DataFrame(all_workouts)
    print("\nWorkout data sample:")
    print(workouts_df.head())
    
    # Basic statistics of the dataset
    print("\nBasic statistics of numeric columns:")
    print(workouts_df.describe())
    
    return workouts_df

def prepare_training_data(X, y, test_size=0.2, val_size=0.0, random_state=42, normalize=True):
    """
    Split data into training, validation, and optionally test sets.
    
    Args:
        X: Feature sequences
        y: Target values
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        normalize: Whether to normalize the target values
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        If val_size is 0, X_val and y_val will be None
    """
    # First split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # If validation size is specified, split training data into train and validation
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for previous split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, random_state=random_state
        )
    else:
        X_val, y_val = None, None
    
    # Normalize target values if needed
    if normalize and np.max(np.abs(y)) > 100:  # Assuming normalized values are smaller
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        if y_val is not None:
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
        if y_test is not None:
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
        print("Scaled target values")
    else:
        scaler_y = None
    
    # Print dataset sizes
    print(f"Train set: {X_train.shape[0]} sequences")
    if X_val is not None:
        print(f"Validation set: {X_val.shape[0]} sequences")
    if X_test is not None:
        print(f"Test set: {X_test.shape[0]} sequences")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y
