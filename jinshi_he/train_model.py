"""
Main script for training the heart rate prediction model.
"""
import argparse
import numpy as np
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.preprocessing import prepare_sequences_optimized
from utils.data_loader import prepare_training_data
from training.dataset import create_data_loaders
from training.trainer import train_evaluate_plot
from utils.visualization import plot_training_history, plot_predictions_vs_actual
from models import (
    LinearModel,
    SimpleRNNModel,
    SimpleLSTMModel,
    BiLSTMModel,
    AttentionLSTM,
    CNNLSTMModel,
    TransformerModel
)
import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train heart rate prediction models')
    
    # Data parameters
    parser.add_argument('--data', type=str, default=config.DATA_PATH,
                        help='Path to data file')
    parser.add_argument('--features', type=str, nargs='+', default=config.FEATURES,
                        help='Features to use for prediction')
    parser.add_argument('--target', type=str, default=config.TARGET,
                        help='Target variable to predict')
    
    # Sequence parameters
    parser.add_argument('--seq_length', type=int, default=config.SEQUENCE_LENGTH,
                        help='Length of input sequences')
    parser.add_argument('--pred_horizon', type=int, default=config.PREDICTION_HORIZON,
                        help='How many steps ahead to predict')
    
    # Sampling parameters
    parser.add_argument('--sample_ratio', type=float, default=config.SAMPLE_RATIO,
                        help='Ratio of data to sample')
    parser.add_argument('--max_users', type=int, default=config.MAX_USERS,
                        help='Maximum number of users to include')
    parser.add_argument('--max_seqs', type=int, default=config.MAX_SEQUENCES_PER_USER,
                        help='Maximum sequences per user')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=config.PATIENCE,
                        help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['linear', 'rnn', 'lstm', 'bilstm', 'attlstm', 'cnnlstm', 'transformer'],
                        help='Model architecture to train')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size for recurrent models')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for models and results')
    
    return parser.parse_args()

def create_model(model_name, input_size, seq_length, hidden_size=64):
    """Create a model based on the provided name."""
    if model_name == 'linear':
        return LinearModel(input_size=input_size, seq_length=seq_length)
    elif model_name == 'rnn':
        return SimpleRNNModel(input_size=input_size, hidden_size=hidden_size)
    elif model_name == 'lstm':
        return SimpleLSTMModel(input_size=input_size, hidden_size=hidden_size)
    elif model_name == 'bilstm':
        return BiLSTMModel(input_size=input_size, hidden_size=hidden_size)
    elif model_name == 'attlstm':
        return AttentionLSTM(input_size=input_size, hidden_size=hidden_size)
    elif model_name == 'cnnlstm':
        return CNNLSTMModel(input_size=input_size, seq_length=seq_length, hidden_size=hidden_size)
    elif model_name == 'transformer':
        return TransformerModel(input_size=input_size, seq_length=seq_length)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def visualize_results(model_name, history, val_results):
    """Visualize training history and validation results."""
    # Plot training history
    plot_training_history(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        val_epochs=history.get('val_epochs'),
        model_name=model_name
    )
    
    # Plot predictions vs actual values
    if val_results is not None:
        plot_predictions_vs_actual(
            targets=val_results['targets'],
            predictions=val_results['predictions'],
            model_name=model_name
        )

def main():
    """Main function for training the model."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    
    # Prepare sequences
    print("Preparing sequences...")
    X, y = prepare_sequences_optimized(
        data=data,
        seq_length=args.seq_length,
        pred_horizon=args.pred_horizon,
        features=args.features,
        target=args.target,
        sample_ratio=args.sample_ratio,
        max_users=args.max_users,
        max_sequences_per_user=args.max_seqs
    )
    
    # Split data into train and validation sets
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_training_data(
        X=X,
        y=y,
        test_size=0.2,
        val_size=0.1,
        random_state=config.RANDOM_SEED
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=args.batch_size
    )
    
    # Create model
    input_size = X_train.shape[2]  # Number of features
    model = create_model(
        model_name=args.model,
        input_size=input_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size
    )
    print(f"Created {args.model.upper()} model with {model.get_model_size()} trainable parameters")
    
    # Train and evaluate model
    trained_model, val_results = train_evaluate_plot(
        model=model,
        model_name=args.model.upper(),
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        device=config.DEVICE,
        visualize_fn=visualize_results
    )
    
    # Save model
    model_path = output_dir / f"{args.model}_model.pt"
    trained_model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on test set if available
    if 'test' in data_loaders:
        from training.trainer import evaluate_model
        
        print("Evaluating on test set...")
        test_results = evaluate_model(
            model=trained_model,
            data_loader=data_loaders['test'],
            device=config.DEVICE
        )
        
        print("Test set results:")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")
        print(f"  R²: {test_results['r2']:.4f}")
        
        # Save test results
        results_path = output_dir / f"{args.model}_test_results.txt"
        with open(results_path, 'w') as f:
            f.write(f"Model: {args.model.upper()}\n")
            f.write(f"Test set results:\n")
            f.write(f"  RMSE: {test_results['rmse']:.4f}\n")
            f.write(f"  MAE: {test_results['mae']:.4f}\n")
            f.write(f"  R²: {test_results['r2']:.4f}\n")
        
        print(f"Test results saved to {results_path}")

if __name__ == "__main__":
    main()
