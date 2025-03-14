"""
Training modules for machine learning models.
"""
from .dataset import HeartRateDataset, create_data_loaders
from .trainer import (
    train_model_optimized,
    evaluate_model,
    quick_evaluate,
    train_evaluate_plot
)

__all__ = [
    'HeartRateDataset',
    'create_data_loaders',
    'train_model_optimized',
    'evaluate_model',
    'quick_evaluate',
    'train_evaluate_plot'
]
