"""
Configuration settings for the workout analysis project.
"""
import torch

# General settings
RANDOM_SEED = 42

# Data settings
DATA_PATH = "data_shrunk.npy"

# Feature settings
FEATURES = ['heart_rate', 'derived_speed', 'distance']
TARGET = 'heart_rate'

# Sequence settings
SEQUENCE_LENGTH = 10
PREDICTION_HORIZON = 3

# Training settings
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10

# Sampling parameters
SAMPLE_RATIO = 0.1
MAX_USERS = 50
MAX_SEQUENCES_PER_USER = 50

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Visualization settings
PLOT_STYLE = 'ggplot'
FIGURE_SIZE = [12, 8]
FONT_SIZE = 12
