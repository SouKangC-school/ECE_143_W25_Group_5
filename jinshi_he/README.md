# Workout Categorical Analysis and Heart Rate Prediction by Jinshi He

This project provides tools for analyzing workout data and training models to predict heart rate based on time series data.

## Project Structure

```
jinshi_he/
├── notebooks/                # Original Jupyter notebooks
│   ├── categorical_data_analysis.ipynb   # Original workout analysis notebook
│   └── LSTM_model.ipynb      # Original LSTM model notebook
├── data/                     # Data storage directory
├── models/                   # Neural network model architectures
│   ├── base_model.py         # Base model and linear model
│   ├── lstm_models.py        # LSTM-based models
│   ├── rnn_models.py         # RNN-based models
│   ├── transformer_models.py # Transformer models
│   └── cnn_models.py         # CNN-based models
├── utils/                    # Utility functions
│   ├── data_loader.py        # Data loading and preparation
│   ├── preprocessing.py      # Data preprocessing
│   ├── visualization.py      # Visualization utilities
│   └── metrics.py            # Evaluation metrics
├── training/                 # Model training utilities
│   ├── trainer.py            # Training and evaluation functions
│   └── dataset.py            # PyTorch dataset classes
├── analysis/                 # Data analysis
│   ├── workout_analysis.py   # Workout data analysis
│   └── heart_rate_analysis.py # Heart rate-specific analysis
├── config.py                 # Configuration settings
├── run_analysis.py           # Main script for running analysis
└── train_model.py            # Main script for training models
```

## Original Notebooks

The project includes the original Jupyter notebooks used for development:

- **categorical_data_analysis.ipynb**: Contains the complete workout data analysis with visualizations
- **LSTM_model.ipynb**: Contains the development and training of LSTM models for heart rate prediction

These notebooks provide a comprehensive, interactive view of the analysis and modeling process with visualizations and step-by-step explanations. They serve as both documentation and a demonstration of the work done before modularization.

## Modular Code Structure

The project has been refactored into a modular structure for better maintainability and extensibility. The functionality from the original notebooks has been preserved in the modular code but organized into logical components.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter (for running the notebooks)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm jupyter
   ```

### Running the Notebooks

To run the original notebooks with all visualizations:

```bash
jupyter notebook notebooks/categorical_data_analysis.ipynb
jupyter notebook notebooks/LSTM_model.ipynb
```

### Running Analysis

To run the workout data analysis using the modular code:

```bash
python run_analysis.py --data path/to/data_shrunk.npy --analysis categorical intensity zones
```

Available analysis types:
- `categorical`: Basic categorical analysis of workouts
- `intensity`: Workout intensity analysis
- `trends`: Workout trends over time
- `zones`: Heart rate zone analysis
- `trimp`: Training load (TRIMP) analysis
- `altitude`: Altitude impact on heart rate
- `all`: Run all analyses

### Training Models

To train a heart rate prediction model using the modular code:

```bash
python train_model.py --model lstm --data path/to/data_shrunk.npy
```

Available model types:
- `linear`: Simple linear model (baseline)
- `rnn`: RNN model
- `lstm`: LSTM model
- `bilstm`: Bidirectional LSTM model
- `attlstm`: LSTM with attention
- `cnnlstm`: CNN-LSTM hybrid model
- `transformer`: Transformer model

## Data Format

The input data should be a NumPy array of workout dictionaries with the following fields:
- `userId`: User identifier
- `sport`: Sport type
- `gender`: User gender
- `tar_heart_rate`: Heart rate time series
- `tar_derived_speed`: Speed time series
- `distance`: Distance time series
- Other fields like `timestamp`, `altitude`, etc.

## Examples

### Running the Original Analysis

For the complete analysis with all visualizations, run the original notebooks:

```bash
jupyter notebook notebooks/categorical_data_analysis.ipynb
```

### Running Categorical Analysis

Using the modular code:

```bash
python run_analysis.py --data data/data_shrunk.npy --analysis categorical --save_df
```

### Training an LSTM Model

Using the modular code:

```bash
python train_model.py --model lstm --data data/data_shrunk.npy --seq_length 20 --pred_horizon 3 --hidden_size 64 --epochs 100 --batch_size 128
```

## License

This project is licensed under the MIT License.
