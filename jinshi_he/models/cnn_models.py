"""
CNN-based neural network models.
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class CNNLSTMModel(BaseModel):
    """
    Hybrid CNN-LSTM model for time series prediction.
    """
    def __init__(self, input_size, seq_length, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            seq_length: Length of input sequences
            hidden_size: Size of the LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(CNNLSTMModel, self).__init__(input_size, output_size)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate CNN output size (seq_length reduced by MaxPool)
        self.cnn_output_size = 64 * (seq_length // 4)

        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # x shape: [batch_size, seq_length, input_size]
        batch_size = x.size(0)

        # Transpose for CNN: [batch_size, input_size, seq_length]
        x = x.transpose(1, 2)

        # Apply CNN
        x = self.cnn(x)  # [batch_size, 64, seq_length//4]

        # Transpose back for LSTM: [batch_size, seq_length//4, 64]
        x = x.transpose(1, 2)

        # Apply LSTM
        _, (hidden, _) = self.lstm(x)

        # Use the last hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out
