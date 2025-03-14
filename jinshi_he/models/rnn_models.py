"""
RNN-based neural network models.
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class SimpleRNNModel(BaseModel):
    """
    Simple RNN model for time series prediction.
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1, output_size=1):
        """
        Initialize the RNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(SimpleRNNModel, self).__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        _, hidden = self.rnn(x)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out
