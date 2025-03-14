"""
LSTM-based neural network models.
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class SimpleLSTMModel(BaseModel):
    """
    Simple LSTM model for time series prediction.
    """
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.1, output_size=1):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(SimpleLSTMModel, self).__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.lstm = nn.LSTM(
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
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

class BiLSTMModel(BaseModel):
    """
    Bidirectional LSTM model for time series prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(BiLSTMModel, self).__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional LSTM has 2*hidden_size as its output size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # x shape: [batch_size, seq_length, input_size]
        _, (hidden, _) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class AttentionLSTM(BaseModel):
    """
    LSTM model with attention mechanism for time series prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the LSTM with Attention model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(AttentionLSTM, self).__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
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
        output, (hidden, cell) = self.lstm(x)

        # Calculate attention weights
        attn_weights = self.attention(output)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention weights to the output
        context = torch.sum(output * attn_weights, dim=1)

        # Final prediction
        out = self.dropout(context)
        out = self.fc(out)
        return out
