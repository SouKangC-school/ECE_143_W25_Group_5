"""
Transformer-based neural network models.
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class TransformerModel(BaseModel):
    """
    Transformer model for time series prediction.
    """
    def __init__(self, input_size, seq_length, nhead=4, dim_feedforward=128, num_layers=2, dropout=0.2, output_size=1):
        """
        Initialize the Transformer model.
        
        Args:
            input_size: Number of input features
            seq_length: Length of input sequences
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
            output_size: Number of output features (default: 1)
        """
        super(TransformerModel, self).__init__(input_size, output_size)
        self.seq_length = seq_length
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Positional encoding
        self.pos_encoder = nn.Embedding(seq_length, input_size)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # x shape: [batch_size, seq_length, input_size]
        batch_size, seq_length, _ = x.size()

        # Add positional encoding
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoding = self.pos_encoder(positions)
        x = x + pos_encoding

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Final prediction
        x = self.dropout(x)
        x = self.fc(x)
        return x
