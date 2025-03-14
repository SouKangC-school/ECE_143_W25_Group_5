"""
Base model class for all neural network models.
"""
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base class for all models with common functionality.
    """
    def __init__(self, input_size, output_size=1):
        """
        Initialize the base model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features (default: 1)
        """
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    def get_model_size(self):
        """
        Calculate the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'model_class': self.__class__.__name__
        }, path)
    
    @classmethod
    def load_model(cls, path, device=torch.device('cpu')):
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['input_size'], checkpoint['output_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

class LinearModel(BaseModel):
    """
    Simple linear model for baseline comparisons.
    """
    def __init__(self, input_size, seq_length, output_size=1):
        """
        Initialize the linear model.
        
        Args:
            input_size: Number of input features
            seq_length: Length of input sequences
            output_size: Number of output features (default: 1)
        """
        super(LinearModel, self).__init__(input_size, output_size)
        self.seq_length = seq_length
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size * seq_length, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # x shape: [batch_size, seq_length, input_size]
        out = self.flatten(x)
        out = self.fc(out)
        return out
