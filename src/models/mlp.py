# src/models/mlp.py
import torch
import torch.nn as nn
import os
from src.utils.device import DEVICE # Import device

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model."""
    def __init__(self, config):
        super(MLP, self).__init__()
        model_config = config['model']
        input_size = model_config['input_size']
        hidden_sizes = model_config['hidden_sizes']
        output_size = model_config['output_size']
        dropout_rate = model_config['dropout']
        use_batchnorm = model_config['use_batchnorm']

        layers = []
        current_size = input_size

        # Input Layer and First Hidden Layer
        layers.append(nn.Linear(current_size, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        current_size = hidden_sizes[0]

        # Intermediate Hidden Layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(current_size, hidden_sizes[i+1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_sizes[i+1]

        # Output Layer
        layers.append(nn.Linear(current_size, output_size))
        # Note: No Softmax here, as CrossEntropyLoss expects raw logits

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is flattened
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.network(x.float()) # Ensure float input

    def save_checkpoint(self, file_path):
        """Saves the model state dictionary."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)
        print(f"Model checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        """Loads the model state dictionary."""
        if not os.path.exists(file_path):
             print(f"Warning: Checkpoint file not found at {file_path}. Model not loaded.")
             return False
        try:
            # Load state dict, ensuring it's mapped to the correct device
            self.load_state_dict(torch.load(file_path, map_location=DEVICE))
            self.eval() # Set to evaluation mode after loading
            print(f"Model checkpoint loaded successfully from {file_path}")
            return True
        except Exception as e:
             print(f"Error loading checkpoint from {file_path}: {e}")
             return False