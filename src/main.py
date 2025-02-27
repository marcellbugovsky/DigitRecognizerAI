import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import mnist_loader
import yaml
import os

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load model parameters from config
input_size = config["input_size"]
hidden_sizes = config["hidden_sizes"]
output_size = config["output_size"]
batch_size = config["batch_size"]
epochs = config["epochs"]
initial_learning_rate = config["initial_learning_rate"]
lr_decay_step = config["lr_decay_step"]
lr_decay_gamma = config["lr_decay_gamma"]
model_path = config["model_path"]

data_root = config["data_root"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_data, train_labels, test_data, test_labels = mnist_loader.load_data(data_root)