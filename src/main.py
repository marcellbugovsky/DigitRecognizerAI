import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import mnist_loader
import yaml
import neural_network
import os

# Load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load config
data_root = config["data_root"]
model_path = config["model_path"]

input_size = config["input_size"]
hidden_sizes = config["hidden_sizes"]
output_size = config["output_size"]
batch_size = config["batch_size"]

epochs = config["epochs"]
initial_learning_rate = config["initial_learning_rate"]
lr_decay_step = config["lr_decay_step"]
lr_decay_gamma = config["lr_decay_gamma"]

print_every = config["print_every"]

# Set device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = neural_network.NeuralNetwork(input_size, hidden_sizes, output_size).to(device)

model.train_model(device, data_root, batch_size, epochs, initial_learning_rate, lr_decay_step, lr_decay_gamma, print_every)
model.evaluate(device, data_root, batch_size)