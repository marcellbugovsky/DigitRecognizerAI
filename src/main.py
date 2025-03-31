import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import mnist_loader
import yaml
import neural_network
import os
# --- Optional, but good practice for multiprocessing ---
from multiprocessing import freeze_support

# --- Code that can safely run on import (definitions, imports) goes above the guard ---

# --- Main execution block ---
if __name__ == '__main__':
    # --- Add freeze_support() for potential freezing later (optional but recommended) ---
    freeze_support()

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

    # Load existing model weights if available
    if os.path.exists(model_path):
        # Pass device to load_model if your method requires it
        # Assuming your load_model method handles map_location implicitly or doesn't need device
        model.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model file not found at {model_path}, starting fresh.")

    # Train the model
    model.train_model(device, data_root, batch_size, epochs, initial_learning_rate, lr_decay_step, lr_decay_gamma, print_every)

    # Evaluate the model
    model.evaluate(device, data_root, batch_size)

    # Save the model (consider adding this explicitly if train_model doesn't save)
    print(f"Saving final model to {model_path}")
    model.save_model(model_path)
    print("Done.")