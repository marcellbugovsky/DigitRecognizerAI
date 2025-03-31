import yaml
import os
from multiprocessing import freeze_support
from src.utils.device import DEVICE, get_device
from src.data.loader import get_mnist_loaders, get_combined_loader
from src.models.mlp import MLP
from src.training.trainer import get_optimizer, get_scheduler, train_epoch, evaluate_model
import torch # Ensure torch is imported

CONFIG_PATH = "../config/config.yaml"

def run_main_training(config):
    print("--- Starting Initial Training ---")
    print(f"Using device: {DEVICE}")

    # --- Data ---
    if config['training'].get('use_custom_data', False):
        print("Loading Combined MNIST + Custom Data...")
        train_loader = get_combined_loader(config)
         # For evaluation, we usually still use standard MNIST test/val sets
        _, valid_loader, test_loader = get_mnist_loaders(config, validation_split=0.1) # Get val/test MNIST
    else:
        print("Loading MNIST Data...")
        train_loader, valid_loader, test_loader = get_mnist_loaders(config, validation_split=0.1) # Split train for validation

    if train_loader is None:
        print("Failed to load training data. Exiting.")
        return

    # --- Model ---
    model = MLP(config).to(DEVICE)
    print(f"Model initialized:\n{model}")

    # --- Training Setup ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config['training'])
    scheduler = get_scheduler(optimizer, config['training'])

    epochs = config['training']['epochs']
    log_interval = config['training']['log_interval']
    model_save_dir = config['model_save_dir']
    model_basename = config['model_basename']
    initial_model_path = os.path.join(model_save_dir, f"{model_basename}.pth")

    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, epochs)

        if valid_loader:
             val_loss, val_acc = evaluate_model(model, valid_loader, criterion)
             print(f"Epoch [{epoch+1}/{epochs}] Valid | Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

             # Save model checkpoint if validation accuracy improves
             if val_acc > best_val_accuracy:
                 print(f"Validation accuracy improved ({best_val_accuracy:.4f} -> {val_acc:.4f}). Saving model...")
                 best_val_accuracy = val_acc
                 model.save_checkpoint(initial_model_path)

        elif (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
             # Save model periodically if no validation set
             print(f"Saving model checkpoint at epoch {epoch+1}...")
             model.save_checkpoint(initial_model_path)


        if scheduler:
            scheduler.step() # Update learning rate

    print("--- Initial Training Finished ---")

    # --- Final Evaluation ---
    if test_loader:
        print("Running final evaluation on test set...")
        # Load best model if validation was used
        if valid_loader and os.path.exists(initial_model_path):
             print("Loading best validation checkpoint for final test...")
             model.load_checkpoint(initial_model_path)

        evaluate_model(model, test_loader, criterion)

if __name__ == '__main__':
    freeze_support()
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {CONFIG_PATH}: {e}")
        exit(1)

    run_main_training(config_data)
    print("Main training script finished.")