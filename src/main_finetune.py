import yaml
import os
from multiprocessing import freeze_support
from src.utils.device import DEVICE, get_device
from src.data.loader import get_custom_data_loader
from src.models.mlp import MLP
from src.training.trainer import get_optimizer, get_scheduler, train_epoch, evaluate_model # Re-use evaluation
import torch

CONFIG_PATH = "../config/config.yaml"

def run_fine_tuning(config):
    print("--- Starting Fine-Tuning on Custom Data ---")
    print(f"Using device: {DEVICE}")

    # --- Data ---
    print("Loading Custom Data for Fine-tuning...")
    custom_train_loader = get_custom_data_loader(config, shuffle=True)

    if custom_train_loader is None or len(custom_train_loader.dataset) == 0 :
        print("No valid custom data found for fine-tuning. Exiting.")
        return

    # We might want a small validation set from custom data, or evaluate on MNIST test set later
    # For simplicity here, we'll just train on all custom data and maybe eval on MNIST test

    # --- Model ---
    model = MLP(config).to(DEVICE)
    model_save_dir = config['model_save_dir']
    model_basename = config['model_basename']
    # Load the *initial* trained model
    initial_model_path = os.path.join(model_save_dir, f"{model_basename}.pth")
    print(f"Loading initial model from: {initial_model_path}")
    loaded = model.load_checkpoint(initial_model_path)
    if not loaded:
         print("Failed to load initial model. Cannot fine-tune. Exiting.")
         return

    # --- Fine-tuning Setup ---
    criterion = torch.nn.CrossEntropyLoss()
    # Use fine-tuning specific config section
    optimizer = get_optimizer(model, config['fine_tuning'])
    scheduler = get_scheduler(optimizer, config['fine_tuning']) # Often None for fine-tuning

    epochs = config['fine_tuning']['epochs']
    log_interval = config['fine_tuning']['log_interval']
    output_suffix = config['fine_tuning']['output_suffix']
    finetuned_model_path = os.path.join(model_save_dir, f"{model_basename}{output_suffix}.pth")


    # --- Fine-tuning Loop ---
    for epoch in range(epochs):
        # Train on custom data
        train_loss, train_acc = train_epoch(model, custom_train_loader, criterion, optimizer, epoch, epochs)

        # Optional: Evaluate on custom data itself (or a split) periodically
        if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
             print(f"Saving fine-tuned model checkpoint at epoch {epoch+1}...")
             model.save_checkpoint(finetuned_model_path)

        if scheduler:
            scheduler.step()

    print("--- Fine-Tuning Finished ---")
    print(f"Final fine-tuned model saved to: {finetuned_model_path}")

    # --- Optional: Final Evaluation on MNIST Test Set ---
    run_final_eval = config.get('fine_tuning', {}).get('evaluate_on_mnist_test', True)
    if run_final_eval:
        print("Running final evaluation of fine-tuned model on MNIST test set...")
        # Need to load MNIST test data
        from src.data.loader import get_mnist_loaders
        try:
            _, _, mnist_test_loader = get_mnist_loaders(config, validation_split=0)
            if mnist_test_loader:
                 evaluate_model(model, mnist_test_loader, criterion)
            else:
                 print("Could not load MNIST test loader for evaluation.")
        except Exception as e:
            print(f"Error loading/evaluating on MNIST test set: {e}")


if __name__ == '__main__':
    freeze_support()
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {CONFIG_PATH}: {e}")
        exit(1)

    run_fine_tuning(config_data)
    print("Fine-tuning script finished.")