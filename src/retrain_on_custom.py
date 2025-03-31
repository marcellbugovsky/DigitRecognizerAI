import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import yaml # Import YAML library
import neural_network # Imports the NeuralNetwork class definition
import mnist_loader   # Imports load_custom_data

# --- Load Configuration ---
CONFIG_PATH = "config.yaml"
try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    print(f"Loaded configuration from {CONFIG_PATH}")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    exit()
except Exception as e:
    print(f"Error loading or parsing configuration file: {e}")
    exit()

# --- Extract Parameters from Config ---
# Paths
MODEL_PATH = os.path.abspath(config["model_path"])
# Get custom data folder from config, default if not present
CUSTOM_DATA_FOLDER = os.path.abspath(config.get("custom_data_folder", "../data/custom_digits"))
# Define output path - either derive it or add a specific key in config
output_suffix = config.get("finetune_output_model_suffix", "-finetuned")
base, ext = os.path.splitext(MODEL_PATH)
OUTPUT_MODEL_PATH = f"{base}{output_suffix}{ext}"

# Model parameters
INPUT_SIZE = config["input_size"]
HIDDEN_SIZES = config["hidden_sizes"]
OUTPUT_SIZE = config["output_size"]

# Fine-tuning hyperparameters
EPOCHS = config.get("finetune_epochs", 15) # Default to 15 if not in config
# Use specific fine-tune LR, default if not present
LEARNING_RATE = config.get("finetune_learning_rate", 1e-4)
# Use specific fine-tune batch size, otherwise use general batch_size from config
BATCH_SIZE = config.get("finetune_batch_size", config.get("batch_size", 32))

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Fine-tuning model: {MODEL_PATH}")
print(f"Using custom data from: {CUSTOM_DATA_FOLDER}")
print(f"Saving fine-tuned model to: {OUTPUT_MODEL_PATH}")
print(f"Hyperparameters: Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR={LEARNING_RATE}")


# --- Initialize and Load Model ---
model = neural_network.NeuralNetwork(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE).to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_model(MODEL_PATH, device)
        # print(f"Successfully loaded pre-trained model from: {MODEL_PATH}") # load_model prints
    except Exception as e:
        print(f"Error loading model weights from {MODEL_PATH}: {e}")
        print("Proceeding with initial model architecture weights.")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Starting with initial model weights.")
    # Consider exiting if fine-tuning requires a pre-trained model
    # exit()

# --- Load Custom Data ---
print(f"Loading custom digits...")
if not os.path.exists(CUSTOM_DATA_FOLDER):
    print(f"Error: Custom data folder not found at {CUSTOM_DATA_FOLDER}")
    exit()

custom_dataset = mnist_loader.load_custom_data(CUSTOM_DATA_FOLDER)

if len(custom_dataset) == 0:
    print("Error: No custom digits found in the specified folder.")
    exit()
else:
     print(f"Loaded {len(custom_dataset)} custom digit images.")

train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Fine-tuning Setup ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"Starting fine-tuning...")

# --- Fine-tuning Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if not isinstance(labels, torch.Tensor):
             print(f"Warning: Non-tensor label detected: {type(labels)} - {labels}")
             continue

        images, labels = images.to(device), labels.to(device)
        images_flattened = images.view(images.size(0), -1)

        optimizer.zero_grad()
        outputs = model(images_flattened)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # Prevent division by zero if dataset is empty (though checked earlier)
    if total_samples > 0:
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - No data processed.")


# --- Save the Fine-tuned Model ---
try:
    model.save_model(OUTPUT_MODEL_PATH)
    # print(f"Fine-tuned model saved successfully to: {OUTPUT_MODEL_PATH}") # save_model prints
except Exception as e:
    print(f"Error saving fine-tuned model to {OUTPUT_MODEL_PATH}: {e}")

print("Fine-tuning finished.")