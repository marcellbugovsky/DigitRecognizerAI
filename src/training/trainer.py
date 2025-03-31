# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.device import DEVICE
import time

def get_optimizer(model, config):
    """Creates an optimizer based on config."""
    lr = config['learning_rate']
    optimizer_name = config.get('optimizer', 'Adam') # Default to Adam

    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        # Add momentum/weight decay options to config if needed
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config):
    """Creates a learning rate scheduler based on config."""
    scheduler_config = config.get('scheduler')
    if not scheduler_config or not scheduler_config.get('use', False):
        return None

    scheduler_type = scheduler_config.get('type', 'StepLR')
    if scheduler_type.lower() == 'steplr':
        step_size = scheduler_config.get('step_size', 5)
        gamma = scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Add other scheduler types (e.g., ReduceLROnPlateau) here if needed
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_epoch(model, data_loader, criterion, optimizer, epoch_num, total_epochs):
    """Runs a single training epoch."""
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs) # Forward pass (model handles flattening)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) # Weighted loss
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Optional: Log batch progress
        # if batch_idx % 50 == 0:
        #     print(f"  Batch {batch_idx}/{len(data_loader)}")

    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"Epoch [{epoch_num+1}/{total_epochs}] Train | "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, Time: {epoch_duration:.2f}s")

    return epoch_loss, epoch_accuracy

def evaluate_model(model, data_loader, criterion):
    """Evaluates the model on a given dataset."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    eval_duration = time.time() - start_time
    eval_loss = running_loss / total_samples if total_samples > 0 else 0
    eval_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"Evaluation | Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.4f}, Time: {eval_duration:.2f}s")

    return eval_loss, eval_accuracy