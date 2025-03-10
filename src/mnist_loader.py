import torch
import torchvision
import torchvision.transforms as transforms
import yaml

def load_data(data_root, device):
    """
    Loads the MNIST dataset, normalizes it, and converts it to PyTorch tensors.
    Returns train_data, train_labels, test_data, test_labels.
    """

    # Define transformation (normalize MNIST images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std for MNIST
    ])

    # Load MNIST dataset using config-defined path
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    # Convert datasets to PyTorch tensors
    train_data = train_dataset.data.view(-1, 28 * 28).float().to(device)
    train_labels = train_dataset.targets.to(device)
    test_data = test_dataset.data.view(-1, 28 * 28).float().to(device)
    test_labels = test_dataset.targets.to(device)

    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    return train_data, train_labels, test_data, test_labels
