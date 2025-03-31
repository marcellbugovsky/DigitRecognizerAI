import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDigitDataset(Dataset):
    """Loads user-labeled custom digits for training."""
    def __init__(self, data_folder, transform=None):
        self.data = []
        self.labels = [] # Store labels as integers
        self.transform = transform

        for digit in range(10):  # Loop through digit folders (0-9)
            digit_path = os.path.join(data_folder, str(digit))
            if not os.path.exists(digit_path):
                continue
            for img_name in os.listdir(digit_path):
                img_path = os.path.join(digit_path, img_name)
                try: # Add basic error handling for image loading
                    img = Image.open(img_path).convert("L")  # Convert to grayscale
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}. Error: {e}")
                    continue

                if self.transform:
                    img = self.transform(img)
                self.data.append(img)
                # --- CHANGE HERE: Store label as int ---
                self.labels.append(int(digit)) # Store the integer label

        # --- REMOVED: No need to stack images here, handled by DataLoader ---
        # if len(self.data) > 0:
        #     self.data = torch.stack(self.data) # Let DataLoader handle stacking

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # --- CHANGE HERE: Retrieve image and int label ---
        image = self.data[idx] # Image is already transformed (if transform was provided)
        label = self.labels[idx] # Retrieve the integer label
        return image, label # Return (Tensor_Image, int_Label)

def load_custom_data(data_folder):
    """Loads custom hand-drawn digits as a dataset."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return CustomDigitDataset(data_folder, transform=transform)  # ✅ Ensures correct loading

def load_data(data_root): # No need for device argument here
    """Loads the MNIST dataset as PyTorch Dataset objects."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    # Return the dataset objects directly
    return train_dataset, test_dataset
