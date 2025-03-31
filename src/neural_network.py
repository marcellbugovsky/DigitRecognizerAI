import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from src import mnist_loader

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()

        # Define the layers
        layers = []
        previous_size = input_size

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save_model(self, file_path):
        # Saves the model state to a file
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        # Loads the model state from a file
        self.load_state_dict(torch.load(file_path))
        self.eval()
        print(f"Model loaded from {file_path}")

    def train_model(self, device, data_root, batch_size, epochs, learning_rate, lr_decay_step, lr_decay_gamma, print_every):
        # --- CHANGE HERE ---
        # Load dataset OBJECTS
        train_dataset_obj, _ = mnist_loader.load_data(data_root) # Get the training Dataset object

        # Create DataLoader directly from the MNIST Dataset object
        train_loader = data.DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Added num_workers and pin_memory for potential speedup

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

        print("Starting training...")
        # Training loop
        for epoch in range(epochs):
            self.train() # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            # DataLoader now yields (image_tensor, label_int) pairs
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device) # Move data to device

                optimizer.zero_grad()

                # Flatten the input image (28x28 -> 784) for the Linear layer
                outputs = self(inputs.view(inputs.size(0), -1).float())

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Optional: Print batch progress
                # if (i + 1) % 100 == 0:
                #     print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


            scheduler.step() # Step the scheduler once per epoch

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total

            # Print epoch summary
            if (epoch + 1) % print_every == 0 or (epoch + 1) == epochs:
                print(
                    f"Epoch {epoch + 1}/{epochs} completed | Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        print("Training finished.")
        # Consider saving the model here if needed, or ensure it's saved in main.py after call
        # self.save_model(model_path) # Needs model_path variable

    def evaluate(self, device, data_root, batch_size):
        # --- CHANGE HERE ---
        # Load dataset OBJECTS
        _, test_dataset_obj = mnist_loader.load_data(data_root) # Get the test Dataset object

        # Create DataLoader directly from the MNIST Dataset object
        test_loader = data.DataLoader(test_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("Starting evaluation...")
        self.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculations
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device) # Move data to device

                # Flatten the input image
                outputs = self(inputs.view(inputs.size(0), -1).float())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        # return accuracy # Optional: return value if needed