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

    def train_model(self, device, data_root, batch_size, epochs, learning_rate, lr_decay_step, lr_decay_gamma, print_every):
        # Load dataset
        train_data, train_labels, test_data, test_labels = mnist_loader.load_data(data_root, device)

        # Convert to DataLoader
        train_dataset = data.TensorDataset(train_data, train_labels)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

        # Training loop
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            scheduler.step()

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

    def evaluate(self, device, data_root, batch_size):
        # Load dataset
        train_data, train_labels, test_data, test_labels = mnist_loader.load_data(data_root, device)

        # Convert to DataLoader
        test_dataset = data.TensorDataset(test_data, test_labels)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Test Accuracy: {correct / total:.4f}")