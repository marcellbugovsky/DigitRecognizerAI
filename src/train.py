import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import mnist_loader

def train(model, device, data_root, batch_size, epochs, learning_rate):
    # Load dataset
    train_data, train_labels, test_data, test_labels = mnist_loader.load_data(data_root, device)

    # Convert to DataLoader
    train_dataset = data.TensorDataset(train_data, train_labels)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

def evaluate(model, device, data_root, batch_size):
    # Load dataset
    train_data, train_labels, test_data, test_labels = mnist_loader.load_data(data_root, device)

    # Convert to DataLoader
    test_dataset = data.TensorDataset(test_data, test_labels)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")