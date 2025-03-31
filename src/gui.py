import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import Canvas, Label, simpledialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import neural_network
from mnist_loader import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
input_size = 784  # 28x28 images
hidden_sizes = [512, 256, 128, 64]
output_size = 10  # Digits 0-9

# Initialize model
model = neural_network.NeuralNetwork(input_size, hidden_sizes, output_size).to(device)

# Load model
model_path = os.path.abspath("../models/saved_model-finetuned.pth")
if os.path.exists(model_path):
    model.load_model(model_path)
    print("Model loaded successfully!")
else:
    print("Warning: Model file not found. The model will start untrained.")

import matplotlib.pyplot as plt


import cv2


def preprocess_canvas():
    """Preprocesses the drawn digit to match MNIST normalization used during training."""
    # Keep resize and invert
    img = image.resize((28, 28)).convert("L")  # Resize and convert to grayscale
    img = ImageOps.invert(img)  # Invert colors to match MNIST

    # Define the same transform sequence as used in mnist_loader.py
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image [0, 255] to Tensor [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])

    # Apply the transforms directly to the PIL image
    # The transform handles scaling and normalization
    img_tensor = transform(img).to(device)  # Output is already (C, H, W), e.g., (1, 28, 28)

    # The model expects a flattened tensor (batch_size, features)
    # Reshape to (1, 784)
    img_tensor = img_tensor.view(1, 784)

    # img_to_show = img_tensor.view(28, 28).cpu().detach().numpy()
    # plt.imshow(img_to_show, cmap='gray')
    # plt.title("Preprocessed Input to Model")
    # plt.show()

    # --- Remove the old manual normalization ---
    # # Convert to NumPy array
    # img_array = np.array(img, dtype=np.float32)
    # # Normalize manually before applying MNIST normalization
    # img_array = img_array / 255.0  # Scale to 0-1 range
    # # Apply MNIST normalization
    # img_array = (img_array - 0.5) / 0.5  # Correct normalization: Now in range -1 to 1
    # # Convert to PyTorch tensor
    # img_tensor = torch.tensor(img_array, dtype=torch.float32, device=device).view(1, 784)

    # --- Optional: Update or remove debug prints ---
    # The old debug prints were based on the incorrect normalization
    # You could print tensor stats if needed:
    # print(f"Tensor min after transform: {img_tensor.min().item():.4f}")
    # print(f"Tensor max after transform: {img_tensor.max().item():.4f}")
    # print(f"Tensor mean after transform: {img_tensor.mean().item():.4f}")
    # print(f"Tensor std after transform: {img_tensor.std().item():.4f}")

    # Return the original inverted PIL image (for saving) and the correctly processed tensor
    return img, img_tensor


def predict_digit():
    """Predicts the drawn digit and adjusts confidence scaling."""
    img, img_tensor = preprocess_canvas()

    with torch.no_grad():
        output = model(img_tensor)  # Raw logits
        print(f"Raw model output (logits): {output.cpu().numpy()}")  # Print logits

        temperature = 2.0  # Reduce confidence by scaling logits
        probabilities = F.softmax(output / temperature, dim=1).cpu().numpy()

        predicted_digit = probabilities.argmax()
        confidence = probabilities.max() * 100

    print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}%")

    result_label.config(text=f"Predicted: {predicted_digit} ({confidence:.2f}%)")
    confirm_button.config(state=tk.NORMAL)
    correct_button.config(state=tk.NORMAL)

    return predicted_digit


def save_digit(correct_label):
    """Saves the labeled digit for future training."""
    img, _ = preprocess_canvas()
    digit_folder = f"../data/custom_digits/{correct_label}"  # Save in respective digit folder
    os.makedirs(digit_folder, exist_ok=True)
    img.save(os.path.join(digit_folder, f"{len(os.listdir(digit_folder))}.png"))
    messagebox.showinfo("Saved", f"Digit {correct_label} saved successfully!")
    clear_canvas()


def confirm_prediction():
    """Confirms the model's prediction and saves it."""
    predicted_digit = result_label.cget("text").split(": ")[1][0]
    save_digit(predicted_digit)


def correct_prediction():
    """Allows the user to manually correct the prediction and save it."""
    correct_label = simpledialog.askstring("Correction", "Enter the correct digit (0-9):")
    if correct_label and correct_label.isdigit() and 0 <= int(correct_label) <= 9:
        save_digit(correct_label)
    else:
        messagebox.showerror("Error", "Please enter a valid digit (0-9).")


def clear_canvas():
    """Clears the drawing canvas."""
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")

# Setup GUI
root = tk.Tk()
root.title("Digit Recognizer AI")
canvas = Canvas(root, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=3)

# Image to store drawing
image = Image.new("RGB", (280, 280), "white")
draw = ImageDraw.Draw(image)

def paint(event):
    brush_size = 7
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")

canvas.bind("<B1-Motion>", paint)

# Buttons
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.grid(row=1, column=0)
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=1)
result_label = Label(root, text="Draw a digit and press Predict")
result_label.grid(row=2, column=0, columnspan=3)
confirm_button = tk.Button(root, text="Confirm", state=tk.DISABLED, command=confirm_prediction)
confirm_button.grid(row=3, column=0)
correct_button = tk.Button(root, text="Correct", state=tk.DISABLED, command=correct_prediction)
correct_button.grid(row=3, column=1)

root.mainloop()