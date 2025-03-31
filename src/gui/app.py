# src/gui/app.py
import tkinter as tk
from tkinter import Canvas, Label, Button, simpledialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import os
from .predictor import DigitPredictor # Import the predictor logic
from src.data.transforms import get_gui_preprocess_transform # Keep for saving logic reference if needed

class DigitRecognizerApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.predictor = DigitPredictor(config) # Instantiate predictor

        # --- Configuration from YAML ---
        gui_config = config['gui']
        self.canvas_width = gui_config['canvas_width']
        self.canvas_height = gui_config['canvas_height']
        self.brush_size = gui_config['brush_size']
        self.custom_data_root = config['custom_data_root'] # For saving

        self.root.title(gui_config['window_title'])

        # Image for drawing (internal representation)
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Setup GUI elements
        self._setup_widgets()

    def _setup_widgets(self):
        # Canvas
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.paint_release) # Optional: trigger predict on release?

        # Buttons
        self.predict_button = Button(self.root, text="Predict", command=self.run_prediction, width=15)
        self.predict_button.grid(row=1, column=0, padx=5, pady=5)

        self.clear_button = Button(self.root, text="Clear", command=self.clear_canvas, width=15)
        self.clear_button.grid(row=1, column=1, padx=5, pady=5)

        # Result Label
        self.result_label = Label(self.root, text="Draw a digit (0-9) and press Predict", font=("Arial", 12))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=5)

        # Confirmation/Correction Buttons (initially disabled)
        self.confirm_button = Button(self.root, text="Confirm ✔", state=tk.DISABLED, command=self.confirm_prediction, width=15, fg="green")
        self.confirm_button.grid(row=3, column=0, padx=5, pady=10)

        self.correct_button = Button(self.root, text="Correct ✘", state=tk.DISABLED, command=self.correct_prediction, width=15, fg="red")
        self.correct_button.grid(row=3, column=1, padx=5, pady=10)

        # Store last prediction for saving
        self.last_predicted_digit = None
        self.last_processed_image_for_saving = None


    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        # Draw on Tkinter Canvas (visual feedback)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        # Draw on PIL Image (internal data)
        self.draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")
        # Disable confirm/correct buttons while drawing
        self.confirm_button.config(state=tk.DISABLED)
        self.correct_button.config(state=tk.DISABLED)
        self.last_predicted_digit = None


    def paint_release(self, event):
        # Optional: could trigger prediction automatically here
        pass

    def clear_canvas(self):
        self.canvas.delete("all")
        # Reset PIL image
        self.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill="white")
        self.result_label.config(text="Draw a digit (0-9) and press Predict")
        self.confirm_button.config(state=tk.DISABLED)
        self.correct_button.config(state=tk.DISABLED)
        self.last_predicted_digit = None
        self.last_processed_image_for_saving = None

    def run_prediction(self):
        # 1. Prepare image for preprocessing (resize, invert)
        # Use the current PIL image data
        img_resized = self.image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized.convert("L")) # Convert to grayscale before inverting

        # Store this version for potential saving
        self.last_processed_image_for_saving = img_inverted

        # 2. Preprocess for the model using the Predictor's method
        img_tensor_flat = self.predictor.preprocess(img_inverted)

        # 3. Predict using the Predictor's method
        predicted_digit, confidence, logits = self.predictor.predict(img_tensor_flat)
        self.last_predicted_digit = predicted_digit # Store prediction

        print(f"Raw model output (logits): {logits}")
        print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}%")

        # 4. Update GUI
        if predicted_digit != -1: # Check for prediction error
            self.result_label.config(text=f"Predicted: {predicted_digit} ({confidence:.2f}%)")
            self.confirm_button.config(state=tk.NORMAL)
            self.correct_button.config(state=tk.NORMAL)
        else:
            self.result_label.config(text="Prediction Error. Check Logs.")
            self.confirm_button.config(state=tk.DISABLED)
            self.correct_button.config(state=tk.DISABLED)


    def save_digit(self, correct_label_str):
        """Saves the last processed digit image with the correct label."""
        if self.last_processed_image_for_saving is None:
            messagebox.showerror("Error", "No image processed to save.")
            return

        try:
            correct_label = int(correct_label_str)
            if not (0 <= correct_label <= 9):
                 raise ValueError("Digit out of range")

            digit_folder = os.path.join(self.custom_data_root, str(correct_label))
            os.makedirs(digit_folder, exist_ok=True)

            # Find next available filename (e.g., 0.png, 1.png, ...)
            file_index = 0
            while os.path.exists(os.path.join(digit_folder, f"{file_index}.png")):
                file_index += 1
            save_path = os.path.join(digit_folder, f"{file_index}.png")

            self.last_processed_image_for_saving.save(save_path)
            messagebox.showinfo("Saved", f"Digit '{correct_label}' saved successfully to:\n{save_path}")
            self.clear_canvas() # Clear after saving

        except ValueError:
             messagebox.showerror("Error", "Invalid digit label. Please enter 0-9.")
        except Exception as e:
             messagebox.showerror("Error", f"Failed to save digit: {e}")


    def confirm_prediction(self):
        """Confirms the model's prediction and saves the digit."""
        if self.last_predicted_digit is not None:
            self.save_digit(str(self.last_predicted_digit))
        else:
            messagebox.showwarning("Warning", "No prediction available to confirm.")

    def correct_prediction(self):
        """Allows the user to manually correct the prediction and save it."""
        correct_label = simpledialog.askstring("Correction", "Enter the correct digit (0-9):", parent=self.root)
        if correct_label: # Check if user entered something (didn't press cancel)
             # save_digit handles validation
             self.save_digit(correct_label)