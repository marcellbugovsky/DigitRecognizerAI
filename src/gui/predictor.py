# src/gui/predictor.py
import torch
import torch.nn.functional as F
import os
from src.models.mlp import MLP # Import your model class
from src.data.transforms import get_gui_preprocess_transform
from src.utils.device import DEVICE

class DigitPredictor:
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.transform = get_gui_preprocess_transform()
        self.prediction_temperature = config['gui'].get('prediction_temperature', 1.0)

    def _get_model_path(self):
        """Constructs the path to the model file based on config."""
        model_dir = self.config['model_save_dir']
        basename = self.config['model_basename']
        suffix = self.config['gui'].get('model_suffix_to_load', '-finetuned') # e.g., "_finetuned" or ''
        filename = f"{basename}{suffix}.pth"
        return os.path.join(model_dir, filename)

    def _load_model(self):
        """Loads the specified model checkpoint."""
        model_path = self._get_model_path()
        print(f"GUI: Attempting to load model from: {model_path}")

        # Initialize model architecture based on main config
        model = MLP(self.config).to(DEVICE)

        # Load checkpoint
        loaded = model.load_checkpoint(model_path)
        if not loaded:
             print("GUI Warning: Failed to load model weights. Using untrained model.")
        else:
             print("GUI: Model loaded successfully.")
             model.eval() # Ensure model is in eval mode

        return model

    def preprocess(self, pil_image):
        """
        Preprocesses a PIL image (resized, inverted) for prediction.
        Returns a flattened tensor ready for the model.
        """
        # Apply transforms (ToTensor, Normalize)
        img_tensor = self.transform(pil_image).to(DEVICE) # Shape (1, 28, 28)
        # Flatten for MLP input
        img_tensor_flat = img_tensor.view(1, -1) # Shape (1, 784)
        return img_tensor_flat

    def predict(self, image_tensor_flat):
        """
        Makes a prediction on a preprocessed, flattened image tensor.
        Returns predicted digit, confidence, and raw logits.
        """
        if self.model is None:
            print("GUI Error: Model not loaded.")
            return -1, 0.0, None # Error indication

        with torch.no_grad():
            logits = self.model(image_tensor_flat)
            # Apply temperature scaling before softmax for confidence adjustment
            probabilities = F.softmax(logits / self.prediction_temperature, dim=1)

            confidence, predicted_digit = torch.max(probabilities, 1)

            # Move results to CPU and convert to standard types
            predicted_digit = predicted_digit.cpu().item()
            confidence = confidence.cpu().item() * 100 # Percentage
            logits_np = logits.cpu().numpy()

        return predicted_digit, confidence, logits_np