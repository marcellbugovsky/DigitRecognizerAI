import tkinter as tk
import yaml
from src.gui.app import DigitRecognizerApp
from multiprocessing import freeze_support # Needed for potential freezing

CONFIG_PATH = "../config/config.yaml"

if __name__ == '__main__':
    freeze_support() # For multiprocessing, good practice

    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {CONFIG_PATH}: {e}")
        exit(1)

    root = tk.Tk()
    app = DigitRecognizerApp(root, config)
    root.mainloop()