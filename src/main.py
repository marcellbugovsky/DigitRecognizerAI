import torch.cuda
import yaml

# Load the config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DEVICE = config["device"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")