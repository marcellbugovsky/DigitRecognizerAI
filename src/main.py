import yaml

# Load the config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

DEVICE = config["device"]

print(f"Using device: {DEVICE}")