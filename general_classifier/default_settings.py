"""
Settings file for training general classifier.
"""

# NOTE: Use absolute path
DATA_YAML: str = "/home/minh/data/godot_4000_cropped/data.yaml"
BATCH_SIZE: int = 256
EPOCHS: int = 1000

LOGS_PATH: str = "lightning_logs"

DEBUG: bool = False
