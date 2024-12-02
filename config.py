"""Configuration settings for the application."""

import os
import torch

# File paths and directories
DATA_DIR = "data"
DOCS_DIR = "docs"
STORAGE_DIR = "storage"
DB_PATH = "SCORES.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lighter model
EMBEDDING_DIMENSION = 384  # New embedding dimension for the lighter model

# Check available device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    # Set memory efficient settings for GPU
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
    torch.backends.cudnn.benchmark = True

# Database settings
REQUIRED_COLUMNS = [
    "ACCOUNTDOCID",
    "BLENDED_RISK_SCORE",
    "AI_RISK_SCORE",
    "STAT_SCORE",
    "RULES_RISK_SCORE",
    "CONTROL_DEVIATION",
    "MONITORING_DEVIATION"
]

# Batch processing settings
BATCH_SIZE = 32
MAX_LENGTH = 512

# Create required directories
for directory in [DATA_DIR, DOCS_DIR, STORAGE_DIR]:
    os.makedirs(directory, exist_ok=True)
