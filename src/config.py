from pathlib import Path

# Project root: Vendor Invoice/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATABASE_PATH = DATA_DIR / "inventory.db"
FREIGHT_MODEL_PATH = MODEL_DIR / "freight_model.pkl"
RISK_CLASSIFIER_PATH = MODEL_DIR / "risk_classifier.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FREIGHT_FEATURES = ["Quantity", "Dollars"]
RISK_FEATURES = [
    "Quantity",
    "Dollars",
    "freight_per_unit",
    "freight_diff",
]
TARGET_COLUMN = "Freight"
