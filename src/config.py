
from pathlib import Path

# Project root: Vendor Invoice/
# If config.py is inside a subfolder such as src/, use parents[1].
# If config.py is directly in the project root, use parents[0].
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATABASE_PATH = DATA_DIR / "inventory.db"

FREIGHT_MODEL_PATH = MODEL_DIR / "freight_model.pkl"
RISK_CLASSIFIER_PATH = MODEL_DIR / "invoice_risk_classifier.pkl"

# Freight prediction
FREIGHT_FEATURES = [
    "Quantity",
    "Dollars",
]
FREIGHT_TARGET_COLUMN = "Freight"

# Invoice risk classification — X1 features
RISK_FEATURES = [
    "invoice_quantity",
    "Freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
]

RISK_TARGET_COLUMN = "flag_invoice"

# Features used only for the X vs X1 comparison.
# X contains direct target-rule features and may cause target leakage.
RISK_FEATURES_X = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
    "total_item_dollars",
]

# Business rules used to generate the risk target
RISK_DOLLAR_DIFF_THRESHOLD = 5
RISK_RECEIVING_DELAY_THRESHOLD_DAYS = 10