"""
config.py - Central configuration for MachineGuard AI.

Everything that is reused by app.py, prediction.py and database.py lives here:
paths, database settings, model definitions, feature metadata, UI constants.
All paths are relative to the folder that contains this file (the project root).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (relative to the project root, i.e. the machine_failure folder)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_NAME = "MachineGuard AI"
APP_TAGLINE = "Industrial machine failure prediction"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Database (credentials come from .env, never from source code)
# ---------------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "machineguard_db")
MONGO_TIMEOUT_MS = 3000

COL_MACHINES = "machines"
COL_PREDICTIONS = "predictions"
COL_DETAILS = "prediction_details"
COL_TASKS = "maintenance_tasks"
COL_ACTIVITY = "activity_log"

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_ROWS = 500_000         # larger files must be split before upload
MAX_STORED_DETAIL_ROWS = 5_000    # row-level details saved per prediction run
TABLE_PREVIEW_ROWS = 300          # rows rendered in on-screen tables
ISSUE_PREVIEW_ROWS = 200

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
PALETTE = {
    "graphite": "#202725",
    "orange": "#E58A32",
    "sand": "#D8D0BD",
    "sage": "#78877C",
    "ivory": "#F2F0E9",
    "red": "#B94B3D",
}

PAGES = [
    ("dashboard", "Dashboard", "dashboard"),
    ("compressor", "Air Compressor Workspace", "air"),
    ("ai4i", "General Industrial Machine Workspace", "precision_manufacturing"),
    ("analytics", "Risk & Analytics", "monitoring"),
    ("planner", "Maintenance Planner", "event_note"),
    ("history", "Prediction History", "history"),
]

MACHINE_CATEGORIES = [
    "Air compressor",
    "Industrial machine / machine tool",
    "Pump",
    "Electric motor",
    "Conveyor",
    "Other",
]
TASK_PRIORITIES = ["Low", "Medium", "High", "Critical"]
TASK_STATUSES = ["Planned", "In progress", "On hold", "Completed", "Cancelled"]
OPEN_TASK_STATUSES = ["Planned", "In progress", "On hold"]
REVIEW_STATUSES = [
    "Pending review",
    "Under inspection",
    "Reviewed - issue confirmed by inspection",
    "Reviewed - no issue found",
]
INPUT_METHODS = ["Manual input", "File upload"]

# ---------------------------------------------------------------------------
# AI4I 2020 - feature definitions copied from ai4i_feature_engineering.py
# ---------------------------------------------------------------------------
AI4I_TARGET = "Machine failure"
AI4I_TEST_SIZE = 0.20
AI4I_RANDOM_STATE = 42
AI4I_TYPES = ["L", "M", "H"]  # categories seen in the dataset (H, L, M)

AI4I_RAW_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_ENGINEERED = {
    "Temperature_Difference": "Process temperature [K] - Air temperature [K]",
    "Power_Proxy": "Rotational speed [rpm] x Torque [Nm] (relative mechanical power proxy, not electrical power)",
    "Tool_Wear_Category": "Tool wear grouped as Low (<= 100 min), Medium (> 100 and <= 200 min), High (> 200 min)",
    "Torque_Per_RPM": "Torque [Nm] / Rotational speed [rpm]",
}
AI4I_WEAR_BINS = [100, 200]
AI4I_WEAR_LABELS = ["Low", "Medium", "High"]

AI4I_INPUT_META = {
    "Type": {"label": "Product quality variant (Type)", "unit": "L / M / H", "step": None, "default": None,
             "help": "Quality variant of the product being machined."},
    "Air temperature [K]": {"label": "Air temperature", "unit": "K", "step": 0.1, "default": 300.0,
                            "help": "Ambient air temperature."},
    "Process temperature [K]": {"label": "Process temperature", "unit": "K", "step": 0.1, "default": 310.0,
                                "help": "Temperature of the machining process."},
    "Rotational speed [rpm]": {"label": "Rotational speed", "unit": "rpm", "step": 1.0, "default": 1500.0,
                               "help": "Spindle rotational speed. Must be greater than 0."},
    "Torque [Nm]": {"label": "Torque", "unit": "Nm", "step": 0.1, "default": 40.0,
                    "help": "Torque applied by the machine."},
    "Tool wear [min]": {"label": "Tool wear", "unit": "min", "step": 1.0, "default": 0.0,
                        "help": "Accumulated tool usage time."},
}
# hard physical rules checked during validation: column -> (rule, message)
AI4I_RULES = {
    "Air temperature [K]": ("gt0", "Must be greater than 0 K"),
    "Process temperature [K]": ("gt0", "Must be greater than 0 K"),
    "Rotational speed [rpm]": ("gt0", "Must be greater than 0 (torque per rpm is undefined at 0)"),
    "Torque [Nm]": ("ge0", "Must not be negative"),
    "Tool wear [min]": ("ge0", "Must not be negative"),
}

# ---------------------------------------------------------------------------
# MetroPT-3 compressor - formulas copied from compressor_feature_engineering.py
# ---------------------------------------------------------------------------
COMPRESSOR_TARGET = "Failure_Status"
COMPRESSOR_ENGINEERED = {
    "Pressure_Difference_TP3_TP2": "TP3 - TP2",
    "Pressure_Difference_Reservoir_TP2": "Reservoirs - TP2",
    "Motor_Current_Change": "Motor_current - previous Motor_current (previous chronological reading)",
    "Oil_Temperature_Change": "Oil_temperature - previous Oil_temperature (previous chronological reading)",
}
# raw sensors every engineered feature is calculated from
COMPRESSOR_SOURCES = {
    "Pressure_Difference_TP3_TP2": ["TP3", "TP2"],
    "Pressure_Difference_Reservoir_TP2": ["Reservoirs", "TP2"],
    "Motor_Current_Change": ["Motor_current"],
    "Oil_Temperature_Change": ["Oil_temperature"],
}
COMPRESSOR_PREVIOUS_COLUMNS = ["Motor_current", "Oil_temperature"]
COMPRESSOR_TIMESTAMP = "timestamp"

# Descriptions of the MetroPT-3 signals. The exact list of inputs the model uses
# is read from the model file at runtime; this is only display metadata.
COMPRESSOR_SENSORS = {
    "TP2": {"label": "TP2 - compressor pressure", "unit": "bar"},
    "TP3": {"label": "TP3 - pneumatic panel pressure", "unit": "bar"},
    "H1": {"label": "H1 - pressure drop at cyclonic separator filter", "unit": "bar"},
    "DV_pressure": {"label": "DV pressure - pressure drop at towers discharge", "unit": "bar"},
    "Reservoirs": {"label": "Reservoirs pressure", "unit": "bar"},
    "Oil_temperature": {"label": "Oil temperature", "unit": "°C"},
    "Motor_current": {"label": "Motor current", "unit": "A"},
    "COMP": {"label": "COMP - air intake valve signal", "unit": "0/1"},
    "DV_eletric": {"label": "DV electric - discharge valve signal", "unit": "0/1"},
    "Towers": {"label": "Towers - active dryer tower signal", "unit": "0/1"},
    "MPG": {"label": "MPG - load start signal", "unit": "0/1"},
    "LPS": {"label": "LPS - low pressure signal", "unit": "0/1"},
    "Pressure_switch": {"label": "Pressure switch signal", "unit": "0/1"},
    "Oil_level": {"label": "Oil level signal", "unit": "0/1"},
    "Caudal_impulses": {"label": "Caudal impulses - air flow signal", "unit": "0/1"},
}

# ---------------------------------------------------------------------------
# Model definitions (model files go in the models/ folder)
# ---------------------------------------------------------------------------
MODELS = {
    "compressor": {
        "key": "compressor",
        "machine_name": "Industrial Air Compressor",
        "short_name": "Air Compressor",
        "dataset": "MetroPT-3 Air Compressor Dataset",
        "model_file": "compressor_best_model.joblib",
        "model_path": MODELS_DIR / "compressor_best_model.joblib",
        "reference_csv": BASE_DIR / "compressor_engineered.csv",
        "icon": "air",
        "category": "Air compressor",
        "reported_accuracy": 95.72,
        "reported_failure_f1": 76.82,
        "tagline": "Based on the MetroPT-3 metro-train air compressor dataset",
        "purpose": (
            "An industrial air compressor turns electrical power into compressed air that drives "
            "pneumatic equipment. In the MetroPT-3 data it is the air production unit of a metro train, "
            "supplying air for braking and door systems, so its pressures, oil temperature and motor "
            "current are monitored continuously."
        ),
        "class_meaning": (
            "Class 1 means the reading resembles records that fall inside the four documented air-leak "
            "failure intervals of the MetroPT-3 dataset. Class 0 means outside those intervals, which does "
            "not prove the machine is healthy."
        ),
        "key_sensors": [
            ("TP2", "bar"), ("TP3", "bar"), ("Reservoirs", "bar"),
            ("Oil temperature", "°C"), ("Motor current", "A"),
        ],
    },
    "ai4i": {
        "key": "ai4i",
        "machine_name": "General Industrial Machine",
        "short_name": "General Industrial Machine",
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "model_file": "ai4i_best_model.joblib",
        "model_path": MODELS_DIR / "ai4i_best_model.joblib",
        "reference_csv": BASE_DIR / "ai4i_engineered.csv",
        "icon": "precision_manufacturing",
        "category": "Industrial machine / machine tool",
        "reported_accuracy": 98.70,
        "reported_failure_f1": 79.37,
        "tagline": "Based on the synthetic AI4I 2020 predictive-maintenance dataset",
        "purpose": (
            "A general industrial machine such as a milling or turning machine is monitored through its "
            "operating conditions. The AI4I 2020 dataset is synthetic data that imitates this setting, "
            "so it demonstrates the method but does not describe any real factory machine."
        ),
        "class_meaning": (
            "Class 1 means the 'Machine failure' label of the AI4I 2020 dataset. The data are synthetic."
        ),
        "key_sensors": [
            ("Air temperature", "K"), ("Process temperature", "K"), ("Rotational speed", "rpm"),
            ("Torque", "Nm"), ("Tool wear", "min"),
        ],
    },
}

DATA_DISCLAIMER = (
    "The MetroPT-3 data come from a metro-train air compressor and the AI4I 2020 data are synthetic. "
    "Neither represents your own factory machines. Predictions are model outputs, not confirmed failures, "
    "and the models do not estimate remaining useful life or time to failure."
)


def input_meta(model_key: str, column: str) -> dict:
    """Display metadata (label, unit, step, default) for a model input column."""
    if model_key == "ai4i":
        return AI4I_INPUT_META.get(column, {"label": column, "unit": "", "step": 0.1, "default": 0.0, "help": ""})
    meta = COMPRESSOR_SENSORS.get(column, {"label": column, "unit": ""})
    digital = meta["unit"] == "0/1"
    return {
        "label": meta["label"],
        "unit": meta["unit"],
        "step": 1.0 if digital else 0.01,
        "default": 0.0,
        "help": "",
    }
