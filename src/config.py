"""Project paths.

Everything is resolved relative to the project root so the code works on any
machine without absolute paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PIPELINE_PATH = MODELS_DIR / "credit_risk_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
