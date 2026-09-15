"""Scoring helpers for the credit risk model.

The notebook saves a single pipeline that contains the preprocessing and the
model together, plus a metadata file with the decision threshold and the risk
bands. This module loads both and exposes one function for scoring applicants,
so the notebook and the Streamlit app use exactly the same logic.
"""

import joblib
import numpy as np
import pandas as pd

from .config import PIPELINE_PATH, METADATA_PATH


def load_artifacts(pipeline_path=PIPELINE_PATH, metadata_path=METADATA_PATH):
    """Load the saved pipeline and its metadata."""
    if not pipeline_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model files not found. Run the notebook to the end to create "
            f"{pipeline_path.name} and {metadata_path.name} in models/."
        )

    return joblib.load(pipeline_path), joblib.load(metadata_path)


def risk_category(probability, bands):
    """Convert a predicted probability into a project-defined risk band."""
    if probability >= bands["high_cutoff"]:
        return "High Risk"
    if probability >= bands["medium_cutoff"]:
        return "Medium Risk"
    return "Low Risk"


def build_applicant_frame(values, metadata):
    """Build a complete feature row from a partial set of values.

    Fields that are not supplied are filled with the reference values saved
    during training, so the pipeline always receives the columns it expects.
    """
    row = dict(metadata["template_row"])

    unknown = set(values) - set(row)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")

    row.update(values)

    return pd.DataFrame([row])[metadata["feature_order"]]


def predict_credit_risk(applicant_data, pipeline, metadata):
    """Score one or more applicants and return probability, decision and band."""
    if isinstance(applicant_data, dict):
        applicant_data = pd.DataFrame([applicant_data])

    missing = set(metadata["feature_order"]) - set(applicant_data.columns)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns, for example: "
            f"{sorted(missing)[:5]}"
        )

    data = applicant_data[metadata["feature_order"]].copy()

    # Same infinite-value handling that was applied before training
    data = data.replace([np.inf, -np.inf], np.nan)

    probabilities = pipeline.predict_proba(data)[:, 1]
    threshold = metadata["threshold"]
    bands = metadata["risk_bands"]

    return pd.DataFrame({
        "risk_probability": probabilities.round(4),
        "flagged_as_risky": (probabilities >= threshold).astype(int),
        "risk_category": [risk_category(p, bands) for p in probabilities]
    })
