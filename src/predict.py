"""Scoring helpers for the credit risk model.

The notebook saves a single pipeline that contains the preprocessing and the
model together, plus a metadata file with the decision threshold and the risk
bands.

The model files are loaded locally when available. If they are not present
(for example, on Streamlit Cloud), they are downloaded from Hugging Face.
"""

import joblib
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download

from .config import PIPELINE_PATH, METADATA_PATH


# Hugging Face repository containing the saved model files
HF_REPO_ID = "Kona11/credit-risk-scoring-model"


def load_artifacts(pipeline_path=PIPELINE_PATH, metadata_path=METADATA_PATH):
    """Load the saved pipeline and metadata.

    Local files are used when available. Otherwise, the files are downloaded
    from the project's public Hugging Face repository.
    """

    # Use local files when they exist
    if pipeline_path.exists() and metadata_path.exists():
        return joblib.load(pipeline_path), joblib.load(metadata_path)

    # Download from Hugging Face when running without local model files
    pipeline_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="credit_risk_pipeline.joblib",
    )

    metadata_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="model_metadata.joblib",
    )

    return joblib.load(pipeline_file), joblib.load(metadata_file)


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
