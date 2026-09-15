"""Streamlit interface for the credit risk scoring model.

The app does not repeat any preprocessing. It collects a few applicant fields,
fills the remaining features with the reference values saved during training,
and passes the row to the same pipeline and helper functions used in the
notebook.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    load_artifacts,
    build_applicant_frame,
    predict_credit_risk,
)

st.set_page_config(page_title="Credit Risk Scoring", layout="centered")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title("Credit Risk Scoring")
st.write(
    "Estimates the probability that an applicant will have repayment "
    "difficulties, using the model trained in the project notebook."
)

try:
    pipeline, metadata = get_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

template = metadata["template_row"]


def options_for(column, fallback):
    """Use the training value as the first option so it stays valid."""
    value = template.get(column)
    if value is None:
        return fallback
    return [value] + [v for v in fallback if v != value]


st.subheader("Applicant details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
    gender = st.selectbox(
        "Gender",
        options_for("CODE_GENDER", ["M", "F"])
    )
    children = st.number_input(
        "Number of children", min_value=0, max_value=20, value=0
    )
    employed = st.checkbox("Currently employed", value=True)
    years_employed = st.number_input(
        "Years employed", min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        disabled=not employed
    )

with col2:
    income = st.number_input(
        "Annual income", min_value=0.0, value=150000.0, step=5000.0
    )
    credit = st.number_input(
        "Loan amount requested", min_value=0.0, value=500000.0, step=10000.0
    )
    annuity = st.number_input(
        "Annual loan payment", min_value=0.0, value=25000.0, step=1000.0
    )
    goods_price = st.number_input(
        "Price of goods", min_value=0.0, value=450000.0, step=10000.0
    )

education = st.selectbox(
    "Education",
    options_for("NAME_EDUCATION_TYPE", [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ])
)

income_type = st.selectbox(
    "Income type",
    options_for("NAME_INCOME_TYPE", [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
    ])
)

st.subheader("Credit history")

col3, col4 = st.columns(2)

with col3:
    bureau_credits = st.number_input(
        "Previous credits on record", min_value=0, max_value=100, value=3
    )
    bureau_overdue = st.number_input(
        "Previous credits with overdue days", min_value=0, max_value=100, value=0
    )

with col4:
    late_installments = st.number_input(
        "Late installment payments", min_value=0, max_value=500, value=0
    )
    refused_share = st.slider(
        "Share of previous applications refused", 0.0, 1.0, 0.0, 0.05
    )

st.caption(
    "Fields that are not asked for here are filled with the median or most "
    "common value from the training data."
)

if st.button("Score applicant", type="primary"):
    if income <= 0:
        st.error("Income must be greater than zero.")
        st.stop()

    if credit <= 0:
        st.error("Loan amount must be greater than zero.")
        st.stop()

    values = {
        "DAYS_BIRTH": -age * 365,
        "CODE_GENDER": gender,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_EDUCATION_TYPE": education,
        "NAME_INCOME_TYPE": income_type,
        "BUREAU_CREDIT_COUNT": bureau_credits,
        "BUREAU_OVERDUE_COUNT": bureau_overdue,
        "INSTALLMENT_LATE_COUNT": late_installments,
        "PREV_REFUSAL_RATE": refused_share,
    }

    if employed:
        values["DAYS_EMPLOYED"] = -years_employed * 365
        values["DAYS_EMPLOYED_MISSING"] = 0
    else:
        values["DAYS_EMPLOYED"] = float("nan")
        values["DAYS_EMPLOYED_MISSING"] = 1

    # Only send fields the trained model actually knows about
    values = {k: v for k, v in values.items() if k in template}

    try:
        applicant = build_applicant_frame(values, metadata)
        result = predict_credit_risk(applicant, pipeline, metadata).iloc[0]
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not score this applicant: {error}")
        st.stop()

    probability = float(result["risk_probability"])
    category = result["risk_category"]

    st.subheader("Result")

    left, right = st.columns(2)
    left.metric("Predicted risk probability", f"{probability:.1%}")
    right.metric("Risk category", category)

    st.progress(min(probability, 1.0))

    threshold = metadata["threshold"]

    if result["flagged_as_risky"] == 1:
        st.warning(
            f"The predicted probability is at or above the decision threshold "
            f"of {threshold:.2f}, so this applicant is flagged for review."
        )
    else:
        st.success(
            f"The predicted probability is below the decision threshold of "
            f"{threshold:.2f}, so this applicant is not flagged."
        )

    st.caption(
        "The threshold was selected on a validation set and the risk bands are "
        "defined for this project. This is a portfolio model trained on the "
        "Home Credit dataset, not a lending decision tool."
    )

with st.sidebar:
    st.header("Model")
    st.write(f"**{metadata.get('model_name', 'Final model')}**")
    st.write(f"Decision threshold: {metadata['threshold']:.2f}")

    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        st.write("Test set performance:")
        for metric in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"]:
            if metric in test_metrics:
                st.write(f"- {metric}: {test_metrics[metric]:.3f}")
