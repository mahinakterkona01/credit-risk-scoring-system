# Credit Risk Scoring System

A machine learning project that predicts whether a loan applicant is likely to
have repayment difficulties, using application data combined with the
applicant's past credit and payment behaviour.

## Live Demo

[Streamlit App Link]

## Problem

Lenders have to decide who is likely to repay. The available data is imbalanced:
only about 8% of applicants in this dataset had repayment difficulties, so a
model that predicts "no difficulty" for everyone would already be about 92%
accurate and completely useless. The project is built around that problem —
imbalance-aware metrics, an explicit decision threshold, and explainability for
individual predictions.

## Key Features

- Imbalanced classification handled with balanced class weights and
  imbalance-aware metrics (ROC-AUC, PR-AUC) instead of accuracy
- Applicant-level aggregation of three historical data sources on top of the
  main application data
- An ablation experiment that measures what each historical source actually
  adds, rather than assuming more data is better
- Model comparison across Logistic Regression, Decision Tree and Random
  Forest, with two of them tuned by cross-validation
- Validation-based model selection, with the test set touched only once
- A decision threshold chosen on the validation set and frozen before testing
- Global and per-applicant explanations with SHAP
- The full preprocessing + model pipeline saved as one artifact and served
  through a Streamlit app

## Dataset

The Home Credit Default Risk dataset from Kaggle. Four files are used:

| File | Role |
| --- | --- |
| `application_train.csv` | Main applicant data and the `TARGET` column |
| `bureau.csv` | Credit history reported by other institutions |
| `previous_application.csv` | Earlier applications with the same lender |
| `installments_payments.csv` | Historical installment repayment records |

The dataset has 307,511 applicants and 122 columns. The target is heavily
imbalanced — about 8.1% of applicants had repayment difficulties.

The CSV files are not included in this repository. Download them from Kaggle and
put them in `data/raw/`.

## Data Preparation and Feature Engineering

The application data is cleaned first: columns with very high missingness
(mostly property-related fields) are dropped, the `DAYS_EMPLOYED` placeholder
value is treated as missing rather than a real number, and the main financial
columns are checked for suspicious or extreme values.

The three historical files have many rows per applicant, so they're
aggregated to one row per `SK_ID_CURR` before being merged into the main
table:

- **Bureau**: credit count, total credit, total debt, average days since
  credit, overdue count, debt-to-credit ratio
- **Previous applications**: application count, approved/refused counts,
  total and average credit, approval and refusal rates
- **Installments**: installment count, total payment, total due, average
  payment, average payment delay, late payment count, missing payment count

After merging, the dataset has 96 features. Missing values from the merge
(applicants with no history in a given source) are filled with zero where
that's the correct interpretation (e.g. no previous applications) and left
for the pipeline's imputer otherwise.

## Modeling

Logistic Regression, Decision Tree and Random Forest are trained on the same
preprocessed data and compared on a validation set. Preprocessing (median
imputation and scaling for numeric features, most-frequent imputation and
one-hot encoding for categorical features) is fit inside a `ColumnTransformer`
on the training set only, then applied to validation and test.

Data is split 60% train / 20% validation / 20% test, stratified on the
target, so all three sets keep the same ~8% positive rate.

Logistic Regression and Random Forest are then tuned with cross-validation
(grid search for Logistic Regression, randomized search for Random Forest),
optimizing ROC-AUC. All five model versions — three baselines and two tuned —
are compared on the validation set, and the one with the best validation
ROC-AUC is selected as the final model.

An ablation experiment also checks whether the historical features are
actually earning their place, by training separate Random Forests on
application-only data and on application data plus each historical source in
turn.

## Evaluation

Accuracy is reported but not used to pick a model or a threshold — at an 8%
positive rate it barely reacts to how well the risky class is being found.
Model selection and threshold selection are driven by ROC-AUC and PR-AUC
(average precision) instead, since PR-AUC only looks at the positive class
and a random model would score close to the ~0.08 base rate.

Validation results across all five model versions:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6940 | 0.1639 | 0.6800 | 0.2641 | 0.7530 | 0.2237 |
| Decision Tree | 0.6640 | 0.1428 | 0.6322 | 0.2330 | 0.6936 | 0.1756 |
| Random Forest | 0.7501 | 0.1810 | 0.5944 | 0.2775 | 0.7499 | 0.2214 |
| Logistic Regression (tuned) | 0.6938 | 0.1640 | 0.6814 | 0.2643 | 0.7530 | 0.2244 |
| **Random Forest (tuned)** | 0.8653 | 0.2536 | 0.3440 | 0.2920 | 0.7558 | 0.2261 |

Random Forest (tuned) had the best validation ROC-AUC and PR-AUC and was
selected as the final model.

Final test results (single evaluation, after the threshold was frozen on
validation):

| Metric | Value |
| --- | --- |
| Accuracy | 0.8312 |
| Precision | 0.2279 |
| Recall | 0.4570 |
| F1-Score | 0.3042 |
| ROC-AUC | 0.7611 |
| PR-AUC | 0.2398 |

The ablation results (validation set) show a small but consistent gain from
each added historical source:

| Feature Set | Features | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: |
| A: Application only | 75 | 0.7378 | 0.2093 |
| B: + Bureau | 81 | 0.7403 | 0.2117 |
| C: + Previous applications | 88 | 0.7432 | 0.2148 |
| D: + Installments (full) | 96 | 0.7489 | 0.2198 |

## Threshold Selection

A classifier's default 0.5 cutoff has no special meaning — it just happens to
be the midpoint of the probability scale. With an imbalanced target and a
model that outputs calibrated-ish probabilities, the threshold that gives the
best trade-off between false positives and false negatives is almost never
0.5.

The threshold was chosen by scanning a range of cutoffs (0.20 to 0.70) on the
validation set and picking the one that maximizes F1, since there's no real
cost data available to weigh false positives against false negatives
directly. The selected threshold is **0.45**, giving validation precision of
0.221, recall of 0.460, F1 of 0.299, with 8,053 false positives and 2,680
false negatives on the validation set. This threshold was frozen before the
test set was touched.

On top of the threshold, the project also defines three risk bands for
reporting: **Low Risk** below 0.281, **Medium Risk** from 0.281 up to 0.45,
and **High Risk** at 0.45 and above. The lower cutoff is the median predicted
probability on the validation set; the upper cutoff is the same 0.45
decision threshold. These bands describe where an applicant's predicted risk
sits relative to others in the data — they don't predict what will actually
happen to that applicant.

## Explainability

SHAP is used in two ways. The global view ranks features by their average
effect on predicted risk across a sample of test applicants — the credit
bureau score fields (`EXT_SOURCE_1/2/3`), `DAYS_EMPLOYED`, and the engineered
bureau feature for average days since credit came out on top. The individual
view shows which features pushed one specific applicant's prediction up or
down, using a waterfall plot starting from the average prediction over the
sample.

SHAP explains how the model uses its inputs, not why those inputs are
correlated with risk, and it doesn't prove that a feature actually improves
performance — that's what the ablation experiment is for.

## Streamlit App

The workflow is: user input → preprocessing → trained pipeline → risk
probability → risk category.

The notebook saves two artifacts with `joblib`: the fitted pipeline
(preprocessing + final Random Forest bundled together) and a metadata file
containing the decision threshold, the risk band cutoffs, the expected
feature order, and a reference row of median/mode values for every feature.
The Streamlit app loads these directly — it does not retrain or refit
anything. It collects a smaller set of applicant fields from the user, fills
in the remaining fields the model expects using the saved reference values,
and scores the result with the loaded pipeline to get a probability, a
flagged/not-flagged decision at the 0.45 threshold, and a risk category.

## Project Structure

```
Credit-Risk-Scoring-System/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                  # Kaggle CSV files (not tracked)
│   └── processed/
├── models/                   # saved pipeline and metadata (not tracked)
├── notebook/
│   └── credit_risk_scoring_analysis.ipynb
├── reports/
├── src/
│   ├── config.py             # project paths
│   └── predict.py            # scoring helpers used by the app
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## How to Run

1. Put the four Kaggle CSV files in `data/raw/`.
2. Open `notebook/credit_risk_scoring_analysis.ipynb` and run it from top to
   bottom. It writes `models/credit_risk_pipeline.joblib` and
   `models/model_metadata.joblib`.
3. Start the app from the project root:

```bash
streamlit run app/streamlit_app.py
```

The Random Forest hyperparameter search and the ablation experiment are the
slow cells — on a normal laptop the full notebook takes a while to run.

## Results

The final model is a tuned Random Forest, selected on validation ROC-AUC and
PR-AUC, with a decision threshold of 0.45 frozen before testing. On the held-
out test set it reaches a ROC-AUC of 0.7611 and a PR-AUC of 0.2398, with
precision/recall/F1 of 0.2279 / 0.4570 / 0.3042 at the selected threshold.
Precision on the positive class stays low in absolute terms, which is
expected at an 8% base rate — the model is better read as a ranking tool for
prioritizing applications than as a hard accept/reject decision.

## Limitations

- Three files from the same Kaggle dataset (`bureau_balance.csv`,
  `POS_CASH_balance.csv`, `credit_card_balance.csv`) are not used, so there is
  no monthly balance or credit card behaviour in the model.
- There is no real cost data for a wrong decision, so the threshold is chosen
  on model performance rather than on business cost.
- Results come from a single random split, so small differences between
  feature sets in the ablation table shouldn't be over-interpreted.
- The model is validated only on this dataset, with no external or
  out-of-period validation.
- The historical features aggregate an applicant's whole history and don't
  distinguish recent behaviour from old behaviour.
- SHAP values were computed on a sample of 300 test rows for runtime reasons;
  the overall ranking should be stable, but individual values would shift on
  a larger sample.
- Some cleaning decisions, like the missing-value cutoff used to drop the
  property columns, were based on manual review rather than a systematic
  feature-selection method.

## Future Improvements

- Add the remaining Home Credit tables (monthly balance, credit card
  behaviour).
- Add time-aware versions of the historical features, e.g. last-12-months
  behaviour separate from full history.
- Replace the single split with repeated or cross-validated evaluation.
- Test a gradient boosting model on the same validation setup.
- Revisit the threshold with real cost assumptions if they become available.

## Disclaimer

This is a project built on a public Kaggle dataset (`Home Credit Default Risk`). It is developed purely for educational and analytical purposes. It is **not** a production system and should **not** be used to make real-world lending or credit decisions.
