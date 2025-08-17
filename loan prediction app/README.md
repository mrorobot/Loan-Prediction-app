## Loan Approval Prediction — Streamlit App

Interactive demo that predicts whether a loan application is likely to be approved using a pre-trained machine learning model. The app is built with Streamlit and expects a serialized model at `loan_model.pkl`.

### Features

- **Interactive UI**: Enter applicant details and get an instant prediction.
- **Probability (optional)**: If the model supports `predict_proba`, the approval probability is shown.
- **Feature importances (optional)**: If the model exposes `feature_importances_`, a top-features bar chart is displayed.
- **Processed inputs**: View the exact numeric features sent to the model.

## Project Structure

- `loan_app.py`: Streamlit app UI and inference logic.
- `preprocess.py`: Input parsing and mappings to model-ready numeric features.
- `loan_model.pkl`: Pre-trained model loaded with `joblib`.
- `requirements.txt`: Python package requirements.

## Prerequisites

- Python 3.9+ recommended
- pip (or conda/mamba)

## Setup

### 1) Clone or copy this project

Place all files in a folder (e.g., `loan prediction app`).

### 2) Create a virtual environment

- Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

- macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run loan_app.py
```

This will open a local URL (typically `http://localhost:8501`).

## Model Requirements (`loan_model.pkl`)

- Must be loadable via `joblib.load(...)`.
- Must implement `.predict(X)`; optional `.predict_proba(X)` to show approval probability.
- Optional: `.feature_importances_` (e.g., tree-based models) to render the bar chart.

If you need to save a scikit-learn model:

```python
import joblib
joblib.dump(trained_model, "loan_model.pkl")
```

## Input Preprocessing and Feature Mapping

All UI inputs are converted to numeric features in `preprocess.py`. Ensure these mappings match how you trained your model.

Current feature order expected by the model (`FEATURE_ORDER` in `preprocess.py`):

1. `Male` (gender)
2. `Married`
3. `Non Graduate` (education status)
4. `Self_Employed`
5. `ApplicantIncome`
6. `LoanAmount`
7. `Loan_Amount_Term`
8. `Credit_History`
9. `Property_Area`

Default mappings implemented:

- **Gender**: Male → 1, Female → 0
- **Married**: Yes/True/1 → 1, otherwise 0
- **Education**: "Graduate" → 1, otherwise 0
- **Self Employed**: Yes/True/1 → 1, otherwise 0
- **Property Area**: Urban → 2, Semiurban → 1, Rural → 0
- **ApplicantIncome**: Numeric (float)
- **LoanAmount**: Numeric (float), expected in thousands (as per UI label)
- **Loan_Amount_Term**: Numeric (float), in days
- **Credit_History**: 1.0 for good history, 0.0 otherwise

Notes:

- If you used encoders/scalers during training, adapt `preprocess.py` to load and apply the same objects.
- The exact column order must match training. If you change `FEATURE_ORDER`, retrain or adjust the model accordingly.

## Troubleshooting

- **Model fails to load**: Confirm `loan_model.pkl` exists in the project root and is compatible with your `joblib`/scikit-learn versions.
- **Wrong or unstable predictions**: Verify that `FEATURE_ORDER` and all mappings in `preprocess.py` exactly match training-time preprocessing.
- **Probability not shown**: Your model may not implement `predict_proba`.
- **Feature importance chart not shown**: Your model may not expose `feature_importances_`.
- **Streamlit not found**: Re-activate your virtualenv and reinstall requirements.

## Deploying

- You can deploy to Streamlit Community Cloud or any environment that can run `streamlit run loan_app.py`. Ensure `loan_model.pkl` and the same Python libraries are available.

## License

Use and modify freely for educational and demonstration purposes. Replace this section with your license of choice if needed.
