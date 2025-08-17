# preprocess.py
"""
Simple preprocessing helper.
Make sure these mappings & defaults match how you trained the model.
If you used encoders/scalers in training, replace these with loading those objects.
"""

import pandas as pd
import numpy as np

# The column order your model expects (use the exact order used during training)
FEATURE_ORDER = [
    "Male", "Married", "Non Graduate", "Self_Employed",
    "ApplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area"
]

def _map_gender(x):

    """this function maps the gender to a number"""
    # Map Male->1, Female->0 (adjust if you trained differently)
    if isinstance(x, str):
        return 1 if x.strip().lower().startswith("m") else 0
    return int(x)

def _map_married(x):
    return 1 if str(x).strip().lower() in ["yes", "y", "true", "1"] else 0

def _map_dependents(x):
    s = str(x).strip()
    if s == "3+":
        return 3
    try:
        return int(s)
    except:
        return 0

def _map_education(x):
    return 1 if str(x).strip().lower() == "graduate" else 0

def _map_self_employed(x):
    return 1 if str(x).strip().lower() in ["yes", "y", "true", "1"] else 0

def _map_property_area(x):
    # Urban -> 2, Semiurban -> 1, Rural -> 0
    s = str(x).strip().lower()
    if "urban" in s and "semi" not in s:
        return 2
    if "semi" in s:
        return 1
    return 0

def prepare_input_dataframe(raw_input: dict):
    """
    raw_input: dict with keys used in loan_app.py (strings/numbers)
    returns: pandas DataFrame with one row in the FEATURE_ORDER order
    """
    # convert and map
    mapped = {
        "Male": _map_gender(raw_input.get("Gender", "Male")),
        "Married": _map_married(raw_input.get("Married", "No")),
        
        "Non Graduate": _map_education(raw_input.get("Education", "Graduate")),
        "Self_Employed": _map_self_employed(raw_input.get("Self_Employed", "No")),
        "ApplicantIncome": float(raw_input.get("ApplicantIncome", 0.0)),
        
        "LoanAmount": float(raw_input.get("LoanAmount", 0.0)),
        "Loan_Amount_Term": float(raw_input.get("Loan_Amount_Term", 360.0)),
        "Credit_History": float(raw_input.get("Credit_History", 1.0)),
        "Property_Area": _map_property_area(raw_input.get("Property_Area", "Urban"))
    }

    # Build DataFrame in correct order
    df = pd.DataFrame([[ mapped[col] for col in FEATURE_ORDER ]], columns=FEATURE_ORDER)
    return df
