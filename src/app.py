import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Loan Default Credit Scoring API",
    description="Inference API to score borrowers, calculate credit scores, and predict probabilities of default (PD).",
    version="1.0"
)

# Global variables for model and metadata
model = None
meta = None
feature_columns = []
w_factor = 4.03

# Pydantic schema for input validation
class BorrowerFeatures(BaseModel):
    loan_amnt: float = Field(..., example=10000.0, description="The listed amount of the loan applied for by the borrower.")
    term: int = Field(..., example=36, description="The number of payments on the loan (months).")
    int_rate: float = Field(..., example=12.5, description="Interest rate on the loan (%).")
    grade: str = Field(..., example="B", description="Lending Club assigned loan grade (A-G).")
    sub_grade: str = Field(..., example="B3", description="Lending Club assigned loan sub-grade (A1-G5).")
    emp_length_years: float = Field(..., example=5.0, description="Employment length in years (NaN / missing should be passed as -1).")
    emp_length_missing: bool = Field(..., example=False, description="Flag indicating if employment length was missing.")
    home_ownership: str = Field(..., example="MORTGAGE", description="The home ownership status provided by the borrower (MORTGAGE, RENT, OWN, etc.).")
    annual_inc: float = Field(..., example=75000.0, description="The self-reported annual income provided by the borrower.")
    purpose: str = Field(..., example="debt_consolidation", description="A category provided by the borrower for the loan request.")
    dti: float = Field(..., example=18.5, description="A ratio calculated using the borrower’s total monthly debt payments divided by monthly income.")
    dti_missing: bool = Field(..., example=False, description="Flag indicating if DTI was missing.")
    delinq_2yrs: int = Field(..., example=0, description="The number of 30+ days past-due delinquencies in the borrower's credit file for the past 2 years.")
    open_acc: int = Field(..., example=10, description="The number of open credit lines in the borrower's credit file.")
    pub_rec: int = Field(..., example=0, description="Number of derogatory public records.")
    revol_bal: float = Field(..., example=15000.0, description="Total credit revolving balance.")
    revol_util: float = Field(..., example=45.2, description="Revolving line utilization rate (%).")
    revol_util_missing: bool = Field(..., example=False, description="Flag indicating if revolving utility was missing.")
    total_acc: int = Field(..., example=22, description="The total number of credit lines currently in the borrower's credit file.")
    addr_state: str = Field(..., example="CA", description="The state provided by the borrower in the loan application.")
    application_type: str = Field(..., example="Individual", description="Indicates whether the loan is an individual application or a joint application.")

class ScoreResponse(BaseModel):
    probability_of_default: float = Field(..., description="The model predicted probability that the borrower defaults.")
    credit_score: int = Field(..., description="The scaled credit score (300 to 850).")
    risk_tier: str = Field(..., description="The risk category assigned based on the credit score (Low, Medium, High).")

@app.on_event("startup")
def startup_load_artifacts():
    global model, meta, feature_columns, w_factor
    model_path = os.path.join("output", "model.joblib")
    meta_path = os.path.join("data", "processed", "preprocessor_meta.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("Warning: Model artifacts not found. Please run src/train.py and src/feature_engineering.py first.")
        return
        
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feature_columns = meta["features"]
    w_factor = float(meta.get("scale_pos_weight", 4.03))
    print("Model and preprocessor metadata successfully loaded on startup!")

@app.post("/score", response_model=ScoreResponse)
def score_borrower(features: BorrowerFeatures):
    if model is None or meta is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure training scripts have run successfully.")
        
    try:
        # Convert Pydantic model to dict
        data = features.dict()
        
        # 1. Create a zero-initialized Series for the features (matching training columns)
        model_input = pd.Series(0, index=feature_columns)
        
        # 2. Map Numerical & Boolean values directly
        numeric_cols = [
            'loan_amnt', 'term', 'int_rate', 'emp_length_years', 
            'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 
            'revol_bal', 'revol_util', 'total_acc'
        ]
        for col in numeric_cols:
            if col in data:
                model_input[col] = data[col]
                
        # Map boolean indicators
        bool_indicators = ['emp_length_missing', 'dti_missing', 'revol_util_missing']
        for col in bool_indicators:
            if col in data:
                model_input[col] = 1 if data[col] else 0

        # 3. Map Ordinal Encoded Variables
        grade = data['grade'].upper()
        if grade in meta['grade_map']:
            model_input['grade_encoded'] = meta['grade_map'][grade]
        else:
            # Default to worst grade if unrecognized
            model_input['grade_encoded'] = meta['grade_map']['G']
            
        sub_grade = data['sub_grade'].upper()
        if sub_grade in meta['sub_grade_map']:
            model_input['sub_grade_encoded'] = meta['sub_grade_map'][sub_grade]
        else:
            model_input['sub_grade_encoded'] = meta['sub_grade_map']['G5']
            
        # 4. Map One-Hot Encoded Variables
        # Check matching columns (e.g. home_ownership_RENT, purpose_debt_consolidation, etc.)
        one_hot_mappings = {
            'home_ownership': data['home_ownership'],
            'purpose': data['purpose'],
            'addr_state': data['addr_state'].upper(),
            'application_type': data['application_type']
        }
        
        for category_name, val in one_hot_mappings.items():
            col_to_set = f"{category_name}_{val}"
            if col_to_set in model_input.index:
                model_input[col_to_set] = 1
                
        # 5. Model Inference (PD Prediction)
        input_df = pd.DataFrame([model_input])
        pd_val = model.predict_proba(input_df)[0, 1]
        
        # 6. Scale Credit Score (300 to 850)
        pd_val_clipped = np.clip(pd_val, 0.0001, 0.9999)
        odds = (1 - pd_val_clipped) / pd_val_clipped
        true_odds = w_factor * odds
        
        # Base Odds of 4:1 (average default rate in portfolio is 20%) maps to 600 points
        pdo = 20
        base_odds = 4
        base_score = 600
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        
        score = offset + factor * np.log(true_odds)
        score_val = int(np.clip(score, 300, 850))
        
        # 7. Determine Risk Tier
        if score_val >= 660:
            risk_tier = 'Low'
        elif score_val >= 580:
            risk_tier = 'Medium'
        else:
            risk_tier = 'High'
            
        return {
            "probability_of_default": float(pd_val),
            "credit_score": score_val,
            "risk_tier": risk_tier
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/health")
def health_check():
    if model is not None and meta is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False, "notes": "Run training scripts to load model artifacts."}
