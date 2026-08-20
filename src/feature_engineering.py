import os
import sys
import joblib
import pandas as pd
from sqlalchemy import create_engine

# Add parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import get_db_params

def perform_feature_engineering():
    print("1. Connecting to PostgreSQL database...")
    params = get_db_params()
    # Create SQLAlchemy engine
    connection_uri = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
    engine = create_engine(connection_uri)
    
    print("2. Pulling loans table into pandas...")
    df = pd.read_sql_table("loans", con=engine)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    
    # Define features to exclude from modeling
    # loan_id is unique key; issue_d, issue_year, issue_quarter are used for splitting/tracking but not training
    # default_flag is the target
    exclude_cols = ['loan_id', 'issue_d', 'issue_year', 'issue_quarter', 'default_flag']
    
    print("3. Encoding categorical columns...")
    
    # 3.1 Ordinal Encoding (Grade & Sub Grade)
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_map = {g: i for i, g in enumerate(grade_order)}
    df['grade_encoded'] = df['grade'].map(grade_map)
    
    # Sub-grades are A1-A5, B1-B5, ..., G1-G5
    sub_grades = sorted(df['sub_grade'].unique())
    sub_grade_map = {sg: i for i, sg in enumerate(sub_grades)}
    df['sub_grade_encoded'] = df['sub_grade'].map(sub_grade_map)
    
    # Drop original columns since we encoded them
    df = df.drop(columns=['grade', 'sub_grade'])
    
    # 3.2 One-Hot Encoding for other categorical columns
    categorical_cols = ['home_ownership', 'purpose', 'addr_state', 'application_type']
    
    # Keep track of categories for FastAPI validation later
    categories_metadata = {}
    for col in categorical_cols:
        categories_metadata[col] = df[col].dropna().unique().tolist()
        
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert all boolean one-hot columns and targets to integers (0 or 1)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype(int)
        
    print(f"Features after encoding: {df_encoded.shape[1] - len(exclude_cols)}")
    
    # 4. Time-Based Train/Test Split
    # Train: loans issued in 2016 and older (historically models are trained on past data)
    # Test: loans issued in 2017 and 2018 (representing future data)
    print("4. Splitting dataset based on issue_year...")
    train_df = df_encoded[df_encoded['issue_year'] <= 2016]
    test_df = df_encoded[df_encoded['issue_year'] >= 2017]
    
    print(f"   Train set size (<= 2016): {train_df.shape[0]:,} rows ({train_df.shape[0]/df_encoded.shape[0]:.1%})")
    print(f"   Test set size (>= 2017): {test_df.shape[0]:,} rows ({test_df.shape[0]/df_encoded.shape[0]:.1%})")
    
    # Prepare X and y
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['default_flag']
    X_test = test_df[feature_cols]
    y_test = test_df['default_flag']
    
    # Preserve key tracker columns for evaluation/scorecard linking
    meta_train = train_df[['loan_id', 'issue_year', 'issue_quarter']]
    meta_test = test_df[['loan_id', 'issue_year', 'issue_quarter']]
    
    # Create directories if they do not exist
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    print("5. Saving processed train/test sets...")
    X_train.to_csv(os.path.join("data", "processed", "X_train.csv"), index=False)
    y_train.to_csv(os.path.join("data", "processed", "y_train.csv"), index=False)
    X_test.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)
    y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)
    
    meta_train.to_csv(os.path.join("data", "processed", "meta_train.csv"), index=False)
    meta_test.to_csv(os.path.join("data", "processed", "meta_test.csv"), index=False)
    
    # Save preprocessing metadata (necessary for FastAPI and inference pipeline)
    w_factor = float(train_df[train_df['default_flag'] == 0].shape[0] / train_df[train_df['default_flag'] == 1].shape[0])
    preprocessor_meta = {
        "features": feature_cols,
        "grade_map": grade_map,
        "sub_grade_map": sub_grade_map,
        "categories_metadata": categories_metadata,
        "categorical_cols": categorical_cols,
        "scale_pos_weight": w_factor
    }
    joblib.dump(preprocessor_meta, os.path.join("data", "processed", "preprocessor_meta.joblib"))
    print("Preprocessors and metadata saved to data/processed/preprocessor_meta.joblib")
    print("Feature engineering completed successfully!")

if __name__ == '__main__':
    perform_feature_engineering()
