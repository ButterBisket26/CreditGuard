import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

def evaluate_model(model_name, y_true, y_prob):
    # Calculate AUC-ROC
    auc = roc_auc_score(y_true, y_prob)
    
    # Calculate KS Statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = np.max(tpr - fpr)
    
    # Classification metrics at default 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    return {
        "Model": model_name,
        "AUC-ROC": auc,
        "KS-Statistic": ks_stat,
        "Precision (Default)": precision,
        "Recall (Default)": recall,
        "F1-Score": f1
    }

def train_and_evaluate():
    print("1. Loading processed datasets...")
    X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"))
    y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"))
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv")).values.ravel()
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Imbalance in train: {np.mean(y_train):.1%} defaults")
    
    # Calculate pos_weight for XGBoost to handle 80/20 class imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos
    print(f"   Calculated scale_pos_weight for imbalance handling: {scale_pos_weight:.2f}")
    
    os.makedirs("output", exist_ok=True)
    results = []
    
    # ── Model 1: Logistic Regression (Baseline) ──────────────────────────
    print("\n2. Training Logistic Regression baseline...")
    # Using a StandardScaler pipeline to prevent convergence warnings and speed up training significantly
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    start_time = time.time()
    lr.fit(X_train, y_train)
    print(f"   LR training complete in {time.time() - start_time:.1f}s")
    
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Logistic Regression", y_test, y_prob_lr))
    
    # ── Model 2: Decision Tree ───────────────────────────────────────────
    print("\n3. Training Decision Tree (tuned for interpretability)...")
    dt = DecisionTreeClassifier(class_weight='balanced', max_depth=5, min_samples_leaf=100, random_state=42)
    start_time = time.time()
    dt.fit(X_train, y_train)
    print(f"   DT training complete in {time.time() - start_time:.1f}s")
    
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Decision Tree", y_test, y_prob_dt))
    
    # Export Decision Tree rules for documentation
    tree_rules = export_text(dt, feature_names=list(X_train.columns))
    with open(os.path.join("output", "decision_tree.txt"), "w") as f:
        f.write(tree_rules)
    print("   Saved Decision Tree rules to output/decision_tree.txt")
    
    # ── Model 3: XGBoost (Champion Model) ────────────────────────────────
    print("\n4. Training XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    start_time = time.time()
    xgb.fit(X_train, y_train)
    print(f"   XGBoost training complete in {time.time() - start_time:.1f}s")
    
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("XGBoost", y_test, y_prob_xgb))
    
    # Save the final XGBoost model
    joblib.dump(xgb, os.path.join("output", "model.joblib"))
    print("   Saved final XGBoost model to output/model.joblib")
    
    # ── Comparison and Metrics Reporting ───────────────────────────────
    print("\n5. Model Evaluation Results on Test Set:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join("output", "model_comparison.csv"), index=False)
    
    # Plot ROC curves
    print("\n6. Plotting evaluation curves...")
    plt.figure(figsize=(10, 8))
    for name, prob in [("Logistic Regression", y_prob_lr), ("Decision Tree", y_prob_dt), ("XGBoost", y_prob_xgb)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("output", "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ── SHAP Explainability ─────────────────────────────────────────────
    try:
        print("\n7. Generating SHAP Explainability on final XGBoost model...")
        # Sample test set to calculate SHAP fast (SHAP on all 480k rows would take hours)
        shap_sample_size = 5000
        np.random.seed(42)
        sample_indices = np.random.choice(X_test.shape[0], shap_sample_size, replace=False)
        X_test_sample = X_test.iloc[sample_indices]
        
        start_time = time.time()
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer(X_test_sample)
        print(f"   Calculated SHAP values in {time.time() - start_time:.1f}s")
        
        # Save SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, show=False)
        plt.title("SHAP Feature Importance (XGBoost)", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join("output", "shap_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved SHAP summary plot to output/shap_summary.png")
    except Exception as shap_err:
        print(f"\n[Warning] SHAP calculation failed due to package compatibility: {shap_err}")
        print("   Falling back to native XGBoost Feature Importance...")
        
        # Fallback: Plot top 20 native feature importances from XGBoost
        importances = xgb.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title("Top 20 Feature Importances (XGBoost)")
        plt.barh(range(20), importances[indices][::-1], align="center", color="#2C5E8A")
        plt.yticks(range(20), [X_train.columns[i] for i in indices][::-1])
        plt.xlabel("Relative Importance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("output", "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved native feature importance plot to output/feature_importance.png")

if __name__ == '__main__':
    train_and_evaluate()
