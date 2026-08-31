import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to import detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import MVPDetector

def run_evaluation():
    print("Initializing MVP Detector for Formal Evaluation...")
    det = MVPDetector('data/accounts.csv', 'data/transactions.csv')
    det.build_graph()
    det.score_accounts()
    df = det.df_results
    
    # 1. STRICT ACCOUNT-LEVEL TRAIN/TEST SPLIT
    # We split based purely on account IDs (80/20) to ensure zero data leakage.
    all_accounts = df['account_id'].unique()
    train_ids, test_ids = train_test_split(all_accounts, test_size=0.2, random_state=42)
    
    test_df = df[df['account_id'].isin(test_ids)]
    
    print(f"\n--- TEST SET COMPOSITION ---")
    print(f"Total Accounts (Forwarding nodes): {len(test_df)}")
    print(f"True Mules in Test Set: {test_df['is_mule'].sum()}")
    print(f"Legit Accounts in Test Set: {len(test_df) - test_df['is_mule'].sum()}")
    
    # 2. COMPUTE METRICS
    threshold = 0.5
    y_true = test_df['is_mule']
    y_scores = test_df['risk_score']
    y_pred = y_scores > threshold
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(f"\n--- EVALUATION METRICS (Threshold > {threshold}) ---")
    print(f"True Positives  (Mules blocked): {tp}")
    print(f"False Positives (Legit blocked): {fp}")
    print(f"False Negatives (Mules missed):  {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    # 3. PLOT PRECISION-RECALL CURVE
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_scores)
    
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.step(recalls, precisions, where='post', color='b', linewidth=2)
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall (Mules Caught)')
    plt.ylabel('Precision (Accuracy of Flags)')
    plt.title('Precision-Recall Curve - MVP Rule-Based Detector')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate the specific threshold point
    plt.plot(recall, precision, 'ro')
    plt.annotate(f' Threshold > 0.5\n (P:{precision:.2f}, R:{recall:.2f})',
                 (recall, precision), textcoords="offset points", xytext=(-20,-40), ha='center', color='red')
                 
    plt.savefig('outputs/pr_curve.png', dpi=300, bbox_inches='tight')
    print("\nPR Curve saved to outputs/pr_curve.png")

if __name__ == "__main__":
    run_evaluation()
