import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== PART 2: PAYMENT FORMAT SAFELIST (IBM Data) ===")

# 1. Parse Patterns
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            pattern_nodes_all.add(parts[2])
            pattern_nodes_all.add(parts[4])

# 2. Load Data and compute safe inbound volume
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_deg=('Account.1', 'nunique')
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    in_deg=('Account', 'nunique'),
    safe_in=('safe_amount', 'sum')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)

# Feature 1: Pass-through
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
# Feature 2: Topology Degree
accounts_df['max_degree'] = np.maximum(accounts_df['out_deg'], accounts_df['in_deg'])
# Feature 3: Safe Ratio
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)

# Composite Risk Formula (Baseline)
raw_scores = (accounts_df['pass_through_ratio'] > 0.95).astype(float) * 0.4 + \
             (accounts_df['max_degree'] >= 4).astype(float) * 0.6

accounts_df['base_score'] = raw_scores
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Train/Val/Test Split (Exact same seed as eval_ibm_recalibrated.py)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

val_df = accounts_df.loc[X_val].copy()
test_df = accounts_df.loc[X_test].copy()

# Baseline Threshold Tuning (Validation)
best_base_f1, best_base_thresh = 0, 1.0
for th in [0.4, 0.6, 0.8, 1.0]:
    preds = val_df['base_score'] >= th
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_base_f1:
        best_base_f1, best_base_thresh = f1, th

# Safelist Tuning (Validation)
# Rule: If safe_ratio > 0.5, subtract discount from score
best_safe_f1, best_discount = 0, 0.0
for discount in [0.3, 0.4, 0.5, 0.6]:
    discounted_scores = val_df['base_score'] - (val_df['safe_ratio'] > 0.5).astype(float) * discount
    preds = discounted_scores >= best_base_thresh
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_safe_f1:
        best_safe_f1, best_discount = f1, discount

print(f"Optimal Baseline Threshold: {best_base_thresh}")
print(f"Optimal Safelist Discount: -{best_discount}")

# Evaluate on Test Split
# 1. Baseline
base_preds = test_df['base_score'] >= best_base_thresh
p_base = precision_score(y_test, base_preds, zero_division=0)
r_base = recall_score(y_test, base_preds, zero_division=0)

# 2. With Safelist
final_scores = test_df['base_score'] - (test_df['safe_ratio'] > 0.5).astype(float) * best_discount
safe_preds = final_scores >= best_base_thresh
p_safe = precision_score(y_test, safe_preds, zero_division=0)
r_safe = recall_score(y_test, safe_preds, zero_division=0)

print("\n=== TEST SET RESULTS ===")
print(f"Baseline -> Precision: {p_base*100:.2f}%, Recall: {r_base*100:.2f}%")
print(f"With Safelist -> Precision: {p_safe*100:.2f}%, Recall: {r_safe*100:.2f}%")
print(f"Precision Delta: +{(p_safe - p_base)*100:.2f}%")
print(f"Recall Delta: +{(r_safe - r_base)*100:.2f}%")
