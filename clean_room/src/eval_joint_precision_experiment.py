import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== BOUNDED PRECISION EXPERIMENT: JOINT TIER 0 PROMOTION ===")

# 1. Parse Patterns
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            if len(parts) >= 5:
                pattern_nodes_all.add(parts[2])
                pattern_nodes_all.add(parts[4])

print("Loading CSV...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_tx_count=('Amount Paid', 'count'),
    out_unique_dest=('Account.1', 'nunique')
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    safe_in=('safe_amount', 'sum')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = accounts_df['safe_in'].fillna(0.0)

# Feature 1: Pass-Through Ratio
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
# Feature 2: Safe Payment Ratio
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
# Feature 3: Destination Concentration Ratio
accounts_df['dest_concentration_ratio'] = accounts_df['out_unique_dest'] / np.maximum(accounts_df['out_tx_count'], 1)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Train/Val/Test Split (Exact same seed & split as all prior runs)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

val_df = accounts_df.loc[X_val].copy()
test_df = accounts_df.loc[X_test].copy()

# Focus specifically on the partial-retention set: pass_through in (0.50, 0.90] with out_tx_count >= 3
# In the previous run, this used `dest_concentration_ratio <= 0.70` independently of safe_ratio.
# Here we test joint tightening: e.g. require safe_ratio <= S_tight (e.g. 0.0, 0.1, 0.2) OR stricter C on this specific subset.

print("\n--- TUNING JOINT TIER 0 ON VALIDATION SPLIT ---")
# Baseline partial promotion on validation:
val_partial_base = (
    (val_df['pass_through_ratio'] > 0.50) & 
    (val_df['pass_through_ratio'] <= 0.90) & 
    (val_df['dest_concentration_ratio'] <= 0.70) & 
    (val_df['out_tx_count'] >= 3) &
    (val_df['safe_ratio'] <= 0.50)
)
tp_val_base = (val_partial_base & val_df['is_true_mule']).sum()
fp_val_base = (val_partial_base & ~val_df['is_true_mule']).sum()
p_val_base = tp_val_base / (tp_val_base + fp_val_base) if (tp_val_base + fp_val_base) > 0 else 0
print(f"Validation Partial Promotion Subset Baseline (Independent):")
print(f"  TP: {tp_val_base} | FP: {fp_val_base:,} | Precision: {p_val_base*100:.2f}%")

# Sweep joint tightening parameters on this subset:
# Stricter safe_ratio threshold (S_max in [0.0, 0.05, 0.10, 0.20]) and stricter C (C_max in [0.3, 0.5, 0.7])
best_s = 0.50
best_c = 0.70
best_f1 = f1_score(val_df['is_true_mule'], val_partial_base, zero_division=0)
best_p = p_val_base

for c_val in [0.30, 0.50, 0.70]:
    for s_val in [0.0, 0.05, 0.10, 0.20, 0.50]:
        joint_mask = (
            (val_df['pass_through_ratio'] > 0.50) & 
            (val_df['pass_through_ratio'] <= 0.90) & 
            (val_df['dest_concentration_ratio'] <= c_val) & 
            (val_df['out_tx_count'] >= 3) &
            (val_df['safe_ratio'] <= s_val)
        )
        tp = (joint_mask & val_df['is_true_mule']).sum()
        fp = (joint_mask & ~val_df['is_true_mule']).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(val_df['is_true_mule'], joint_mask, zero_division=0)
        print(f"  Tune C<={c_val:.2f}, SafeRatio<={s_val:.2f} -> TP: {tp:2d}, FP: {fp:5,d}, Precision: {p*100:5.2f}%, F1: {f1:.5f}")
        if f1 > best_f1:
            best_f1 = f1
            best_s = s_val
            best_c = c_val
            best_p = p

print(f"\nOptimal Joint Parameters from Validation: C<={best_c}, SafeRatio<={best_s}")

# -------------------------------------------------------------
# EVALUATE ON UNTOUCHED TEST SPLIT
# -------------------------------------------------------------
print("\n=== EVALUATING ON UNTOUCHED TEST SPLIT ===")

# Build Graph & compute 15-hop BFS for test split to measure full pipeline effect
print("Building Graph & computing 15-hop BFS on test split...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

deep_out_counts = {}
deep_in_counts = {}

for node in test_df.index:
    if not G.has_node(node): continue
    v_down = set()
    q = [(node, 0)]
    while q:
        if len(v_down) >= 25: break
        curr, depth = q.pop(0)
        if depth >= 15: continue
        for succ in G.successors(curr):
            if succ not in v_down:
                v_down.add(succ)
                q.append((succ, depth + 1))
    v_up = set()
    q = [(node, 0)]
    while q:
        if len(v_up) >= 25: break
        curr, depth = q.pop(0)
        if depth >= 15: continue
        for pred in G.predecessors(curr):
            if pred not in v_up:
                v_up.add(pred)
                q.append((pred, depth + 1))
    deep_out_counts[node] = len(v_down)
    deep_in_counts[node] = len(v_up)

test_df['deep_out'] = test_df.index.map(lambda x: deep_out_counts.get(x, 0))
test_df['deep_in'] = test_df.index.map(lambda x: deep_in_counts.get(x, 0))

# 1. Baseline Full Pipeline (Independent C=0.70, Safe<=0.50):
test_df['t0_base'] = (
    (test_df['pass_through_ratio'] > 0.90) |
    ((test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= 0.70) & (test_df['out_tx_count'] >= 3))
)
test_df['auto_freeze_base'] = (
    test_df['t0_base'] &
    ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['extreme_top_base'] = (
    ((test_df['deep_out'] >= 15) | (test_df['deep_in'] >= 15)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['total_detected_base'] = test_df['auto_freeze_base'] | test_df['extreme_top_base']

p_base = precision_score(test_df['is_true_mule'], test_df['total_detected_base'], zero_division=0)
r_base = recall_score(test_df['is_true_mule'], test_df['total_detected_base'], zero_division=0)
f1_base = f1_score(test_df['is_true_mule'], test_df['total_detected_base'], zero_division=0)
fp_base = (test_df['total_detected_base'] & ~test_df['is_true_mule']).sum()
tp_base = (test_df['total_detected_base'] & test_df['is_true_mule']).sum()

# 2. Joint Tightened Pipeline:
test_df['t0_joint'] = (
    (test_df['pass_through_ratio'] > 0.90) |
    ((test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= best_c) & (test_df['safe_ratio'] <= best_s) & (test_df['out_tx_count'] >= 3))
)
test_df['auto_freeze_joint'] = (
    test_df['t0_joint'] &
    ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['total_detected_joint'] = test_df['auto_freeze_joint'] | test_df['extreme_top_base']

p_joint = precision_score(test_df['is_true_mule'], test_df['total_detected_joint'], zero_division=0)
r_joint = recall_score(test_df['is_true_mule'], test_df['total_detected_joint'], zero_division=0)
f1_joint = f1_score(test_df['is_true_mule'], test_df['total_detected_joint'], zero_division=0)
fp_joint = (test_df['total_detected_joint'] & ~test_df['is_true_mule']).sum()
tp_joint = (test_df['total_detected_joint'] & test_df['is_true_mule']).sum()

print("\n==========================================================================")
print("=== EXPERIMENT RESULTS: INDEPENDENT VS. JOINT TIER 0 (TEST SPLIT) ===")
print("==========================================================================")
print(f"1. Independent Signals Pipeline (Baseline):")
print(f"   * True Positives:  {tp_base} / 619 ({r_base*100:.2f}% Recall)")
print(f"   * False Positives: {fp_base:,}")
print(f"   * Precision:       {p_base*100:.2f}%")
print(f"   * F1 Score:        {f1_base:.4f}")

print(f"\n2. Joint Signal Pipeline (C<={best_c}, SafeRatio<={best_s}):")
print(f"   * True Positives:  {tp_joint} / 619 ({r_joint*100:.2f}% Recall)")
print(f"   * False Positives: {fp_joint:,}")
print(f"   * Precision:       {p_joint*100:.2f}%")
print(f"   * F1 Score:        {f1_joint:.4f}")

print(f"\nDelta from Joint Combination:")
print(f"   * Precision Shift: {p_base*100:.2f}% -> {p_joint*100:.2f}% ({p_joint*100 - p_base*100:+.2f}%)")
print(f"   * Recall Shift:    {r_base*100:.2f}% -> {r_joint*100:.2f}% ({r_joint*100 - r_base*100:+.2f}%)")
print(f"   * FP Reduction:    {fp_base - fp_joint:,} false positives eliminated")
print("==========================================================================")
