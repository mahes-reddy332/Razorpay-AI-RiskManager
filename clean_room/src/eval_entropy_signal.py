import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== BOUNDED TEST: OUTBOUND AMOUNT ENTROPY SIGNAL (IBM DATASET) ===")

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

# 2. Compute Amount Entropy per Account
# Mules move structured, repetitive amounts (low entropy), while legitimate businesses have organic, varied amounts (high entropy).
print("Computing Shannon Entropy of outbound transaction amounts...")

# Bin amounts using log-scale bins to capture orders of magnitude
bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, np.inf]
df['amt_bin'] = pd.cut(df['Amount Paid'], bins=bins, labels=False)

# Compute entropy per source account
def compute_entropy(series):
    if len(series) < 2:
        return 0.0 # Single txn accounts
    counts = series.value_counts(normalize=True)
    return float(entropy(counts, base=2))

entropy_stats = df.groupby('Account')['amt_bin'].agg(
    out_entropy=compute_entropy,
    out_tx_count='count'
)

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_unique_dest=('Account.1', 'nunique')
)
out_stats = out_stats.join(entropy_stats)

in_stats = df.groupby('Account.1').agg(tot_in=('Amount Paid', 'sum'))

df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']
safe_in_stats = df.groupby('Account.1')['safe_amount'].sum()

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = safe_in_stats.reindex(accounts_df.index).fillna(0.0)
accounts_df['out_entropy'] = accounts_df['out_entropy'].fillna(0.0)

# Feature 1: Pass-Through Ratio
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
# Feature 2: Safe Payment Ratio
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
# Feature 3: Destination Concentration Ratio
accounts_df['dest_concentration_ratio'] = accounts_df['out_unique_dest'] / np.maximum(accounts_df['out_tx_count'], 1)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

print("\n--- ENTROPY DISTRIBUTION ---")
mule_accounts = accounts_df[accounts_df['is_true_mule'] & (accounts_df['out_tx_count'] >= 3)]
legit_accounts = accounts_df[~accounts_df['is_true_mule'] & (accounts_df['out_tx_count'] >= 3)]
print(f"Mules (>=3 txns) Mean Outbound Entropy: {mule_accounts['out_entropy'].mean():.3f} (Median: {mule_accounts['out_entropy'].median():.3f})")
print(f"Legit (>=3 txns) Mean Outbound Entropy: {legit_accounts['out_entropy'].mean():.3f} (Median: {legit_accounts['out_entropy'].median():.3f})")

# Train/Val/Test Split (Exact same split as all prior runs)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

val_df = accounts_df.loc[X_val].copy()
test_df = accounts_df.loc[X_test].copy()

# Build Graph & compute 15-hop BFS for Validation & Test
print("Building Graph & Computing 15-Hop Traversal...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

def compute_bfs_reach(target_df):
    deep_out_counts = {}
    deep_in_counts = {}
    for node in target_df.index:
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
    return target_df.index.map(lambda x: deep_out_counts.get(x, 0)), target_df.index.map(lambda x: deep_in_counts.get(x, 0))

val_df['deep_out'], val_df['deep_in'] = compute_bfs_reach(val_df)
test_df['deep_out'], test_df['deep_in'] = compute_bfs_reach(test_df)

# Base Pipeline definition on Validation
val_df['tier0_gate'] = (
    (val_df['pass_through_ratio'] > 0.90) |
    ((val_df['pass_through_ratio'] > 0.50) & (val_df['dest_concentration_ratio'] <= 0.70) & (val_df['out_tx_count'] >= 3))
)
val_df['auto_freeze_base'] = (
    val_df['tier0_gate'] &
    ((val_df['deep_out'] >= 4) | (val_df['deep_in'] >= 4)) &
    ~(val_df['safe_ratio'] > 0.50)
)
val_df['extreme_top_base'] = (
    ((val_df['deep_out'] >= 15) | (val_df['deep_in'] >= 15)) &
    ~(val_df['safe_ratio'] > 0.50)
)
val_df['total_base'] = val_df['auto_freeze_base'] | val_df['extreme_top_base']

p_val_base = precision_score(val_df['is_true_mule'], val_df['total_base'], zero_division=0)
r_val_base = recall_score(val_df['is_true_mule'], val_df['total_base'], zero_division=0)
f1_val_base = f1_score(val_df['is_true_mule'], val_df['total_base'], zero_division=0)
print(f"\nValidation Baseline Pipeline -> Precision: {p_val_base*100:.2f}%, Recall: {r_val_base*100:.2f}%, F1: {f1_val_base:.4f}")

# 3. Tune Entropy Threshold (E) on Validation Split
# Rule: If an account has out_tx_count >= 3 AND out_entropy > E (high entropy, organic varied amounts), exempt it from alert!
print("\nTuning Entropy Exemption Threshold (E) on Validation Split...")
best_e = None
best_f1 = f1_val_base
best_p = p_val_base

for e_thresh in [0.5, 1.0, 1.5, 2.0, 2.5]:
    # Exemption: high entropy (> E) accounts with >= 3 txns are cleared as organic merchants
    val_exempt = (val_df['out_tx_count'] >= 3) & (val_df['out_entropy'] > e_thresh)
    val_pred_entropy = val_df['total_base'] & ~val_exempt
    
    p = precision_score(val_df['is_true_mule'], val_pred_entropy, zero_division=0)
    r = recall_score(val_df['is_true_mule'], val_pred_entropy, zero_division=0)
    f1 = f1_score(val_df['is_true_mule'], val_pred_entropy, zero_division=0)
    fp = (val_pred_entropy & ~val_df['is_true_mule']).sum()
    tp = (val_pred_entropy & val_df['is_true_mule']).sum()
    
    print(f"  Tune E > {e_thresh:.1f} (Exempt Organic) -> TP: {tp}, FP: {fp:,}, Precision: {p*100:5.2f}%, Recall: {r*100:5.2f}%, F1: {f1:.5f}")
    if f1 > best_f1:
        best_f1 = f1
        best_e = e_thresh
        best_p = p

print(f"\nOptimal Entropy Threshold: E = {best_e}")

# 4. Evaluate Once on Untouched Test Split
test_df['tier0_gate'] = (
    (test_df['pass_through_ratio'] > 0.90) |
    ((test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= 0.70) & (test_df['out_tx_count'] >= 3))
)
test_df['auto_freeze_base'] = (
    test_df['tier0_gate'] &
    ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['extreme_top_base'] = (
    ((test_df['deep_out'] >= 15) | (test_df['deep_in'] >= 15)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['total_base'] = test_df['auto_freeze_base'] | test_df['extreme_top_base']

p_test_base = precision_score(test_df['is_true_mule'], test_df['total_base'], zero_division=0)
r_test_base = recall_score(test_df['is_true_mule'], test_df['total_base'], zero_division=0)
f1_test_base = f1_score(test_df['is_true_mule'], test_df['total_base'], zero_division=0)
fp_test_base = (test_df['total_base'] & ~test_df['is_true_mule']).sum()
tp_test_base = (test_df['total_base'] & test_df['is_true_mule']).sum()

if best_e is not None:
    test_exempt = (test_df['out_tx_count'] >= 3) & (test_df['out_entropy'] > best_e)
    test_pred_entropy = test_df['total_base'] & ~test_exempt
else:
    test_pred_entropy = test_df['total_base']

p_test_new = precision_score(test_df['is_true_mule'], test_df['pred_entropy'] if 'pred_entropy' in test_df else test_pred_entropy, zero_division=0)
r_test_new = recall_score(test_df['is_true_mule'], test_pred_entropy, zero_division=0)
f1_test_new = f1_score(test_df['is_true_mule'], test_pred_entropy, zero_division=0)
fp_test_new = (test_pred_entropy & ~test_df['is_true_mule']).sum()
tp_test_new = (test_pred_entropy & test_df['is_true_mule']).sum()

print("\n==========================================================================")
print("=== FINAL TEST SPLIT RESULTS: AMOUNT ENTROPY SIGNAL ===")
print("==========================================================================")
print(f"1. Baseline Frozen Pipeline (Without Entropy):")
print(f"   * True Positives:  {tp_test_base} / 619 ({r_test_base*100:.2f}% Recall)")
print(f"   * False Positives: {fp_test_base:,}")
print(f"   * Precision:       {p_test_base*100:.2f}%")
print(f"   * F1 Score:        {f1_test_base:.4f}")

print(f"\n2. With Outbound Amount Entropy (Exempting E > {best_e}):")
print(f"   * True Positives:  {tp_test_new} / 619 ({r_test_new*100:.2f}% Recall)")
print(f"   * False Positives: {fp_test_new:,}")
print(f"   * Precision:       {p_test_new*100:.2f}%")
print(f"   * F1 Score:        {f1_test_new:.4f}")

print(f"\nDelta:")
print(f"   * Precision Shift: {p_test_base*100:.2f}% -> {p_test_new*100:.2f}% ({p_test_new*100 - p_test_base*100:+.2f}%)")
print(f"   * Recall Shift:    {r_test_base*100:.2f}% -> {r_test_new*100:.2f}% ({r_test_new*100 - r_test_base*100:+.2f}%)")
print(f"   * False Positives: {fp_test_base - fp_test_new:,} false positives eliminated")
print("==========================================================================")
