import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== RUNNING FULL FROZEN PIPELINE END-TO-END ON UNTOUCHED IBM TEST SPLIT ===")

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
test_df = accounts_df.loc[X_test].copy()
test_mules = test_df['is_true_mule'].sum()
test_legit = len(test_df) - test_mules

print(f"Test Split Size: {len(test_df):,} accounts ({test_mules} True Mules, {test_legit:,} Legitimate Accounts)")

# Build Graph
print("Building Graph & Computing 15-Hop Traversal on Test Split...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

deep_out_counts = {}
deep_in_counts = {}

for node in test_df.index:
    if not G.has_node(node): continue
    
    # Downstream BFS (limit 25)
    v_down = set()
    q = [(node, 0)]
    while q:
        if len(v_down) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for succ in G.successors(c):
            if succ not in v_down:
                v_down.add(succ)
                q.append((succ, d + 1))
    
    # Upstream BFS (limit 25)
    v_up = set()
    q = [(node, 0)]
    while q:
        if len(v_up) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for pred in G.predecessors(c):
            if pred not in v_up:
                v_up.add(pred)
                q.append((pred, d + 1))
                
    deep_out_counts[node] = len(v_down)
    deep_in_counts[node] = len(v_up)

test_df['deep_out'] = test_df.index.map(lambda x: deep_out_counts.get(x, 0))
test_df['deep_in'] = test_df.index.map(lambda x: deep_in_counts.get(x, 0))

# -------------------------------------------------------------
# FINAL FROZEN PIPELINE DEFINITION:
# 1. Enhanced Tier 0 Gate:
#    (pass_through > 0.90) OR (pass_through > 0.50 AND dest_concentration <= 0.70 AND out_tx_count >= 3)
# 2. Tier 1 Corroborated Risk Containment:
#    Tier 0 Gate AND ((deep_out >= 4) | (deep_in >= 4)) AND (safe_ratio <= 0.50)
# 3. Decoupled Extreme-Topology Review:
#    ((deep_out >= 15) | (deep_in >= 15)) AND (safe_ratio <= 0.50) AND NOT Risk Containment
# 4. Total Pipeline Detection:
#    Risk Containment OR Decoupled Extreme-Topology Review
# -------------------------------------------------------------

test_df['tier0_gate'] = (
    (test_df['pass_through_ratio'] > 0.90) |
    ((test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= 0.70) & (test_df['out_tx_count'] >= 3))
)

test_df['auto_freeze'] = (
    test_df['tier0_gate'] &
    ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) &
    ~(test_df['safe_ratio'] > 0.50)
)

test_df['extreme_topology'] = (
    ((test_df['deep_out'] >= 15) | (test_df['deep_in'] >= 15)) &
    ~(test_df['safe_ratio'] > 0.50)
)

test_df['manual_review'] = test_df['extreme_topology'] & ~test_df['auto_freeze']
test_df['total_detected'] = test_df['auto_freeze'] | test_df['manual_review']
test_df['safe_cleared'] = ~test_df['total_detected']

# Final End-to-End Metrics
y_true = test_df['is_true_mule']
y_pred_total = test_df['total_detected']
y_pred_freeze = test_df['auto_freeze']

p_total = precision_score(y_true, y_pred_total, zero_division=0)
r_total = recall_score(y_true, y_pred_total, zero_division=0)
f1_total = f1_score(y_true, y_pred_total, zero_division=0)

p_freeze = precision_score(y_true, y_pred_freeze, zero_division=0)
r_freeze = recall_score(y_true, y_pred_freeze, zero_division=0)
f1_freeze = f1_score(y_true, y_pred_freeze, zero_division=0)

tp_tot = (y_pred_total & y_true).sum()
fp_tot = (y_pred_total & ~y_true).sum()
fn_tot = (~y_pred_total & y_true).sum()
tn_tot = (~y_pred_total & ~y_true).sum()

tp_frz = (y_pred_freeze & y_true).sum()
fp_frz = (y_pred_freeze & ~y_true).sum()

tp_rev = (test_df['manual_review'] & y_true).sum()
fp_rev = (test_df['manual_review'] & ~y_true).sum()

print("\n==========================================================================")
print("=== FINAL END-TO-END FROZEN PIPELINE METRICS (IBM TEST SPLIT) ===")
print("==========================================================================")

print("\n1. TOTAL COMBINED PIPELINE PERFORMANCE (Risk Containment + L2 Review):")
print(f"   * True Positives (Caught Mules):     {tp_tot:6d} / {test_mules} ({r_total*100:.2f}% Recall)")
print(f"   * False Positives (Total Flagged):   {fp_tot:6,d} / {test_legit:,}")
print(f"   * False Negatives (Missed Mules):    {fn_tot:6d} / {test_mules}")
print(f"   * True Negatives (Cleared Safe):     {tn_tot:6,d} / {test_legit:,}")
print(f"   * Precision:                         {p_total*100:.2f}%")
print(f"   * Recall:                            {r_total*100:.2f}%")
print(f"   * F1 Score:                          {f1_total:.4f}")

print("\n2. OPERATIONAL ACTION BREAKDOWN:")
print(f"   A. RISK_CONTAINMENT_REQUIRED ACTION (`HIGH_RISK`):")
print(f"      - True Mules Contained for Risk:         {tp_frz:6d} ({tp_frz/test_mules*100:5.2f}% of all mules)")
print(f"      - Legitimate Accounts Frozen:     {fp_frz:6,d} ({fp_frz/test_legit*100:5.2f}% of all legit)")
print(f"      - Risk Containment Precision:          {p_freeze*100:.2f}%")
print(f"      - Risk Containment Recall:             {r_freeze*100:.2f}%")
print(f"      - Risk Containment F1:                 {f1_freeze:.4f}")
print(f"\n   B. ASYNCHRONOUS REVIEW QUEUE (`MANUAL_REVIEW_REQUIRED`):")
print(f"      - True Mules Sent to Review:      {tp_rev:6d} ({tp_rev/test_mules*100:5.2f}% of all mules)")
print(f"      - High-Degree Clearing in Review: {fp_rev:6,d} ({fp_rev/test_legit*100:5.2f}% of all legit)")
print(f"      - Legitimate Accounts Saved:      {fp_rev:,} innocent accounts safely routed to review (0% contained for risk)")

print("==========================================================================")
