import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== 1. INGESTION & GROUND TRUTH MAPPING ===")
# Parse patterns
pattern_nodes_all = set()
pattern_nodes_intermediate = set()
cycle_nodes = set()

current_header = ""
current_chain = []
all_chains = []

with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT'):
            current_header = line
            current_chain = []
        elif line.startswith('END LAUNDERING ATTEMPT'):
            if current_chain:
                all_chains.append((current_header, current_chain))
            current_chain = []
        elif current_header and line:
            parts = line.split(',')
            current_chain.append((parts[2], parts[4]))

for header, txns in all_chains:
    srcs = set(t[0] for t in txns)
    tgts = set(t[1] for t in txns)
    pattern_nodes_all.update(srcs.union(tgts))
    inter = srcs.intersection(tgts)
    pattern_nodes_intermediate.update(inter)
    if "CYCLE" in header:
        cycle_nodes.update(srcs.union(tgts))

# Load data and build graph
print("Reading CSV and building graph...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format', 'Is Laundering'])

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_deg=('Account.1', 'nunique'),
    ach_out=('Payment Format', lambda x: (x == 'ACH').sum())
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    in_deg=('Account', 'nunique')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)

# Feature 1: Pass-through (1 - retention)
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['retention_ratio'] = 1.0 - accounts_df['pass_through_ratio']

# Feature 2: Topology Degree (normalized)
accounts_df['max_degree'] = np.maximum(accounts_df['out_deg'], accounts_df['in_deg'])

# Composite Risk Formula (FIXED FEATURE WEIGHTS from our validated model):
# base: pass_through > 0.95 -> weight 0.4
# topology: max_degree >= 4 -> weight 0.6
raw_scores = (accounts_df['pass_through_ratio'] > 0.95).astype(float) * 0.4 + \
             (accounts_df['max_degree'] >= 4).astype(float) * 0.6

accounts_df['composite_score'] = raw_scores
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

print(f"Total Eligible Accounts: {len(accounts_df):,}")
print(f"Total Ground Truth Mules: {accounts_df['is_true_mule'].sum():,}")

# === 1. RECALIBRATE THRESHOLD ON 60/20/20 SPLIT ===
print("\n=== FIX 1: RECALIBRATION ON IBM 60/20/20 SPLIT ===")
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

train_df = accounts_df.loc[X_train]
val_df = accounts_df.loc[X_val]
test_df = accounts_df.loc[X_test]

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Sweep threshold on Validation split only
thresholds = [0.4, 0.6, 0.8, 1.0]
best_f1 = 0
best_thresh = 1.0

for t in thresholds:
    val_preds = val_df['composite_score'] >= t
    f1 = f1_score(val_df['is_true_mule'], val_preds, zero_division=0)
    p = precision_score(val_df['is_true_mule'], val_preds, zero_division=0)
    r = recall_score(val_df['is_true_mule'], val_preds, zero_division=0)
    print(f"  Val Threshold {t:.1f} -> Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\nFrozen Threshold selected from Validation: {best_thresh}")

# Evaluate ONCE on untouched Test set
test_preds = test_df['composite_score'] >= best_thresh
p_test = precision_score(test_df['is_true_mule'], test_preds, zero_division=0)
r_test = recall_score(test_df['is_true_mule'], test_preds, zero_division=0)
f1_test = f1_score(test_df['is_true_mule'], test_preds, zero_division=0)
cm = confusion_matrix(test_df['is_true_mule'], test_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\nFINAL UNTOUCHED TEST SET METRICS (Threshold = {best_thresh}):")
print(f"  True Positives : {tp:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Negatives : {tn:,}")
print(f"  Precision: {p_test*100:.2f}%")
print(f"  Recall   : {r_test*100:.2f}%")
print(f"  F1 Score : {f1_test:.4f}")

# === 2. CYCLE DETECTION FALSE POSITIVE AUDIT ===
print("\n=== FIX 2: CYCLE DETECTION FALSE POSITIVE AUDIT ===")
# Build NetworkX DiGraph on sample to count SCCs / cycles across entire population
G = nx.DiGraph()
for src, tgt in zip(df['Account'], df['Account.1']):
    G.add_edge(src, tgt)

sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
all_cycle_nodes = set()
for c in sccs:
    all_cycle_nodes.update(c)

cycle_tp = len(all_cycle_nodes.intersection(pattern_nodes_all))
cycle_fp = len(all_cycle_nodes - pattern_nodes_all)
cycle_precision = cycle_tp / len(all_cycle_nodes) if len(all_cycle_nodes) > 0 else 0

print(f"Total Nodes in Strongly Connected Components (Closed Loops): {len(all_cycle_nodes):,}")
print(f"  True Positives (Laundering Cycles): {cycle_tp:,}")
print(f"  False Positives (Legitimate Closed Loops/Refunds): {cycle_fp:,}")
print(f"  Cycle Detection Precision: {cycle_precision*100:.2f}%")

# === 3. RAW ACH VOLUME COUNTS ===
print("\n=== FIX 3: ACH RAW COUNTS RECONCILIATION ===")
ach_df = df[df['Payment Format'] == 'ACH']
non_ach_df = df[df['Payment Format'] != 'ACH']

legit_ach_count = (ach_df['Is Laundering'] == 0).sum()
launder_ach_count = (ach_df['Is Laundering'] == 1).sum()
total_launder_count = (df['Is Laundering'] == 1).sum()

print(f"Total ACH Transactions: {len(ach_df):,}")
print(f"  - Legitimate ACH Transactions: {legit_ach_count:,} ({legit_ach_count/len(ach_df)*100:.2f}%)")
print(f"  - Laundering ACH Transactions: {launder_ach_count:,} ({launder_ach_count/len(ach_df)*100:.2f}%)")
print(f"Total Laundering Transactions across all formats: {total_launder_count:,}")
print(f"Fraction of all laundering done via ACH: {launder_ach_count/total_launder_count*100:.2f}%")
