import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== COMBINED EVALUATION (Deep 15-Hop + Payment Safelist) ===")

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
print("Loading CSV and building graph...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(tot_out=('Amount Paid', 'sum'))
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    safe_in=('safe_amount', 'sum')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = accounts_df['safe_in'].fillna(0.0)

accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Build Graph
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

# Phase 1 Filter: Accounts with pass_through > 0.90
suspect_nodes = accounts_df[accounts_df['pass_through_ratio'] > 0.90].index.tolist()
print(f"Suspect Nodes (Retention Filter Passed): {len(suspect_nodes):,}")

# 3. Fast 15-Hop BFS
print("Running 15-hop BFS...")
deep_out_counts = {}
deep_in_counts = {}

for node in suspect_nodes:
    if not G.has_node(node): continue
    
    # Downstream BFS (limit 15 nodes found)
    visited_down = set()
    queue = [(node, 0)]
    while queue:
        if len(visited_down) >= 15: break
        curr, depth = queue.pop(0)
        if depth >= 15: continue
        for succ in G.successors(curr):
            if succ not in visited_down:
                visited_down.add(succ)
                queue.append((succ, depth + 1))
    
    # Upstream BFS (limit 15 nodes found)
    visited_up = set()
    queue = [(node, 0)]
    while queue:
        if len(visited_up) >= 15: break
        curr, depth = queue.pop(0)
        if depth >= 15: continue
        for pred in G.predecessors(curr):
            if pred not in visited_up:
                visited_up.add(pred)
                queue.append((pred, depth + 1))
                
    deep_out_counts[node] = len(visited_down)
    deep_in_counts[node] = len(visited_up)

accounts_df['deep_out'] = accounts_df.index.map(lambda x: deep_out_counts.get(x, 0))
accounts_df['deep_in'] = accounts_df.index.map(lambda x: deep_in_counts.get(x, 0))

# 4. Predictions
# Baseline 1: Deep Search Only (The 58% Recall Baseline)
accounts_df['pred_deep_only'] = (accounts_df['pass_through_ratio'] > 0.90) & ((accounts_df['deep_out'] >= 4) | (accounts_df['deep_in'] >= 4))

# Baseline 2: Deep Search + Payment Safelist
# Rule: If >50% of volume is Cheque/Credit Card, it bypasses the freeze (False)
accounts_df['pred_combined'] = accounts_df['pred_deep_only'] & ~(accounts_df['safe_ratio'] > 0.50)

def print_metrics(name, pred_col):
    p = precision_score(accounts_df['is_true_mule'], accounts_df[pred_col], zero_division=0)
    r = recall_score(accounts_df['is_true_mule'], accounts_df[pred_col], zero_division=0)
    print(f"{name}:")
    print(f"  Precision: {p*100:.2f}%")
    print(f"  Recall:    {r*100:.2f}%")
    return p, r

print("\n=== FINAL RESULTS (Full Dataset) ===")
p_deep, r_deep = print_metrics("1. Deep Traversal Only", 'pred_deep_only')
p_comb, r_comb = print_metrics("2. Deep Traversal + Payment Safelist", 'pred_combined')

print("\n=== NET DELTA FROM SAFELIST ===")
print(f"Precision Boost: +{(p_comb - p_deep)*100:.2f}%")
print(f"Recall Cost:     {(r_comb - r_deep)*100:.2f}%")
