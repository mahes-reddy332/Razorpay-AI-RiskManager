import sys
import os
import pandas as pd
import numpy as np
import networkx as nx

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== 1. PARSING LABELS & LOADING GRAPH ===")
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            pattern_nodes_all.add(parts[2])
            pattern_nodes_all.add(parts[4])

df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid'])

out_stats = df.groupby('Account').agg(tot_out=('Amount Paid', 'sum'))
in_stats = df.groupby('Account.1').agg(tot_in=('Amount Paid', 'sum'))

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Build NetworkX DiGraph for traversal
print("Building DiGraph...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

# Phase 1 Filter: Accounts with pass_through > 0.90
# We only run expensive BFS on these to save time.
suspect_nodes = accounts_df[accounts_df['pass_through_ratio'] > 0.90].index.tolist()
print(f"Suspect Nodes (Retention Filter Passed): {len(suspect_nodes):,}")

# Part A: Shallow Baseline (1-hop)
accounts_df['out_deg'] = accounts_df.index.map(lambda x: G.out_degree(x) if G.has_node(x) else 0)
accounts_df['in_deg'] = accounts_df.index.map(lambda x: G.in_degree(x) if G.has_node(x) else 0)
accounts_df['shallow_pred'] = (accounts_df['pass_through_ratio'] > 0.90) & ((accounts_df['out_deg'] >= 4) | (accounts_df['in_deg'] >= 4))

def calc_metrics(preds, truth):
    tp = (preds & truth).sum()
    fp = (preds & ~truth).sum()
    fn = (~preds & truth).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1, tp, fp, fn

p_s, r_s, f1_s, tp_s, fp_s, fn_s = calc_metrics(accounts_df['shallow_pred'], accounts_df['is_true_mule'])
print(f"\nSHALLOW BASELINE (1-Hop Fan-out >= 4):")
print(f"  Precision: {p_s*100:.2f}%, Recall: {r_s*100:.2f}%, F1: {f1_s:.4f} (TP: {tp_s})")

print("\n=== PART 1: DEEP TRAVERSAL (Unbounded with Cycle Protection) ===")
# We traverse downstream and upstream up to 15 hops to count total unique descendants/ancestors
# This explicitly targets deep "Stack" layer typologies
deep_out_counts = {}
deep_in_counts = {}

print("Running 15-hop BFS...")
count = 0
for node in suspect_nodes:
    if not G.has_node(node): continue
    
    # Downstream BFS (limit 15)
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
    
    # Upstream BFS (limit 15)
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
    
    count += 1
    if count % 10000 == 0: print(f"  BFS completed for {count} nodes...")

accounts_df['deep_out'] = accounts_df.index.map(lambda x: deep_out_counts.get(x, 0))
accounts_df['deep_in'] = accounts_df.index.map(lambda x: deep_in_counts.get(x, 0))

# Deep baseline: Pass-through > 0.90 AND (deep_out >= 4 OR deep_in >= 4)
accounts_df['deep_pred'] = (accounts_df['pass_through_ratio'] > 0.90) & ((accounts_df['deep_out'] >= 4) | (accounts_df['deep_in'] >= 4))

p_d, r_d, f1_d, tp_d, fp_d, fn_d = calc_metrics(accounts_df['deep_pred'], accounts_df['is_true_mule'])
print(f"\nDEEP BASELINE (15-Hop Extended Fan-out >= 4):")
print(f"  Precision: {p_d*100:.2f}%, Recall: {r_d*100:.2f}%, F1: {f1_d:.4f} (TP: {tp_d})")

print(f"\nDELTA:")
print(f"  Recall Shift: {r_s*100:.2f}% -> {r_d*100:.2f}% (+{tp_d - tp_s} additional mules caught)")
print(f"  Precision Shift: {p_s*100:.2f}% -> {p_d*100:.2f}%")
