import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== PART A: PER-TYPOLOGY RECALL BREAKDOWN ON IBM DATASET ===")

# 1. Parse Patterns with specific Typology Categories
# Categories to track:
# - Fan-In / Gather (In + Gather)
# - Fan-Out / Scatter (Out)
# - Stack
# - Cycle
# - Random
# - Bipartite (for completeness)

typology_nodes = {
    'Fan-In/Gather': set(),
    'Fan-Out/Scatter': set(),
    'Stack': set(),
    'Cycle': set(),
    'Random': set(),
    'Bipartite': set()
}
all_mule_nodes = set()

current_type = None
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT'):
            parts = line.split('-')
            header_type = parts[-1].strip() if len(parts) > 1 else ''
            if 'In' in header_type or 'GATHER' in header_type:
                current_type = 'Fan-In/Gather'
            elif 'Out' in header_type:
                current_type = 'Fan-Out/Scatter'
            elif 'STACK' in header_type:
                current_type = 'Stack'
            elif 'CYCLE' in header_type:
                current_type = 'Cycle'
            elif 'RANDOM' in header_type:
                current_type = 'Random'
            elif 'BIPARTITE' in header_type:
                current_type = 'Bipartite'
            else:
                current_type = None
        elif line.startswith('END LAUNDERING ATTEMPT'):
            current_type = None
        elif current_type and line:
            parts = line.split(',')
            if len(parts) >= 5:
                src, tgt = parts[2], parts[4]
                typology_nodes[current_type].add(src)
                typology_nodes[current_type].add(tgt)
                all_mule_nodes.add(src)
                all_mule_nodes.add(tgt)

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
accounts_df['is_true_mule'] = accounts_df.index.isin(all_mule_nodes)

# Map typologies to accounts
for typ, nodes in typology_nodes.items():
    accounts_df[f'is_{typ}'] = accounts_df.index.isin(nodes)

# 3. Train/Val/Test Split (Exact same split as our frozen baseline)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

test_accounts = accounts_df.loc[X_test].copy()
print(f"Test Set Size: {len(test_accounts):,} accounts ({test_accounts['is_true_mule'].sum()} true mules)")

# Build Graph for BFS
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

# Filter suspect nodes in test set or across graph
# Note: Suspect nodes with pass_through > 0.90
suspect_test_nodes = test_accounts[test_accounts['pass_through_ratio'] > 0.90].index.tolist()
print(f"Suspect Nodes in Test Set: {len(suspect_test_nodes):,}")

# Fast 15-Hop BFS
deep_out_counts = {}
deep_in_counts = {}

for node in suspect_test_nodes:
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

test_accounts['deep_out'] = test_accounts.index.map(lambda x: deep_out_counts.get(x, 0))
test_accounts['deep_in'] = test_accounts.index.map(lambda x: deep_in_counts.get(x, 0))

# Combined Frozen Model Prediction:
# (pass_through > 0.90) & (deep_out >= 4 | deep_in >= 4) & (safe_ratio <= 0.50)
test_accounts['pred_combined'] = (
    (test_accounts['pass_through_ratio'] > 0.90) &
    ((test_accounts['deep_out'] >= 4) | (test_accounts['deep_in'] >= 4)) &
    ~(test_accounts['safe_ratio'] > 0.50)
)

# Total Test Set Metrics
tp = (test_accounts['pred_combined'] & test_accounts['is_true_mule']).sum()
total_mules = test_accounts['is_true_mule'].sum()
agg_recall = tp / total_mules if total_mules > 0 else 0
print(f"\nAGGREGATE TEST SET RECALL: {agg_recall*100:.2f}% ({tp}/{total_mules})")

print("\n--- PER-TYPOLOGY RECALL BREAKDOWN (TEST SET) ---")
for typ in ['Fan-In/Gather', 'Fan-Out/Scatter', 'Stack', 'Cycle', 'Random', 'Bipartite']:
    col = f'is_{typ}'
    typ_mules = test_accounts[test_accounts[col] & test_accounts['is_true_mule']]
    n_mules = len(typ_mules)
    if n_mules > 0:
        caught = (typ_mules['pred_combined']).sum()
        r = caught / n_mules
        print(f"  {typ:18s}: {r*100:6.2f}% ({caught:3d} / {n_mules:3d})")
    else:
        print(f"  {typ:18s}: No nodes in test set")
