import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== 1. INGESTION & TYPOLOGY GROUND TRUTH MAPPING ===")
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

# Load data and compute metrics
print("Loading CSV and building graph...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(tot_out=('Amount Paid', 'sum'), out_deg=('Account.1', 'nunique'))
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    in_deg=('Account', 'nunique'),
    safe_in=('safe_amount', 'sum')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = accounts_df['safe_in'].fillna(0.0)
accounts_df['out_deg'] = accounts_df['out_deg'].fillna(0)
accounts_df['in_deg'] = accounts_df['in_deg'].fillna(0)

accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['is_true_mule'] = accounts_df.index.isin(all_mule_nodes)

# Map typologies
for typ, nodes in typology_nodes.items():
    accounts_df[f'is_{typ}'] = accounts_df.index.isin(nodes)

# Build Graph
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

# Train/Val/Test Split (60/20/20)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

train_df = accounts_df.loc[X_train].copy()
val_df = accounts_df.loc[X_val].copy()
test_df = accounts_df.loc[X_test].copy()

print(f"Total Accounts: {len(accounts_df):,} | Test Set: {len(test_df):,} accounts ({test_df['is_true_mule'].sum()} true mules)")

# Compute 15-Hop BFS reach for test set
print("Computing 15-hop BFS for test set...")
deep_out_counts = {}
deep_in_counts = {}

# Compute for all test set accounts or suspect accounts
for node in test_df.index:
    if not G.has_node(node): continue
    
    # Downstream BFS
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
    
    # Upstream BFS
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

test_df['deep_out'] = test_df.index.map(lambda x: deep_out_counts.get(x, 0))
test_df['deep_in'] = test_df.index.map(lambda x: deep_in_counts.get(x, 0))

# Also compute Tier 0 & Tier 1 flags
# Tier 0 Gate: pass_through_ratio > 0.90
test_df['tier0_pass'] = test_df['pass_through_ratio'] > 0.90
# Tier 1 Topology Flag: deep_out >= 4 or deep_in >= 4, and safe_ratio <= 0.50
test_df['tier1_flag'] = ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) & ~(test_df['safe_ratio'] > 0.50)
test_df['combined_flag'] = test_df['tier0_pass'] & test_df['tier1_flag']

print("\n============================================================")
print("=== PART A: PER-TYPOLOGY FUNNEL DECOMPOSITION (TEST SET) ===")
print("============================================================")
print(f"{'Typology':<18} | {'Mules':<6} | {'Tier 0 Pass Rate':<18} | {'Tier 1 Catch Rate (Given T0)':<28} | {'Overall Recall':<14}")
print("-" * 92)

for typ in ['Fan-In/Gather', 'Fan-Out/Scatter', 'Stack', 'Cycle', 'Random', 'Bipartite']:
    col = f'is_{typ}'
    mules_typ = test_df[test_df[col] & test_df['is_true_mule']]
    total = len(mules_typ)
    if total == 0: continue
    
    t0_passed = (mules_typ['tier0_pass']).sum()
    t0_rate = (t0_passed / total) * 100
    
    # Tier 1 given Tier 0
    t0_mules = mules_typ[mules_typ['tier0_pass']]
    t1_caught_given_t0 = (t0_mules['tier1_flag']).sum()
    t1_rate = (t1_caught_given_t0 / t0_passed * 100) if t0_passed > 0 else 0
    
    overall_caught = (mules_typ['combined_flag']).sum()
    overall_recall = (overall_caught / total) * 100
    
    print(f"{typ:<18} | {total:<6} | {t0_passed:3d}/{total:3d} ({t0_rate:5.1f}%)   | {t1_caught_given_t0:3d}/{t0_passed:3d} ({t1_rate:5.1f}%)             | {overall_caught:3d}/{total:3d} ({overall_recall:5.1f}%)")

# Overall aggregate
tot_mules = test_df['is_true_mule'].sum()
tot_t0 = (test_df[test_df['is_true_mule']]['tier0_pass']).sum()
tot_t1_given_t0 = (test_df[test_df['is_true_mule'] & test_df['tier0_pass']]['tier1_flag']).sum()
tot_overall = (test_df[test_df['is_true_mule']]['combined_flag']).sum()

print("-" * 92)
print(f"{'AGGREGATE':<18} | {tot_mules:<6} | {tot_t0:3d}/{tot_mules:3d} ({tot_t0/tot_mules*100:5.1f}%)   | {tot_t1_given_t0:3d}/{tot_t0:3d} ({tot_t1_given_t0/tot_t0*100:5.1f}%)             | {tot_overall:3d}/{tot_mules:3d} ({tot_overall/tot_mules*100:5.1f}%)")

# ============================================================
# === PART B: EXTREME TOPOLOGY DECOUPLING ===
# ============================================================
print("\n============================================================")
print("=== PART B: EXTREME TOPOLOGY DECOUPLING (VALIDATION & TEST) ===")
print("============================================================")

# Compute deep_out / deep_in on Validation set to tune high-degree bound D
val_nodes = val_df.index
val_deep_out = {}
val_deep_in = {}

for node in val_nodes:
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
    val_deep_out[node] = len(v_down)
    val_deep_in[node] = len(v_up)

val_df['deep_out'] = val_df.index.map(lambda x: val_deep_out.get(x, 0))
val_df['deep_in'] = val_df.index.map(lambda x: val_deep_in.get(x, 0))
val_df['tier0_pass'] = val_df['pass_through_ratio'] > 0.90
val_df['tier1_standard'] = ((val_df['deep_out'] >= 4) | (val_df['deep_in'] >= 4)) & ~(val_df['safe_ratio'] > 0.50)
val_df['base_flag'] = val_df['tier0_pass'] & val_df['tier1_standard']

# Tuning Extreme Degree Bound D in [10, 12, 15, 20]
# Decoupled Rule: Flag if (Tier 0 & Tier 1) OR (Extreme Topology: max(deep_out, deep_in) >= D and safe_ratio <= 0.50)
best_d = None
best_val_f1 = f1_score(val_df['is_true_mule'], val_df['base_flag'], zero_division=0)
best_val_p = precision_score(val_df['is_true_mule'], val_df['base_flag'], zero_division=0)
best_val_r = recall_score(val_df['is_true_mule'], val_df['base_flag'], zero_division=0)

print(f"Validation Baseline (Standard Model) -> Precision: {best_val_p*100:.2f}%, Recall: {best_val_r*100:.2f}%, F1: {best_val_f1:.4f}")

for D in [10, 12, 15, 20]:
    val_extreme = ((val_df['deep_out'] >= D) | (val_df['deep_in'] >= D)) & ~(val_df['safe_ratio'] > 0.50)
    val_decoupled_pred = val_df['base_flag'] | val_extreme
    
    p = precision_score(val_df['is_true_mule'], val_decoupled_pred, zero_division=0)
    r = recall_score(val_df['is_true_mule'], val_decoupled_pred, zero_division=0)
    f = f1_score(val_df['is_true_mule'], val_decoupled_pred, zero_division=0)
    print(f"  Tune D={D:2d} -> Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F1: {f:.4f} (FP: {(val_decoupled_pred & ~val_df['is_true_mule']).sum():,})")
    
    if f > best_val_f1:
        best_val_f1 = f
        best_d = D

if best_d is None:
    best_d = 15 # Default high bound if F1 is flat/penalized by FP
    print(f"\nSelection: Setting Frozen D={best_d} for conservative evaluation.")
else:
    print(f"\nOptimal Frozen Decoupling Threshold: D={best_d}")

# Evaluate on Untouched Test Split
# 1. Baseline Test Metrics
p_base = precision_score(test_df['is_true_mule'], test_df['combined_flag'], zero_division=0)
r_base = recall_score(test_df['is_true_mule'], test_df['combined_flag'], zero_division=0)
f1_base = f1_score(test_df['is_true_mule'], test_df['combined_flag'], zero_division=0)
fp_base = (test_df['combined_flag'] & ~test_df['is_true_mule']).sum()
tp_base = (test_df['combined_flag'] & test_df['is_true_mule']).sum()

# 2. Decoupled Test Metrics
test_extreme = ((test_df['deep_out'] >= best_d) | (test_df['deep_in'] >= best_d)) & ~(test_df['safe_ratio'] > 0.50)
test_decoupled_pred = test_df['combined_flag'] | test_extreme

p_dec = precision_score(test_df['is_true_mule'], test_decoupled_pred, zero_division=0)
r_dec = recall_score(test_df['is_true_mule'], test_decoupled_pred, zero_division=0)
f1_dec = f1_score(test_df['is_true_mule'], test_decoupled_pred, zero_division=0)
fp_dec = (test_decoupled_pred & ~test_df['is_true_mule']).sum()
tp_dec = (test_decoupled_pred & test_df['is_true_mule']).sum()

print("\n=== FINAL TEST SPLIT PERFORMANCE COMPARISON ===")
print(f"Baseline Combined Model:")
print(f"  Precision: {p_base*100:.2f}% | Recall: {r_base*100:.2f}% | F1: {f1_base:.4f} (TP: {tp_base}, FP: {fp_base:,})")
print(f"\nDecoupled Extreme-Topology Model (D >= {best_d}):")
print(f"  Precision: {p_dec*100:.2f}% | Recall: {r_dec*100:.2f}% | F1: {f1_dec:.4f} (TP: {tp_dec}, FP: {fp_dec:,})")
print(f"\nDelta:")
print(f"  Recall Shift:    {r_base*100:.2f}% -> {r_dec*100:.2f}% (+{tp_dec - tp_base} mules)")
print(f"  Precision Shift: {p_base*100:.2f}% -> {p_dec*100:.2f}%")
print(f"  New False Positives Introduced: +{fp_dec - fp_base:,}")

print("\n--- PER-TYPOLOGY RECALL SHIFT (TEST SET) ---")
for typ in ['Fan-In/Gather', 'Fan-Out/Scatter', 'Stack', 'Cycle', 'Random', 'Bipartite']:
    col = f'is_{typ}'
    mules_typ = test_df[test_df[col] & test_df['is_true_mule']]
    total = len(mules_typ)
    if total == 0: continue
    
    r_old = (mules_typ['combined_flag']).sum() / total * 100
    r_new = (test_decoupled_pred.loc[mules_typ.index]).sum() / total * 100
    print(f"  {typ:18s}: {r_old:5.1f}% -> {r_new:5.1f}% (Delta: {r_new - r_old:+5.1f}%)")

