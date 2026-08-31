import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

# Paths
csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("Loading data...")
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            if len(parts) >= 5:
                pattern_nodes_all.add(parts[2])
                pattern_nodes_all.add(parts[4])

df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(tot_out=('Amount Paid', 'sum'), out_tx_count=('Amount Paid', 'count'), out_unique_dest=('Account.1', 'nunique'))
in_stats = df.groupby('Account.1').agg(tot_in=('Amount Paid', 'sum'), safe_in=('safe_amount', 'sum'))

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = accounts_df['safe_in'].fillna(0.0)

# Calculate both CAPPED and RAW pass-through
accounts_df['pass_through_raw'] = accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['pass_through_raw'], 1.0)

accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['dest_concentration_ratio'] = accounts_df['out_unique_dest'] / np.maximum(accounts_df['out_tx_count'], 1)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Train/Test Split
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values
X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)

train_df = accounts_df.loc[X_train_val]
train_legit = train_df[~train_df['is_true_mule']]

# 1. Compute Percentiles on TRAIN legit accounts
p95_capped = np.percentile(train_legit['pass_through_ratio'], 95)
p97_capped = np.percentile(train_legit['pass_through_ratio'], 97)
p95_raw = np.percentile(train_legit['pass_through_raw'], 95)
p97_raw = np.percentile(train_legit['pass_through_raw'], 97)

print(f"--- TRAIN SPLIT PERCENTILES (Legit Accounts Only) ---")
print(f"P95 of CAPPED pass-through: {p95_capped:.4f}")
print(f"P97 of CAPPED pass-through: {p97_capped:.4f}")
print(f"P95 of RAW pass-through:    {p95_raw:.4f}")
print(f"P97 of RAW pass-through:    {p97_raw:.4f}")

# Using RAW P95 as the threshold to guarantee only 5% of legit train accounts pass
threshold = p95_raw

print("\nBuilding Graph & Computing 15-Hop Traversal on Test Split...")
test_df = accounts_df.loc[X_test].copy()
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

deep_out_counts = {}
deep_in_counts = {}
for i, node in enumerate(test_df.index):
    if not G.has_node(node): continue
    # BFS Down
    v_down = set(); q = [(node, 0)]
    while q:
        if len(v_down) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for succ in G.successors(c):
            if succ not in v_down: v_down.add(succ); q.append((succ, d + 1))
    # BFS Up
    v_up = set(); q = [(node, 0)]
    while q:
        if len(v_up) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for pred in G.predecessors(c):
            if pred not in v_up: v_up.add(pred); q.append((pred, d + 1))
    deep_out_counts[node] = len(v_down)
    deep_in_counts[node] = len(v_up)

test_df['deep_out'] = test_df.index.map(lambda x: deep_out_counts.get(x, 0))
test_df['deep_in'] = test_df.index.map(lambda x: deep_in_counts.get(x, 0))

# Evaluate Tier 0
tier0_b = (test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= 0.70) & (test_df['out_tx_count'] >= 3)
# New Rule A: population-relative percentile on raw pass-through
tier0_a = (test_df['pass_through_raw'] > threshold)
test_df['tier0_gate'] = tier0_a | tier0_b

# Risk Containment (Tier 1)
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

# Metrics
y_true = test_df['is_true_mule']
y_pred_total = test_df['total_detected']
y_pred_freeze = test_df['auto_freeze']

test_mules = y_true.sum()
test_legit = len(y_true) - test_mules

fp_frz = (y_pred_freeze & ~y_true).sum()
tp_frz = (y_pred_freeze & y_true).sum()
p_freeze = tp_frz / max(tp_frz + fp_frz, 1)
r_freeze = tp_frz / test_mules
f1_freeze = 2 * (p_freeze * r_freeze) / max(p_freeze + r_freeze, 1e-9)

fp_tot = (y_pred_total & ~y_true).sum()
tp_tot = (y_pred_total & y_true).sum()
p_total = tp_tot / max(tp_tot + fp_tot, 1)
r_total = tp_tot / test_mules
f1_total = 2 * (p_total * r_total) / max(p_total + r_total, 1e-9)

print("\n=== POPULATION-RELATIVE TIER 0 RESULTS (TEST SPLIT) ===")
print(f"Threshold applied: pass_through_raw > {threshold:.4f} (P95 of Train Legit)")

print("\n1. RISK_CONTAINMENT_REQUIRED ONLY:")
print(f"   Wrongful Risk Containments (FP): {fp_frz:,} (Compare against previous 17,657)")
print(f"   Mules Caught (TP):          {tp_frz:,} / {test_mules}")
print(f"   Precision:                  {p_freeze*100:.2f}%")
print(f"   Recall:                     {r_freeze*100:.2f}%")
print(f"   F1 Score:                   {f1_freeze:.4f}")

print("\n2. TOTAL PIPELINE (Risk Containment + L2 Review):")
print(f"   Total Mules Caught:         {tp_tot:,} / {test_mules} ({r_total*100:.2f}% Recall)")
print(f"   Total False Positives:      {fp_tot:,}")
print(f"   Total Precision:            {p_total*100:.2f}%")
print(f"   Total F1 Score:             {f1_total:.4f}")
