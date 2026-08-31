import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== 1. VERIFYING FULL DATASET DIMENSIONS ===")
# Parse patterns
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            if len(parts) >= 5:
                pattern_nodes_all.add(parts[2])
                pattern_nodes_all.add(parts[4])

print("Reading full 5M transaction CSV...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
num_txns = len(df)
print(f"Total Transactions in Dataset: {num_txns:,}")

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

accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

total_accounts = len(accounts_df)
print(f"Total Eligible Accounts in Dataset (in + out edges): {total_accounts:,}")
print(f"Total Ground Truth Mules: {accounts_df['is_true_mule'].sum():,}")

# Build Graph
print("Building Graph...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

# Train/Val/Test Split (60/20/20)
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
test_df = accounts_df.loc[X_test].copy()
test_mules = test_df['is_true_mule'].sum()
test_legit = len(test_df) - test_mules

print(f"\nTest Set (20% Split): {len(test_df):,} accounts ({test_mules} True Mules, {test_legit:,} Legitimate Accounts)")

# Compute 15-hop BFS for test set
print("Computing 15-hop BFS reachability for Test Set accounts...")
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

# Operational Action Rules:
# Rule 1: RISK_CONTAINMENT_REQUIRED (HIGH_RISK) -> Full Corroboration: Tier 0 (pass_through > 0.90) AND Tier 1 (deep_deg >= 4) AND safe_ratio <= 0.50
test_df['auto_freeze'] = (
    (test_df['pass_through_ratio'] > 0.90) &
    ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) &
    ~(test_df['safe_ratio'] > 0.50)
)

# Rule 2: MANUAL_REVIEW_REQUIRED -> Extreme Topology ALONE: (deep_deg >= 15) AND safe_ratio <= 0.50 AND NOT auto_freeze
test_df['extreme_topology'] = (
    ((test_df['deep_out'] >= 15) | (test_df['deep_in'] >= 15)) &
    ~(test_df['safe_ratio'] > 0.50)
)
test_df['manual_review'] = test_df['extreme_topology'] & ~test_df['auto_freeze']

# Rule 3: SAFE -> Neither
test_df['safe'] = ~test_df['auto_freeze'] & ~test_df['manual_review']

# Breakdown of Mules
mules_total = test_df['is_true_mule'].sum()
mules_frozen = (test_df['auto_freeze'] & test_df['is_true_mule']).sum()
mules_reviewed = (test_df['manual_review'] & test_df['is_true_mule']).sum()
mules_missed = (test_df['safe'] & test_df['is_true_mule']).sum()
total_caught = mules_frozen + mules_reviewed

# Breakdown of Legitimate Accounts
legit_total = (~test_df['is_true_mule']).sum()
legit_frozen = (test_df['auto_freeze'] & ~test_df['is_true_mule']).sum()
legit_reviewed = (test_df['manual_review'] & ~test_df['is_true_mule']).sum()
legit_safe = (test_df['safe'] & ~test_df['is_true_mule']).sum()

print("\n==========================================================================")
print("=== FINAL OPERATIONAL TIERS: ACTIONABLE BREAKDOWN ON IBM TEST SPLIT ===")
print("==========================================================================")

print(f"\n1. GROUND TRUTH MULES ({mules_total} Total):")
print(f"   * CONTAINED_FOR_RISK (HIGH_RISK - Tier 0 + Tier 1 Corroborated):      {mules_frozen:3d} ({mules_frozen/mules_total*100:5.2f}%)")
print(f"   * ROUTED TO MANUAL REVIEW (Extreme Topology D >= 15 Only):     {mules_reviewed:3d} ({mules_reviewed/mules_total*100:5.2f}%)")
print(f"   -----------------------------------------------------------------------")
print(f"   * TOTAL DETECTED (Combined System Pipeline Catch Rate):        {total_caught:3d} ({total_caught/mules_total*100:5.2f}%)")
print(f"   * FALSE NEGATIVES (Missed / Silent Low-Volume):                {mules_missed:3d} ({mules_missed/mules_total*100:5.2f}%)")

print(f"\n2. LEGITIMATE ACCOUNTS ({legit_total:,} Total):")
print(f"   * CONTAINED_FOR_RISK FALSE POSITIVES (Direct Freeze Error):          {legit_frozen:6,d} ({legit_frozen/legit_total*100:5.2f}%)")
print(f"   * ROUTED TO ASYNCHRONOUS REVIEW QUEUE (High-Degree Clearing): {legit_reviewed:6,d} ({legit_reviewed/legit_total*100:5.2f}%)")
print(f"   * AUTO-CLEARED AS SAFE (Zero Disruption):                    {legit_safe:6,d} ({legit_safe/legit_total*100:5.2f}%)")

print("\n==========================================================================")
print(f"KEY TAKEAWAY: 0% of the {legit_reviewed:,} new high-degree false positives are contained for risk.")
print("They land exclusively in the review queue, preserving legitimate business operations.")
print("==========================================================================")
