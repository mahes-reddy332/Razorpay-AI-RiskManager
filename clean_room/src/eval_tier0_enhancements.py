import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== CHECK 1: DORMANCY ON IBM DATASET ===")
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

# Load IBM dataset with Timestamp
print("Loading IBM transactions with timestamps...")
df = pd.read_csv(csv_path, usecols=['Timestamp', 'Account', 'Account.1', 'Amount Paid', 'Payment Format'])
print("Columns and head:")
print(df[['Timestamp', 'Account', 'Account.1', 'Amount Paid']].head())

# Check Timestamp format
# IBM dataset Timestamp is typically formatted as 'YYYY/MM/DD hh:mm'
df['dt'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')
min_time = df['dt'].min()
max_time = df['dt'].max()
print(f"Time span of IBM dataset: {min_time} to {max_time} (Total days: {(max_time - min_time).days})")

# Check dormancy on IBM data
# For each account, find its first inbound and outbound txn
in_first = df.groupby('Account.1')['dt'].min()
out_first = df.groupby('Account')['dt'].min()
in_last = df.groupby('Account.1')['dt'].max()
out_last = df.groupby('Account')['dt'].max()

# Look at dormant gap: max gap between consecutive txns per account, or initial inactivity
print("\n=== CHECK 2: DESTINATION CONCENTRATION (TIER 0 FEATURE) ===")
# Destination Concentration = distinct outbound counterparties / total outbound transaction count
# Low ratio (e.g. 1 / 10 = 0.1) -> repeated funnels to the same 1-2 destinations (mule-like)
# High ratio (e.g. 10 / 10 = 1.0) -> unique counterparties for every txn (SMB / Salary / Normal)

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_tx_count=('Amount Paid', 'count'),
    out_unique_dest=('Account.1', 'nunique')
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    in_tx_count=('Amount Paid', 'count'),
    in_unique_src=('Account', 'nunique')
)

df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']
safe_in_stats = df.groupby('Account.1')['safe_amount'].sum()

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = safe_in_stats.reindex(accounts_df.index).fillna(0.0)

accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Compute Destination Concentration Ratio: unique_dest / total_out_tx
# For accounts with out_tx_count >= 2:
accounts_df['dest_concentration_ratio'] = accounts_df['out_unique_dest'] / np.maximum(accounts_df['out_tx_count'], 1)

print("\nDestination Concentration Ratio Distribution:")
print("Mules mean dest_concentration:", accounts_df[accounts_df['is_true_mule']]['dest_concentration_ratio'].mean())
print("Legit mean dest_concentration:", accounts_df[~accounts_df['is_true_mule']]['dest_concentration_ratio'].mean())

# Split 60/20/20
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

val_df = accounts_df.loc[X_val].copy()
test_df = accounts_df.loc[X_test].copy()

# Base Tier 0 filter on Validation
val_t0_base = val_df['pass_through_ratio'] > 0.90
print(f"\nValidation Base Tier 0 Mules Passed: {(val_t0_base & val_df['is_true_mule']).sum()} / {val_df['is_true_mule'].sum()} ({(val_t0_base & val_df['is_true_mule']).sum() / val_df['is_true_mule'].sum() * 100:.2f}%)")

# Tune destination concentration threshold on Validation Split:
# Idea: If an account has low pass-through (e.g. 50-90%), BUT has extreme destination concentration (funnels repeatedly to 1-2 destinations, ratio <= C), promote it past Tier 0!
best_c = None
best_val_recall = (val_t0_base & val_df['is_true_mule']).sum()
best_promoted_f1 = 0

for c_thresh in [0.2, 0.3, 0.5, 0.7]:
    # Expanded Tier 0: (pass_through > 0.90) OR (pass_through > 0.50 AND dest_concentration <= c_thresh AND out_tx_count >= 3)
    val_t0_new = val_t0_base | ((val_df['pass_through_ratio'] > 0.50) & (val_df['dest_concentration_ratio'] <= c_thresh) & (val_df['out_tx_count'] >= 3))
    mules_passed = (val_t0_new & val_df['is_true_mule']).sum()
    legit_passed = (val_t0_new & ~val_df['is_true_mule']).sum()
    
    print(f"  Tune C={c_thresh:.2f} -> Mules Passed: {mules_passed}/{val_df['is_true_mule'].sum()} ({mules_passed/val_df['is_true_mule'].sum()*100:.2f}%), Legit Passed to Tier 1: {legit_passed:,}")
    if mules_passed > best_val_recall:
        best_val_recall = mules_passed
        best_c = c_thresh

print(f"\nOptimal Destination Concentration Threshold: C={best_c}")

# Evaluate on Test Split
test_t0_base = test_df['pass_through_ratio'] > 0.90
mules_t0_base = (test_t0_base & test_df['is_true_mule']).sum()

if best_c is not None:
    test_t0_new = test_t0_base | ((test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= best_c) & (test_df['out_tx_count'] >= 3))
else:
    test_t0_new = test_t0_base

mules_t0_new = (test_t0_new & test_df['is_true_mule']).sum()
total_test_mules = test_df['is_true_mule'].sum()

print("\n=== TEST SPLIT RESULTS ===")
print(f"Baseline Tier 0 Pass Rate: {mules_t0_base} / {total_test_mules} ({mules_t0_base/total_test_mules*100:.2f}%)")
print(f"With Destination Concentration: {mules_t0_new} / {total_test_mules} ({mules_t0_new/total_test_mules*100:.2f}%)")
print(f"Additional Mules Promoted to Tier 1: +{mules_t0_new - mules_t0_base} ({+(mules_t0_new - mules_t0_base)/total_test_mules*100:+.2f}%)")
