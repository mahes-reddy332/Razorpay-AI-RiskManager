"""
DIAGNOSTIC: Test 6 proposed Tier 0 features on TRAIN split, FILTERED to accounts with >= 8 total transactions.
This removes degenerate cases (accounts with 1-2 transactions) to see if true separation exists among active accounts.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("Loading data...")
pattern_nodes = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            if len(parts) >= 5:
                pattern_nodes.add(parts[2])
                pattern_nodes.add(parts[4])

df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False)

# Transaction counts
out_stats = df.groupby('Account').agg(out_tx_count=('Amount Paid', 'count'))
in_stats = df.groupby('Account.1').agg(in_tx_count=('Amount Paid', 'count'))

all_accounts = list(set(df['Account']).union(set(df['Account.1'])))
accounts_df = pd.DataFrame(index=all_accounts)
accounts_df = accounts_df.join(out_stats).join(in_stats)
accounts_df['out_tx_count'] = accounts_df['out_tx_count'].fillna(0)
accounts_df['in_tx_count'] = accounts_df['in_tx_count'].fillna(0)
accounts_df['total_txns'] = accounts_df['out_tx_count'] + accounts_df['in_tx_count']
accounts_df['is_mule'] = accounts_df.index.isin(pattern_nodes)

print("Computing features...")
# 1. CAI
out_cp = df.groupby('Account')['Account.1'].nunique().rename('distinct_targets')
in_cp = df.groupby('Account.1')['Account'].nunique().rename('distinct_sources')
cai_df = pd.DataFrame({'distinct_targets': out_cp, 'distinct_sources': in_cp}).fillna(0)
cai_df['CAI'] = cai_df['distinct_targets'] / (cai_df['distinct_sources'] + cai_df['distinct_targets'] + 1e-9)

# 2. One-Shot
txn_counts_per_cp = df.groupby(['Account', 'Account.1']).size().reset_index(name='txn_count')
one_shot = txn_counts_per_cp.groupby('Account').apply(
    lambda g: (g['txn_count'] == 1).sum() / len(g) if len(g) > 0 else 0
).rename('one_shot_ratio')

# 3. Burst Concentration
df['hour_bucket'] = df['Timestamp'].dt.floor('h')
hourly_vol = df.groupby(['Account', 'hour_bucket'])['Amount Paid'].sum().reset_index()
def top3_conc(g):
    vals = g['Amount Paid'].values
    return np.sort(vals)[-3:].sum() / vals.sum() if vals.sum() > 0 else 0.0
burst_conc = hourly_vol.groupby('Account').apply(top3_conc).rename('burst_concentration')

# 4. Lag
df_out = df[['Account', 'Timestamp']].copy(); df_out.columns = ['account', 'out_time']
df_in = df[['Account.1', 'Timestamp']].copy(); df_in.columns = ['account', 'in_time']
first_in = df_in.groupby('account')['in_time'].min()
first_out = df_out.groupby('account')['out_time'].min()
lag_df = pd.DataFrame({'first_in': first_in, 'first_out': first_out}).dropna()
lag_df['lag_seconds'] = (lag_df['first_out'] - lag_df['first_in']).dt.total_seconds()

# 5. CV Outbound
out_cv = df.groupby('Account')['Amount Paid'].agg(['std', 'mean'])
out_cv['cv_out'] = out_cv['std'] / (out_cv['mean'] + 1e-9)

# 6. CV Gaps
def cv_gaps(g):
    t = g['Timestamp'].sort_values()
    if len(t) < 3: return np.nan
    diffs = t.diff().dt.total_seconds().dropna()
    return diffs.std() / (diffs.mean() + 1e-9) if diffs.mean() > 0 else 0.0
gap_cv = df.groupby('Account').apply(cv_gaps).rename('gap_cv')

# Join features
accounts_df = accounts_df.join(cai_df['CAI'])
accounts_df = accounts_df.join(one_shot)
accounts_df = accounts_df.join(burst_conc)
accounts_df = accounts_df.join(lag_df['lag_seconds'])
accounts_df = accounts_df.join(out_cv['cv_out'])
accounts_df = accounts_df.join(gap_cv)

# Train/Test Split
all_idx = accounts_df.index.values
y = accounts_df['is_mule'].values
X_train_val, _, _, _ = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)

train_df = accounts_df.loc[X_train_val]

# FILTER TO >= 8 TRANSACTIONS
active_df = train_df[train_df['total_txns'] >= 8]

mules = active_df[active_df['is_mule']]
legit = active_df[~active_df['is_mule']]

print("\n" + "=" * 80)
print("FEATURE DISCRIMINATION REPORT (TRAIN SPLIT, >= 8 TXNS ONLY)")
print(f"Original Train Set: {len(train_df):,} accounts")
print(f"Filtered (>=8 txns): {len(active_df):,} accounts")
print(f"  Mules passing filter: {len(mules):,} / {train_df['is_mule'].sum():,} ({(len(mules)/train_df['is_mule'].sum())*100:.1f}%)")
print(f"  Legit passing filter: {len(legit):,} / {(~train_df['is_mule']).sum():,} ({(len(legit)/(~train_df['is_mule']).sum())*100:.1f}%)")
print("=" * 80)

features = {
    'CAI': 'CAI',
    'One-Shot Ratio': 'one_shot_ratio',
    'Burst Concentration': 'burst_concentration',
    'Forwarding Lag (s)': 'lag_seconds',
    'CV of Outbound Amounts': 'cv_out',
    'CV of Inter-Txn Gaps': 'gap_cv',
}

for name, col in features.items():
    print(f"\n--- {name} ({col}) ---")
    m_vals = mules[col].dropna()
    l_vals = legit[col].dropna()
    if len(m_vals) == 0 or len(l_vals) == 0:
        print("  INSUFFICIENT DATA")
        continue
        
    print(f"  Mule:  n={len(m_vals):,}  median={m_vals.median():.4f}  (mean={m_vals.mean():.2f})  p95={m_vals.quantile(0.95):.4f}")
    print(f"  Legit: n={len(l_vals):,}  median={l_vals.median():.4f}  (mean={l_vals.mean():.2f})  p95={l_vals.quantile(0.95):.4f}")
    
    iqr_m = m_vals.quantile(0.75) - m_vals.quantile(0.25)
    iqr_l = l_vals.quantile(0.75) - l_vals.quantile(0.25)
    pooled_iqr = (iqr_m + iqr_l) / 2 + 1e-9
    separation = abs(m_vals.median() - l_vals.median()) / pooled_iqr
    print(f"  ** Separation Score (|median_diff| / avg_IQR): {separation:.4f} **")

print("\n" + "=" * 80)
print("FILTERED DIAGNOSTIC COMPLETE")
print("=" * 80)
