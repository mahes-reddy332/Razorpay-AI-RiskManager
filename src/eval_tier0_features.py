"""
FAST DIAGNOSTIC: Test all 4 proposed Tier 0 replacement features on TRAIN split.
Check which ones actually separate mules from legitimate on the IBM dataset.
Do NOT touch test set.
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

df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False)

# =====================================================================
# FEATURE 1: Counterparty Asymmetry Index (CAI)
# CAI = distinct_targets / (distinct_sources + distinct_targets)
# Mule ≈ 0.5 (symmetric), Legit ≈ 0.01 or 0.99 (asymmetric hub)
# =====================================================================
print("\nComputing Feature 1: Counterparty Asymmetry Index (CAI)...")
out_counterparties = df.groupby('Account')['Account.1'].nunique().rename('distinct_targets')
in_counterparties = df.groupby('Account.1')['Account'].nunique().rename('distinct_sources')
cai_df = pd.DataFrame({'distinct_targets': out_counterparties, 'distinct_sources': in_counterparties})
cai_df = cai_df.fillna(0)
cai_df['CAI'] = cai_df['distinct_targets'] / (cai_df['distinct_sources'] + cai_df['distinct_targets'] + 1e-9)

# =====================================================================
# FEATURE 2: One-Shot Counterparty Ratio (outbound)
# What % of outbound counterparties were used exactly once?
# =====================================================================
print("Computing Feature 2: One-Shot Counterparty Ratio...")
txn_counts_per_cp = df.groupby(['Account', 'Account.1']).size().reset_index(name='txn_count')
one_shot = txn_counts_per_cp.groupby('Account').apply(
    lambda g: (g['txn_count'] == 1).sum() / len(g) if len(g) > 0 else 0
).rename('one_shot_ratio')

# =====================================================================
# FEATURE 3: Temporal Burst Concentration (top-3 hourly buckets)
# B = sum(top-3 hourly volumes) / total volume
# =====================================================================
print("Computing Feature 3: Temporal Burst Concentration...")
df['hour_bucket'] = df['Timestamp'].dt.floor('h')
hourly_vol = df.groupby(['Account', 'hour_bucket'])['Amount Paid'].sum().reset_index()

def top3_concentration(group):
    vals = group['Amount Paid'].values
    total = vals.sum()
    if total <= 0:
        return 0.0
    top3 = np.sort(vals)[-3:].sum()
    return top3 / total

burst_conc = hourly_vol.groupby('Account').apply(top3_concentration).rename('burst_concentration')

# =====================================================================
# FEATURE 4: Median Forwarding Lag (seconds)
# For each outgoing txn, find most recent preceding incoming txn, compute lag
# =====================================================================
print("Computing Feature 4: Median Forwarding Lag...")
# Simplified: per-account, compute median(out_timestamps) - median(in_timestamps)
# More precise: use merge_asof
df_out = df[['Account', 'Timestamp', 'Amount Paid']].copy()
df_out.columns = ['account', 'out_time', 'out_amount']
df_in = df[['Account.1', 'Timestamp', 'Amount Paid']].copy()
df_in.columns = ['account', 'in_time', 'in_amount']

# For speed, compute per-account: median time between first_in and first_out
# (approximation that's fast enough for 5M rows)
first_in = df_in.groupby('account')['in_time'].min()
first_out = df_out.groupby('account')['out_time'].min()
lag_df = pd.DataFrame({'first_in': first_in, 'first_out': first_out}).dropna()
lag_df['lag_seconds'] = (lag_df['first_out'] - lag_df['first_in']).dt.total_seconds()

# Also compute: average time between consecutive in/out events
out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_count=('Amount Paid', 'count'),
    out_unique_dest=('Account.1', 'nunique')
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    safe_in=('Amount Paid', 'sum')
)

# =====================================================================
# FEATURE 5: CV of Outbound Amounts
# =====================================================================
print("Computing Feature 5: CV of Outbound Amounts...")
out_cv = df.groupby('Account')['Amount Paid'].agg(['std', 'mean'])
out_cv['cv_out'] = out_cv['std'] / (out_cv['mean'] + 1e-9)
out_cv = out_cv['cv_out']

# =====================================================================
# FEATURE 6: Outbound Burstiness (CV of inter-transaction gaps)
# =====================================================================
print("Computing Feature 6: CV of Outbound Inter-Transaction Gaps...")
def cv_gaps(group):
    times = group['Timestamp'].sort_values()
    if len(times) < 3:
        return np.nan
    gaps = times.diff().dt.total_seconds().dropna()
    if gaps.mean() == 0:
        return 0.0
    return gaps.std() / (gaps.mean() + 1e-9)

gap_cv = df.groupby('Account').apply(cv_gaps).rename('gap_cv')

# =====================================================================
# Merge all features
# =====================================================================
print("\nMerging features...")
accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['is_mule'] = accounts_df.index.isin(pattern_nodes)
accounts_df = accounts_df.join(cai_df['CAI'])
accounts_df = accounts_df.join(one_shot)
accounts_df = accounts_df.join(burst_conc)
accounts_df = accounts_df.join(lag_df['lag_seconds'])
accounts_df = accounts_df.join(out_cv)
accounts_df = accounts_df.join(gap_cv)

# Train/test split (same seed)
all_idx = accounts_df.index.values
y = accounts_df['is_mule'].values
X_train_val, X_test, _, _ = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
train_df = accounts_df.loc[X_train_val]

mules = train_df[train_df['is_mule']]
legit = train_df[~train_df['is_mule']]

# =====================================================================
# REPORT: Distribution comparison for each feature
# =====================================================================
print("\n" + "=" * 80)
print("FEATURE DISCRIMINATION REPORT (TRAIN SPLIT ONLY)")
print(f"Train: {len(train_df):,} accounts ({len(mules):,} mules, {len(legit):,} legit)")
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
    
    print(f"  Mule:  n={len(m_vals):,}  mean={m_vals.mean():.4f}  median={m_vals.median():.4f}  "
          f"p25={m_vals.quantile(0.25):.4f}  p75={m_vals.quantile(0.75):.4f}  "
          f"p95={m_vals.quantile(0.95):.4f}")
    print(f"  Legit: n={len(l_vals):,}  mean={l_vals.mean():.4f}  median={l_vals.median():.4f}  "
          f"p25={l_vals.quantile(0.25):.4f}  p75={l_vals.quantile(0.75):.4f}  "
          f"p95={l_vals.quantile(0.95):.4f}")
    
    # Separation metric: difference of medians / pooled IQR
    iqr_m = m_vals.quantile(0.75) - m_vals.quantile(0.25)
    iqr_l = l_vals.quantile(0.75) - l_vals.quantile(0.25)
    pooled_iqr = (iqr_m + iqr_l) / 2 + 1e-9
    separation = abs(m_vals.median() - l_vals.median()) / pooled_iqr
    print(f"  ** Separation Score (|median_diff| / avg_IQR): {separation:.4f} **")
    
    if separation > 0.5:
        print(f"  ==> PROMISING: Good separation between mules and legit")
    elif separation > 0.2:
        print(f"  ==> MODERATE: Some separation, may help in combination")
    else:
        print(f"  ==> WEAK: Little separation on this dataset")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
