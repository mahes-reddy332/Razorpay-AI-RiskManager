import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_v2 import extract_features, FROZEN_CONFIG

print("=== PART B: GRADUAL ESCALATION (TREND-SLOPE ON EXTENDED WINDOW) ===")

# 1. Load Data
txns = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
accs = pd.read_csv('data/accounts.csv').set_index('account_id')

min_date = txns['timestamp'].min()
max_date = txns['timestamp'].max()
total_days = (max_date - min_date).days
print(f"Transaction date range: {min_date.date()} to {max_date.date()} ({total_days} days)")

# Inject Hard Negative: Genuine organic business growth
# E.g. Freelancer/SMB gaining clients each week steadily (e.g. week 1: 500, week 2: 1000, week 3: 1500, week 4: 2000)
hn_acc = 'ACC_HARD_NEG_ORGANIC_GROWTH'
accs.loc[hn_acc] = {'creation_date': min_date, 'device_id': 'DEV_HN_SMB', 'activity_level': 'high', 'kyc_tier': 3, 'mcc_code': None, 'is_mule': False, 'is_adversarial': False, 'behavior_type': 'normal'}

hn_txns = []
for w in range(4): # 4-week synthetic window
    weekly_base = (w + 1) * 600.0 # 600, 1200, 1800, 2400 (steady linear ramp)
    for i in range(4):
        hn_txns.append({
            'transaction_id': f'TXN_HN_GROWTH_{w}_{i}',
            'source': hn_acc,
            'target': f'EXT_CLIENT_{w}_{i}',
            'amount': weekly_base / 4.0,
            'timestamp': min_date + pd.Timedelta(days=w*7 + i),
            'is_laundering': False
        })

# Also inject an adversarial Gradual Escalation Mule (patiently escalating amounts)
mule_adv_acc = 'ACC_MULE_GRADUAL_ESCALATION'
accs.loc[mule_adv_acc] = {'creation_date': min_date, 'device_id': 'DEV_MULE_ESC', 'activity_level': 'medium', 'kyc_tier': 2, 'mcc_code': None, 'is_mule': True, 'is_adversarial': True, 'behavior_type': 'mule_hop'}

mule_txns = []
for w in range(4):
    weekly_amt = (2 ** w) * 500.0 # 500, 1000, 2000, 4000 (exponential acceleration)
    in_time = min_date + pd.Timedelta(days=w*7 + 1)
    out_time = min_date + pd.Timedelta(days=w*7 + 2)
    # Inflow
    mule_txns.append({
        'transaction_id': f'TXN_MULE_ESC_IN_{w}',
        'source': 'EXT_VICTIM',
        'target': mule_adv_acc,
        'amount': weekly_amt,
        'timestamp': in_time,
        'is_laundering': True
    })
    # Outflow
    mule_txns.append({
        'transaction_id': f'TXN_MULE_ESC_OUT_{w}',
        'source': mule_adv_acc,
        'target': 'EXT_CASHOUT',
        'amount': weekly_amt * 0.98,
        'timestamp': out_time,
        'is_laundering': True
    })

txns = pd.concat([txns, pd.DataFrame(hn_txns), pd.DataFrame(mule_txns)], ignore_index=True)

# 2. Compute Weekly Volume and Trend-Slope (Linear Regression Slope across weekly volumes)
txns['week'] = ((txns['timestamp'] - min_date).dt.days // 7).clip(0, 3)
weekly_vol = txns.groupby(['source', 'week'])['amount'].sum().unstack(fill_value=0.0)

# Ensure 4 weeks exist
for w in range(4):
    if w not in weekly_vol.columns:
        weekly_vol[w] = 0.0

weekly_vol = weekly_vol[[0, 1, 2, 3]]
weeks_arr = np.array([0, 1, 2, 3])

# Slope = Cov(weeks, vols) / Var(weeks)
def calc_slope(row):
    v = row.values
    if np.sum(v) < 100: return 0.0 # Ignore inactive
    # Normalize by mean to get percentage growth slope per week
    mean_v = np.mean(v)
    if mean_v <= 0: return 0.0
    slope = np.polyfit(weeks_arr, v, 1)[0]
    return float(slope / mean_v) # Normalized growth rate per week

weekly_vol['trend_slope'] = weekly_vol.apply(calc_slope, axis=1)
weekly_vol = weekly_vol.reindex(accs.index).fillna({'trend_slope': 0.0})

# 3. Base Features
df_features = extract_features()
# Add missing injected rows to df_features if needed
for extra_acc in [hn_acc, mule_adv_acc]:
    if extra_acc not in df_features.index:
        df_features.loc[extra_acc] = 0.0

df = df_features.join(weekly_vol[['trend_slope']]).fillna({'trend_slope': 0.0})
df['is_mule'] = accs.loc[df.index, 'is_mule'].astype(int)

# 4. Train/Val/Test Split
y = df['is_mule'].values
all_accounts = df.index.values

# Exclude synthetic benchmark test nodes from tuning
tune_mask = ~np.isin(all_accounts, [hn_acc, mule_adv_acc])
X_train_val, X_test, y_train_val, y_test = train_test_split(all_accounts[tune_mask], y[tune_mask], test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

val_df = df.loc[X_val].copy()

# 5. Tune Trend-Slope on Validation Split Only
best_f1 = 0
best_slope_thresh = 0.5
best_weight = 0.0

print("\nTuning 'Trend Slope >= S' on Validation Split...")
for s in [0.3, 0.5, 0.8, 1.0]: # 30%, 50%, 80%, 100% weekly growth slope
    for weight in [0.0, 0.4, 0.8, 1.2]:
        scores = val_df['risk_score'].values.copy()
        
        nc = val_df['node_count'].values
        scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
        scores += (val_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
        scores += (val_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
        scores += (val_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
        scores += (val_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
        scores += (val_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']
        
        # Trend Slope Signal
        scores += (val_df['trend_slope'] >= s).values * weight
        
        y_pred = scores >= FROZEN_CONFIG['dec_thresh']
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_slope_thresh = s
            best_weight = weight

print(f"Optimal Validation Parameters:")
print(f"  Trend Slope Threshold: {best_slope_thresh}")
print(f"  Trend Slope Weight:    {best_weight}")

# 6. Evaluate on Test Split
test_df = df.loc[X_test].copy()
scores = test_df['risk_score'].values.copy()
nc = test_df['node_count'].values
scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
scores += (test_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
scores += (test_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
scores += (test_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
scores += (test_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
scores += (test_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']

# Baseline
y_pred_base = scores >= FROZEN_CONFIG['dec_thresh']
r_base = recall_score(y_test, y_pred_base)
p_base = precision_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)

# With Slope
scores += (test_df['trend_slope'] >= best_slope_thresh).values * best_weight
y_pred_new = scores >= FROZEN_CONFIG['dec_thresh']
r_new = recall_score(y_test, y_pred_new)
p_new = precision_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)

print(f"\nTEST SET RESULTS:")
print(f"  Baseline (Test Set) -> Precision: {p_base*100:.2f}%, Recall: {r_base*100:.2f}%, F1: {f1_base:.4f}")
print(f"  With Trend-Slope    -> Precision: {p_new*100:.2f}%, Recall: {r_new*100:.2f}%, F1: {f1_new:.4f}")

# 7. Check Hard Negative (Organic SMB Growth) vs Adversarial Mule
print(f"\n--- HARD NEGATIVE & ADVERSARIAL CASE AUDIT ---")
hn_slope = weekly_vol.loc[hn_acc, 'trend_slope']
mule_slope = weekly_vol.loc[mule_adv_acc, 'trend_slope']
print(f"1. Legitimate Organic SMB ({hn_acc}):")
print(f"   Weekly Volumes: {weekly_vol.loc[hn_acc, [0,1,2,3]].values}")
print(f"   Calculated Trend Slope: {hn_slope:.2f}")
print(f"   Fires Slope Rule (>= {best_slope_thresh})? {'YES (False Positive Risk)' if hn_slope >= best_slope_thresh else 'NO'}")

print(f"\n2. Adversarial Gradual Mule ({mule_adv_acc}):")
print(f"   Weekly Volumes: {weekly_vol.loc[mule_adv_acc, [0,1,2,3]].values}")
print(f"   Calculated Trend Slope: {mule_slope:.2f}")
print(f"   Fires Slope Rule (>= {best_slope_thresh})? {'YES (Caught)' if mule_slope >= best_slope_thresh else 'NO'}")
