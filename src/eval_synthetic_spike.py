import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_v2 import extract_features, FROZEN_CONFIG

print("=== PART 3: SELF-RELATIVE SPIKE RATIO (Synthetic Data) ===")

# 1. Load Data
txns = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
accs = pd.read_csv('data/accounts.csv').set_index('account_id')

# Inject Hard Negative: Active account (weeks 1-3) with a massive legitimate purchase (week 4)
start_date = txns['timestamp'].min()
week4_start = start_date + pd.Timedelta(days=21)

hn_acc = 'ACC_HARD_NEGATIVE_BONUS'
accs.loc[hn_acc] = {'creation_date': start_date, 'device_id': 'DEV_HN', 'activity_level': 'high', 'kyc_tier': 3, 'mcc_code': None, 'is_mule': False, 'is_adversarial': False, 'behavior_type': 'normal'}

hn_txns = []
# 3 weeks of normal spending (~1000 per week)
for w in range(3):
    for _ in range(5):
        hn_txns.append({'transaction_id': f'TXN_HN_{w}_{_}', 'source': hn_acc, 'target': 'EXT_MERCHANT', 'amount': 200.0, 'timestamp': start_date + pd.Timedelta(days=w*7 + _), 'is_laundering': False})
# 1 massive purchase in week 4 (e.g. buying a car for 20000 -> 20x spike)
hn_txns.append({'transaction_id': 'TXN_HN_CAR', 'source': hn_acc, 'target': 'EXT_DEALERSHIP', 'amount': 20000.0, 'timestamp': week4_start + pd.Timedelta(days=2), 'is_laundering': False})

txns = pd.concat([txns, pd.DataFrame(hn_txns)], ignore_index=True)

# 2. Calculate Weekly Volumes
hist_txns = txns[txns['timestamp'] < week4_start]
curr_txns = txns[txns['timestamp'] >= week4_start]

hist_out = hist_txns.groupby('source')['amount'].sum() / 3.0 # Weekly average over 3 weeks
curr_out = curr_txns.groupby('source')['amount'].sum()       # Volume in week 4

spike_df = pd.DataFrame({'hist_avg': hist_out, 'curr_vol': curr_out}).fillna(0.0)
spike_df = spike_df.reindex(accs.index).fillna({'hist_avg': 0.0, 'curr_vol': 0.0})
spike_df['spike_ratio'] = spike_df['curr_vol'] / np.maximum(spike_df['hist_avg'], 1.0)

# 3. Merge with base features
df_features = extract_features()
# manually append the HN to df_features with default 0s so it doesn't break
hn_feats = pd.DataFrame(index=[hn_acc], columns=df_features.columns).fillna(0.0)
df_features = pd.concat([df_features, hn_feats])

df = df_features.join(accs[['behavior_type']])
df = df.join(spike_df[['spike_ratio']]).fillna({'spike_ratio': 0.0})

# 4. Tune on Validation Split
y = df['is_mule'].astype(int).values
all_accounts = df.index.values

# Exclude HN from tuning, it's just for the final check
tune_mask = all_accounts != hn_acc
X_train_val, X_test, y_train_val, y_test = train_test_split(all_accounts[tune_mask], y[tune_mask], test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

val_df = df.loc[X_val].copy()

best_f1 = 0
best_n = 15
best_weight = 0.0

print("\nTuning 'Spike Ratio >= N' rule on Validation Split...")
for n in [10, 15, 20, 25]:
    for weight in [0.0, 0.5, 1.0, 1.5]:
        scores = val_df['risk_score'].values.copy()
        
        nc = val_df['node_count'].values
        scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
        scores += (val_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
        scores += (val_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
        scores += (val_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
        scores += (val_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
        scores += (val_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']
        
        # SPIKE FEATURE
        # Only apply if it's an actively used account (hist_avg > 500) to avoid dormancy false positives
        active_mask = (val_df['hist_avg'] > 500.0).values if 'hist_avg' in val_df.columns else np.ones(len(val_df), dtype=bool)
        if 'hist_avg' not in val_df.columns:
            active_mask = (df.loc[X_val, 'spike_ratio'] > 0).values # fallback
            
        # Re-fetch hist_avg correctly
        hist_avg_val = df.loc[X_val, 'spike_ratio'] * 0 # placeholder
        if 'hist_avg' in spike_df.columns:
            hist_avg_val = spike_df.loc[X_val, 'hist_avg'].fillna(0).values
            
        scores += ((val_df['spike_ratio'] >= n) & (hist_avg_val > 500.0)).values * weight
        
        y_pred = scores >= FROZEN_CONFIG['dec_thresh']
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_n = n
            best_weight = weight

print(f"Optimal Validation Parameters:")
print(f"  Spike Multiplier (N): {best_n}x")
print(f"  Spike Weight: {best_weight}")

# 5. Evaluate exactly ONCE on Test Split
print("\nEvaluating on untouched Test Split...")
test_df = df.loc[X_test].copy()
scores = test_df['risk_score'].values.copy()
nc = test_df['node_count'].values
scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
scores += (test_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
scores += (test_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
scores += (test_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
scores += (test_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
scores += (test_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']

y_pred_base = scores >= FROZEN_CONFIG['dec_thresh']
r_base = recall_score(y_test, y_pred_base)
p_base = precision_score(y_test, y_pred_base)

hist_avg_test = spike_df.loc[X_test, 'hist_avg'].fillna(0).values
scores += ((test_df['spike_ratio'] >= best_n) & (hist_avg_test > 500.0)).values * best_weight
y_pred_new = scores >= FROZEN_CONFIG['dec_thresh']
r_new = recall_score(y_test, y_pred_new)
p_new = precision_score(y_test, y_pred_new)

print(f"BASELINE METRICS (Test Set):")
print(f"  Precision: {p_base*100:.2f}%, Recall: {r_base*100:.2f}%")

print(f"\nNEW SPIKE-LINKED METRICS (Test Set):")
print(f"  Precision: {p_new*100:.2f}%, Recall: {r_new*100:.2f}%")

# 6. Hard Negative Check
print(f"\n--- LEGITIMATE HARD NEGATIVE CHECK (Car Purchase) ---")
hn_row = df.loc[hn_acc]
hn_hist = spike_df.loc[hn_acc, 'hist_avg']
hn_curr = spike_df.loc[hn_acc, 'curr_vol']
hn_spike = hn_row['spike_ratio']
print(f"Account {hn_acc}:")
print(f"  Historical Weekly Avg: {hn_hist}")
print(f"  Current Weekly Vol: {hn_curr}")
print(f"  Spike Ratio: {hn_spike:.1f}x")
hn_score = (hn_spike >= best_n) and (hn_hist > 500.0)
print(f"  Would Spike Rule Fire? {'YES (FAIL)' if hn_score else 'NO (PASS)'}")
