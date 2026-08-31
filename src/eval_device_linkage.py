import sys
import os
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_v2 import extract_features, FROZEN_CONFIG

print("=== PART 2: DEVICE/IP IDENTITY LINKAGE (Synthetic Data) ===")

# 1. Extract base features and load accounts
df_features = extract_features()
acc_df = pd.read_csv('data/accounts.csv').set_index('account_id')
df = df_features.join(acc_df[['device_id', 'behavior_type']])

# 2. Build Identity Graph / Device Linkage Feature
print("\nMapping Device Fingerprints...")
device_counts = df['device_id'].value_counts()
df['shared_device_count'] = df['device_id'].map(device_counts)

# Let's inspect the hard negatives (families sharing devices) vs mules (burner swarms)
family_accounts = df[df['behavior_type'] == 'normal']
burner_accounts = df[df['behavior_type'] == 'mule_hop'] # Swarm uses this, though behavior might be same

print("\n--- DEVICE DISTRIBUTION ---")
print("Legit accounts max shared devices:", family_accounts['shared_device_count'].max())
print("Mule accounts max shared devices:", df[df['is_mule']]['shared_device_count'].max())

# 3. Train / Val / Test Split
all_accounts = df.index.values
y = df['is_mule'].values
X_train_val, X_test, y_train_val, y_test = train_test_split(all_accounts, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) 

val_df = df.loc[X_val]
test_df = df.loc[X_test]

# 4. Tune on Validation Split
# We add a boolean flag: does this account share a device with >= N others?
# We will tune N in [2, 3, 4] and the weight of the signal.

best_f1 = 0
best_n = 3
best_weight = 0.0
best_params = FROZEN_CONFIG.copy()

print("\nTuning 'Shared Device >= N' rule on Validation Split...")
for n in [2, 3, 4, 5]:
    for weight in [0.0, 0.4, 0.8, 1.2]:
        scores = val_df['risk_score'].values.copy()
        
        # Apply standard features
        nc = val_df['node_count'].values
        scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
        scores += (val_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
        scores += (val_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
        scores += (val_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
        scores += (val_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
        scores += (val_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']
        
        # NEW IDENTITY LINKAGE FEATURE
        scores += (val_df['shared_device_count'] >= n).values * weight
        
        y_pred = scores >= FROZEN_CONFIG['dec_thresh']
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_n = n
            best_weight = weight

print(f"Optimal Validation Parameters:")
print(f"  Shared Device Threshold (N): {best_n}")
print(f"  Shared Device Weight: {best_weight}")

# 5. Evaluate exactly ONCE on Test Split
print("\nEvaluating on untouched Test Split...")
scores = test_df['risk_score'].values.copy()
nc = test_df['node_count'].values
scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
scores += (test_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
scores += (test_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
scores += (test_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
scores += (test_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
scores += (test_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * FROZEN_CONFIG['retention_weight']

# Baseline metrics (without device linkage)
y_pred_base = scores >= FROZEN_CONFIG['dec_thresh']
r_base = recall_score(test_df['is_mule'], y_pred_base)
p_base = precision_score(test_df['is_mule'], y_pred_base)
f1_base = f1_score(test_df['is_mule'], y_pred_base)

# With Device Linkage
scores += (test_df['shared_device_count'] >= best_n).values * best_weight
y_pred_new = scores >= FROZEN_CONFIG['dec_thresh']
r_new = recall_score(test_df['is_mule'], y_pred_new)
p_new = precision_score(test_df['is_mule'], y_pred_new)
f1_new = f1_score(test_df['is_mule'], y_pred_new)

print(f"BASELINE METRICS (Test Set):")
print(f"  Precision: {p_base*100:.2f}%, Recall: {r_base*100:.2f}%, F1: {f1_base:.4f}")

print(f"\nNEW DEVICE-LINKED METRICS (Test Set):")
print(f"  Precision: {p_new*100:.2f}%, Recall: {r_new*100:.2f}%, F1: {f1_new:.4f}")
print(f"  Recall Delta: +{(r_new - r_base)*100:.2f}%")

# 6. Check Hard Negative
fam_sample = df[df['device_id'].str.contains("FAMILY")].head(1)
if not fam_sample.empty:
    print(f"\n--- LEGITIMATE HARD NEGATIVE CHECK ---")
    print(f"Account {fam_sample.index[0]} shares device {fam_sample['device_id'].values[0]} with {fam_sample['shared_device_count'].values[0] - 1} other family members.")
    fam_score = (fam_sample['shared_device_count'] >= best_n).values[0] * best_weight
    print(f"Device Identity Risk Contribution: {fam_score}")
    if fam_score == 0:
        print("PASS: Legitimate family sharing safely ignored.")
    else:
        print("FAIL: Legitimate family penalized.")
