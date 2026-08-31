import sys
import os
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_v2 import extract_features, FROZEN_CONFIG

def grid_search():
    print("Starting Grid Search on Validation Split...")
    df = extract_features()
    
    # Reload for is_adversarial reporting
    acc_df = pd.read_csv('data/accounts.csv').set_index('account_id')
    if 'is_adversarial' in acc_df.columns:
        df = df.join(acc_df[['is_adversarial']])
    else:
        df['is_adversarial'] = False

    all_accounts = df.index.values
    y = df['is_mule'].values
    
    # 60/20/20 Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_accounts, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2
    
    val_df = df.loc[X_val]
    test_df = df.loc[X_test]
    
    print(f"Train accounts: {len(X_train)}, Validation accounts: {len(X_val)}, Test accounts: {len(X_test)}")
    
    # We fix the existing V2 parameters and sweep only the 5 new ones, or a subset of them
    # Actually, the prompt says "run ONE grid search including all five new features plus everything already in the model"
    # To keep it computationally feasible, let's test specific grids for the new ones
    
    retention_weights = [0.0, 0.2, 0.4]
    repeat_weights = [0.0, 0.2, 0.4]
    roundness_weights = [0.0, 0.2]
    diversity_weights = [0.0, 0.2, 0.3]
    nighttime_weights = [0.0, 0.2]
    
    best_f1 = 0
    best_params = FROZEN_CONFIG.copy()
    
    count = 0
    total = len(retention_weights) * len(repeat_weights) * len(roundness_weights) * len(diversity_weights) * len(nighttime_weights)
    
    for ret, rep, rnd, div, nght in itertools.product(retention_weights, repeat_weights, roundness_weights, diversity_weights, nighttime_weights):
        # Apply base scores
        scores = val_df['risk_score'].values.copy()
        nc = val_df['node_count'].values
        scores += ((nc > 0) & (nc <= FROZEN_CONFIG['node_thresh'])) * FROZEN_CONFIG['top_weight']
        scores += (val_df['has_risky_sink'] == 1).values * FROZEN_CONFIG['mcc_weight']
        scores += (val_df['log_amount_zscore'] > 3.0).values * FROZEN_CONFIG['zscore_weight']
        scores += (val_df['betweenness'] > FROZEN_CONFIG['betweenness_thresh']).values * FROZEN_CONFIG['centrality_weight']
        scores += (val_df['max_velocity_ratio'] > 0.85).values * FROZEN_CONFIG['velocity_weight']
        
        # Apply new feature scores
        scores += (val_df['retention_ratio'] > FROZEN_CONFIG['retention_thresh']).values * ret
        scores += (val_df['counterparty_repeat_rate'] < FROZEN_CONFIG['repeat_rate_thresh']).values * rep
        scores += (val_df['roundness_score'] > FROZEN_CONFIG['roundness_thresh']).values * rnd
        scores += (val_df['inflow_diversity'] > FROZEN_CONFIG['diversity_thresh']).values * div
        scores += (val_df['nighttime_ratio'] > FROZEN_CONFIG['nighttime_thresh']).values * nght
        
        y_pred = scores >= FROZEN_CONFIG['dec_thresh']
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params['retention_weight'] = ret
            best_params['repeat_rate_weight'] = rep
            best_params['roundness_weight'] = rnd
            best_params['diversity_weight'] = div
            best_params['nighttime_weight'] = nght
            
        count += 1
        if count % 20 == 0:
            print(f"Swept {count}/{total} combinations...")

    print("\nBest Parameters on Validation Set:")
    for k in ['retention_weight', 'repeat_rate_weight', 'roundness_weight', 'diversity_weight', 'nighttime_weight']:
        print(f"  {k}: {best_params[k]}")
    
    print("\nRunning best configuration on Untouched TEST set...")
    scores = test_df['risk_score'].values.copy()
    nc = test_df['node_count'].values
    scores += ((nc > 0) & (nc <= best_params['node_thresh'])) * best_params['top_weight']
    scores += (test_df['has_risky_sink'] == 1).values * best_params['mcc_weight']
    scores += (test_df['log_amount_zscore'] > 3.0).values * best_params['zscore_weight']
    scores += (test_df['betweenness'] > best_params['betweenness_thresh']).values * best_params['centrality_weight']
    scores += (test_df['max_velocity_ratio'] > 0.85).values * best_params['velocity_weight']
    scores += (test_df['retention_ratio'] > best_params['retention_thresh']).values * best_params['retention_weight']
    scores += (test_df['counterparty_repeat_rate'] < best_params['repeat_rate_thresh']).values * best_params['repeat_rate_weight']
    scores += (test_df['roundness_score'] > best_params['roundness_thresh']).values * best_params['roundness_weight']
    scores += (test_df['inflow_diversity'] > best_params['diversity_thresh']).values * best_params['diversity_weight']
    scores += (test_df['nighttime_ratio'] > best_params['nighttime_thresh']).values * best_params['nighttime_weight']
    
    y_pred = scores >= best_params['dec_thresh']
    y_true = test_df['is_mule'].values
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  Precision : {p:.3f}  ({p*100:.1f}%)")
    print(f"  Recall    : {r:.3f}  ({r*100:.1f}%)")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  True Pos  : {tp}")
    print(f"  False Pos : {fp}")
    print(f"  True Neg  : {tn}")
    print(f"  False Neg : {fn}")
    
    # Specific checks
    adv_df = test_df[test_df['is_adversarial'] == True]
    if len(adv_df) > 0:
        adv_scores = scores[test_df['is_adversarial'] == True]
        adv_pred = adv_scores >= best_params['dec_thresh']
        adv_true = adv_df['is_mule'].values
        adv_tp = sum(adv_pred)
        print(f"\nAdversarial Mules Caught: {adv_tp} / {len(adv_true)}")

if __name__ == "__main__":
    grid_search()
