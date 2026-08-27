import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from detector import MVPDetector
from tracer import get_chain_metrics

def extract_features():
    print("Extracting features for V2 Pipeline...")
    det = MVPDetector('data/accounts.csv', 'data/transactions.csv')
    det.build_graph()
    det.score_accounts()  # Gives baseline pt_score + dormancy_score in det.df_results
    
    df = det.df_results.copy()
    df = df.set_index('account_id')
    
    node_counts = {}
    risky_mcc_flags = {}
    
    # We only care about accounts that scored > 0 in MVP (to save BFS computation)
    # Actually, let's extract topology and MCC for all accounts in df
    print("Running BFS on accounts to extract topology and MCC features...")
    count = 0
    for node in df.index:
        # Optimization: only trace if base score indicates activity
        if df.loc[node, 'risk_score'] > 0:
            total_hops, node_count, fwd_sinks = get_chain_metrics(det.G, node)
            node_counts[node] = node_count
            
            is_risky = 0
            for sink in fwd_sinks:
                mcc = det.G.nodes[sink].get('mcc_code', 'NONE')
                if mcc in ["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"]:
                    is_risky = 1
                    break
            risky_mcc_flags[node] = is_risky
        else:
            node_counts[node] = 0
            risky_mcc_flags[node] = 0
            
        count += 1
        if count % 500 == 0: print(f"Processed {count} nodes...")
            
    df['node_count'] = df.index.map(node_counts)
    df['has_risky_sink'] = df.index.map(risky_mcc_flags)
    return df

def tune_and_evaluate(df):
    all_accounts = df.index.values
    y = df['is_mule'].values
    
    # 60% Train, 20% Val, 20% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_accounts, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.20
    
    train_df = df.loc[X_train]
    val_df = df.loc[X_val]
    test_df = df.loc[X_test]
    
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("\n--- Tuning on Validation Set ---")
    best_f1 = -1
    best_params = {}
    
    # The composite score = mvp_score (0.0 to 1.0)
    # If node_count <= node_threshold (tight topology), add topology_weight
    # If has_risky_sink == 1, add mcc_weight
    
    # We sweep these parameters to maximize F1 on Val
    node_thresholds = list(range(4, 21, 2))
    topology_weights = [0.0, 0.2, 0.4, 0.6, 0.8]
    mcc_weights = [0.0, 0.2, 0.4, 0.6, 0.8]
    decision_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    for nt in node_thresholds:
        for tw in topology_weights:
            for mw in mcc_weights:
                for dt in decision_thresholds:
                    
                    scores = val_df['risk_score'].copy() # Base MVP score
                    
                    # Apply topology weight
                    mask_topology = (val_df['node_count'] > 0) & (val_df['node_count'] <= nt)
                    scores[mask_topology] += tw
                    
                    # Apply MCC weight
                    mask_mcc = val_df['has_risky_sink'] == 1
                    scores[mask_mcc] += mw
                    
                    preds = scores >= dt
                    f1 = f1_score(val_df['is_mule'], preds, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {'node_thresh': nt, 'top_weight': tw, 'mcc_weight': mw, 'dec_thresh': dt}
                        
    print(f"Best Validation F1: {best_f1:.3f}")
    print(f"Frozen Parameters: {best_params}")
    
    print("\n--- Evaluating Frozen Config on Test Set (One-Time Run) ---")
    
    def apply_config(data, params):
        scores = data['risk_score'].copy()
        mask_topology = (data['node_count'] > 0) & (data['node_count'] <= params['node_thresh'])
        scores[mask_topology] += params['top_weight']
        
        mask_mcc = data['has_risky_sink'] == 1
        scores[mask_mcc] += params['mcc_weight']
        
        return scores >= params['dec_thresh']
        
    y_true = test_df['is_mule']
    y_pred = apply_config(test_df, best_params)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.ravel()[1] if len(cm.ravel()) > 1 else 0
    fn = cm.ravel()[2] if len(cm.ravel()) > 2 else 0
    
    print(f"Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {cm.ravel()[3] if len(cm.ravel())>3 else sum(y_true)}")

if __name__ == "__main__":
    df = extract_features()
    tune_and_evaluate(df)
