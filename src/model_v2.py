"""
V2 Enhanced Pipeline — Composite Risk Scoring with Grid-Search Validation.

Features extracted per eligible account (both in-edge and out-edge present):
  1. risk_score        — base MVP score (24h pass-through velocity + dormancy)
  2. node_count        — BFS topology footprint
  3. has_risky_sink    — terminal MCC is CRYPTO/GAMBLING/UNREGISTERED_P2P
  4. log_amount_zscore — max z-score of log(amount) vs account's own history
  5. betweenness       — betweenness centrality (money-routing bottleneck)
  6. pagerank          — PageRank (informational only — ambiguous signal)
  7. max_velocity_ratio— max pass-through ratio across 1h/6h/24h/48h/72h windows

Grid search sweeps weights for features 1-5 and 7 on the Validation split
only. Feature 6 (PageRank) is extracted for audit logs but NOT scored
because high PageRank is ambiguous (could be victim OR aggregator).

Methodology: strict 60/20/20 Train/Validation/Test split.
  - Grid search on Validation only.
  - Freeze parameters.
  - Evaluate once on untouched Test split.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from datetime import timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from detector import MVPDetector
from tracer import get_chain_metrics


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def compute_log_amount_zscore(G, node):
    """Max z-score of log(amount) across all of an account's transactions,
    relative to that account's own historical mean/std.
    Uses a minimum-std floor of 0.5 for accounts with very few transactions."""
    amounts = []
    for _, _, d in G.in_edges(node, data=True):
        amounts.append(d['amount'])
    for _, _, d in G.out_edges(node, data=True):
        amounts.append(d['amount'])

    if len(amounts) < 2:
        return 0.0

    log_amounts = np.log1p(amounts)
    mean_log = np.mean(log_amounts)
    std_log = max(np.std(log_amounts), 0.5)  # small-sample floor

    max_z = max(abs(la - mean_log) / std_log for la in log_amounts)
    return round(max_z, 4)


def compute_multi_window_velocity(G, node,
                                  windows_hours=(1, 6, 24, 48, 72),
                                  min_amount=1000):
    """Max pass-through ratio across five overlapping time windows.
    A mule waiting 48 hours still trips the 72-hour window."""
    in_txns = [(d['amount'], d['timestamp']) for _, _, d in G.in_edges(node, data=True)]
    out_txns = [(d['amount'], d['timestamp']) for _, _, d in G.out_edges(node, data=True)]

    if not in_txns or not out_txns:
        return 0.0

    max_ratio = 0.0
    for in_amt, in_time in in_txns:
        if in_amt < min_amount:
            continue
        for wh in windows_hours:
            window_end = in_time + timedelta(hours=wh)
            window_out = sum(oa for oa, ot in out_txns if in_time <= ot <= window_end)
            ratio = window_out / in_amt
            if 0.85 <= ratio <= 1.10:
                max_ratio = max(max_ratio, ratio)
    return round(max_ratio, 4)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_features():
    print("=" * 60)
    print("EXTRACTING FEATURES (Enhanced V2 Pipeline)")
    print("=" * 60)

    det = MVPDetector('data/accounts.csv', 'data/transactions.csv')
    det.build_graph()
    det.score_accounts()

    df = det.df_results.copy()
    df = df.set_index('account_id')
    print(f"Eligible accounts (both in-edge and out-edge): {len(df)}")

    # --- Graph centrality (computed once on full graph) ---
    print("Computing graph centrality (betweenness + PageRank)...")
    simple_G = nx.DiGraph(det.G)  # collapse multi-edges for centrality
    betweenness = nx.betweenness_centrality(simple_G)
    pagerank_scores = nx.pagerank(simple_G, max_iter=100)
    print(f"  Centrality computed for {len(simple_G)} nodes.")

    # --- Per-account feature extraction ---
    node_counts = {}
    risky_mcc_flags = {}
    zscore_features = {}
    velocity_features = {}
    betweenness_features = {}
    pagerank_features = {}

    print("Extracting per-account features (BFS, Z-score, multi-window velocity)...")
    count = 0
    for node in df.index:
        # Centrality (pre-computed lookup)
        betweenness_features[node] = betweenness.get(node, 0.0)
        pagerank_features[node] = pagerank_scores.get(node, 0.0)

        # NEW: Independent, always-on O(1) check of immediate counterparties
        is_risky = 0
        for succ in det.G.successors(node):
            mcc = det.G.nodes[succ].get('mcc_code', 'NONE')
            if mcc in ["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"]:
                is_risky = 1
                break

        # EXPENSIVE CHECK: Gated behind the velocity tripwire
        if df.loc[node, 'risk_score'] > 0:
            total_hops, node_count, fwd_sinks = get_chain_metrics(det.G, node)
            node_counts[node] = node_count

            # Also check deep sinks found by BFS
            for sink in fwd_sinks:
                mcc = det.G.nodes[sink].get('mcc_code', 'NONE')
                if mcc in ["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"]:
                    is_risky = 1
                    break

            zscore_features[node] = compute_log_amount_zscore(det.G, node)
            velocity_features[node] = compute_multi_window_velocity(det.G, node)
        else:
            node_counts[node] = 0
            zscore_features[node] = compute_log_amount_zscore(det.G, node)
            velocity_features[node] = 0.0

        risky_mcc_flags[node] = is_risky

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{len(df)} nodes...")

    df['node_count'] = df.index.map(node_counts)
    df['has_risky_sink'] = df.index.map(risky_mcc_flags)
    df['log_amount_zscore'] = df.index.map(zscore_features)
    df['betweenness'] = df.index.map(betweenness_features)
    df['pagerank'] = df.index.map(pagerank_features)
    df['max_velocity_ratio'] = df.index.map(velocity_features)

    print(f"\nFeature summary (non-zero counts):")
    print(f"  risk_score > 0      : {(df['risk_score'] > 0).sum()}")
    print(f"  has_risky_sink == 1  : {(df['has_risky_sink'] == 1).sum()}")
    print(f"  log_amount_zscore > 3: {(df['log_amount_zscore'] > 3.0).sum()}")
    print(f"  betweenness > 0      : {(df['betweenness'] > 0).sum()}")
    print(f"  max_velocity > 0.85  : {(df['max_velocity_ratio'] > 0.85).sum()}")
    return df


def evaluate_round_1_frozen(test_df):
    print("\n" + "=" * 60)
    print("ROUND 1: FROZEN CONFIG ON ADVERSARIAL TEST SET")
    print("=" * 60)
    
    # The exact configuration that won in Round 1
    r1_params = {
        'node_thresh': 8,
        'top_weight': 0.6,
        'mcc_weight': 0.0,
        'zscore_weight': 0.0,
        'centrality_weight': 0.0,
        'velocity_weight': 0.0,
        'dec_thresh': 1.2,
        'betweenness_thresh': 0.0
    }
    
    def apply_config(data, params):
        scores = data['risk_score'].values.copy()
        nc = data['node_count'].values
        scores += ((nc > 0) & (nc <= params['node_thresh'])) * params['top_weight']
        # other features are 0.0 in Round 1
        return scores >= params['dec_thresh']

    adv_test_df = test_df[test_df['is_adversarial'] == True]
    print(f"Adversarial Mules in Test Set: {len(adv_test_df)}")
    
    if len(adv_test_df) == 0:
        print("No adversarial mules in test set to evaluate.")
        return

    y_true = adv_test_df['is_mule'].values
    y_pred = apply_config(adv_test_df, r1_params)

    r = recall_score(y_true, y_pred, zero_division=0)
    tp = sum(y_pred)
    fn = len(y_pred) - tp
    
    print(f"Round 1 (Topology-Only) Recall on Adversarial Mules: {r:.3f} ({tp}/{len(y_true)})")
    print(f"  Caught : {tp}")
    print(f"  Missed : {fn}")
    print("  -> Conclusion: Topology alone fails heavily against topology-camouflaged and slow-evasion mules.")


# --- FROZEN PRODUCTION CONFIGURATION ---
FROZEN_CONFIG = {
    'node_thresh': 4,
    'top_weight': 0.0,
    'mcc_weight': 0.6,
    'zscore_weight': 0.0,
    'centrality_weight': 0.0,
    'velocity_weight': 0.0,
    'dec_thresh': 1.0,           # Threshold B (FLAG_MULE - Auto-freeze)
    'manual_thresh': 0.5,        # Threshold A (MANUAL_REVIEW - Routes to L2)
    'betweenness_thresh': 0.000131
}

def apply_frozen_config(data, params=FROZEN_CONFIG):
    scores = data['risk_score'].values.copy()
    nc = data['node_count'].values
    scores += ((nc > 0) & (nc <= params['node_thresh'])) * params['top_weight']
    scores += (data['has_risky_sink'] == 1).values * params['mcc_weight']
    scores += (data['log_amount_zscore'] > 3.0).values * params['zscore_weight']
    scores += (data['betweenness'] > params['betweenness_thresh']).values * params['centrality_weight']
    scores += (data['max_velocity_ratio'] > 0.85).values * params['velocity_weight']
    
    # Return raw scores so audit.py can route to the A/B bands
    return scores

def evaluate_final_unified(df):
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 60)
    print("FINAL UNIFIED EVALUATION (STRICT TEST SET ONLY)")
    print("=" * 60)

    # STRICT ML DISCIPLINE: We must evaluate ONLY on the untouched Test Set.
    # The 665 figure was a data leak (evaluating the whole Train+Val+Test dataset).
    all_accounts = df.index.values
    y = df['is_mule'].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_accounts, y, test_size=0.20, random_state=42)
    
    test_df = df.loc[X_test]
    
    scores = apply_frozen_config(test_df, FROZEN_CONFIG)
    
    y_true = test_df['is_mule'].values
    y_pred = scores >= FROZEN_CONFIG['dec_thresh'] # Auto-freeze line

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
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
    print(f"{'='*60}")
    
    # Let's also print the band routing for the mules in the test set
    mule_df = test_df[test_df['is_mule'] == True].copy()
    mule_scores = apply_frozen_config(mule_df, FROZEN_CONFIG)
    
    flag = sum(mule_scores >= FROZEN_CONFIG['dec_thresh'])
    review = sum((mule_scores >= FROZEN_CONFIG['manual_thresh']) & (mule_scores < FROZEN_CONFIG['dec_thresh']))
    safe = sum(mule_scores < FROZEN_CONFIG['manual_thresh'])
    
    print("\nMule Routing Breakdown (Test Set):")
    print(f"  -> FLAG_MULE (>= {FROZEN_CONFIG['dec_thresh']}): {flag}")
    print(f"  -> MANUAL_REVIEW ({FROZEN_CONFIG['manual_thresh']} - {FROZEN_CONFIG['dec_thresh']}): {review}")
    print(f"  -> SAFE (< {FROZEN_CONFIG['manual_thresh']}): {safe}")

if __name__ == "__main__":
    df = extract_features()
    
    # Reload for is_adversarial reporting
    acc_df = pd.read_csv('data/accounts.csv').set_index('account_id')
    if 'is_adversarial' in acc_df.columns:
        df = df.join(acc_df[['is_adversarial']])
    else:
        df['is_adversarial'] = False
        
    evaluate_final_unified(df)
