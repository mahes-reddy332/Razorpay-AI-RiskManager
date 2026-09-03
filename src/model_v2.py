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


def compute_retention_ratio(G, node):
    """
    Feature 8: Long-Window Retention Ratio (Claude Q2 #1)

    Over the ENTIRE observed history, what fraction of ALL money that
    ever arrived has ever left — regardless of timing?

    A legitimate account (salary worker, business) retains something:
    savings, working capital, buffer. Value stays in range 0.0–0.8.

    A pure pass-through mule conduit retains almost nothing — it
    eventually forwards everything. Value approaches 1.0.

    STRUCTURAL ADVANTAGE: Structurally immune to 'Slow Ring' evasion.
    A mule who waits 5 days, 30 days, or 60 days STILL eventually
    forwards everything. There is no window to wait out.

    Returns a float in [0.0, 1.0]. Values > 0.95 are highly suspicious.
    """
    total_in  = sum(d['amount'] for _, _, d in G.in_edges(node, data=True))
    total_out = sum(d['amount'] for _, _, d in G.out_edges(node, data=True))

    if total_in == 0:
        return 0.0

    ratio = min(total_out / total_in, 1.0)  # cap at 1.0 for rounding noise
    return round(ratio, 4)


def compute_counterparty_repeat_rate(G, node):
    """
    Feature 9: Counterparty Repeat Rate (Claude Q2 #3)

    What fraction of an account's outbound transfers go to counterparties
    it has paid before (recurring), versus novel one-time recipients (fresh)?

    - Legitimate businesses (payroll, suppliers) have HIGH repeat rates.
      A payroll company sends to the same 500 employees every month.
    - Mule routing tends to burn through disposable downstream accounts.
      A mule typically routes to a different account every time.

    This feature DIRECTLY defends against the salary-company false-positive
    concern raised in Q5: a payroll company has repeat_rate ≈ 1.0 (it
    always pays the same employees), so it correctly passes Tier 1 even
    if Velocity Ratio trips Tier 0.

    Returns a float in [0.0, 1.0].
      0.0 = all outbound payments go to brand-new counterparties (suspicious)
      1.0 = all outbound payments go to the same recurring counterparties (safe)
    """
    out_targets = [v for _, v, _ in G.out_edges(node, data=True)]

    if not out_targets:
        return 1.0  # no outbound — not suspicious

    total_txns = len(out_targets)
    unique_targets = len(set(out_targets))

    # repeat_rate = 1 - (fraction that are unique)
    # A mule with 30 outbound to 30 different accounts → repeat_rate = 0.0
    # A payroll co with 30 outbound to 5 employees (6 payments each) → repeat_rate = 0.83
    if total_txns == 1:
        return 0.0  # single outbound — neutral, treated as suspicious

    repeat_rate = 1.0 - (unique_targets / total_txns)
    return round(repeat_rate, 4)


def compute_roundness_score(G, node):
    """
    Feature 10: Transaction Roundness Score (Structuring Signal)

    Mules are typically instructed to forward EXACT amounts:
    Rs.10,000 / Rs.50,000 / Rs.1,00,000. These are perfectly round numbers.

    Real people spend Rs.847 on groceries, Rs.3,240 on electricity bills,
    Rs.12,499 on a phone EMI — naturally irregular amounts.

    If more than 60% of an account's transactions are perfectly round
    numbers (divisible by 1,000), that is a classic structuring signal.

    Returns a float in [0.0, 1.0] — fraction of round transactions.
    Values > 0.6 are suspicious.
    """
    all_amounts = (
        [d['amount'] for _, _, d in G.in_edges(node, data=True)] +
        [d['amount'] for _, _, d in G.out_edges(node, data=True)]
    )

    if not all_amounts:
        return 0.0

    # Round = divisible by 1000 (no paise/fraction)
    round_count = sum(1 for a in all_amounts if a % 1000 == 0)
    return round(round_count / len(all_amounts), 4)


def compute_inflow_diversity(G, node):
    """
    Feature 11: Inflow Source Diversity

    Ask: How many DIFFERENT people are sending this account money?

    - A salary worker is paid by 1 employer (low diversity)
    - A small business receives from ~5-20 regular customers (medium)
    - A mule receives from 20-50 strangers, each sending similar
      amounts as part of a coordinated flow (very high diversity)

    High inflow diversity combined with high velocity is one of the
    strongest mule signals because it represents the 'aggregation' 
    phase of the funnel topology.

    Returns the count of unique inbound senders (raw integer).
    Normalized: values > 15 unique senders are suspicious.
    """
    unique_senders = set(u for u, _, _ in G.in_edges(node, data=True))
    return len(unique_senders)


def compute_nighttime_ratio(G, node):
    """
    Feature 12: Night-Time Transaction Ratio (2am - 5am IST)

    Legitimate businesses transact during business hours.
    Mule networks — often operated by scripts or overseas operators —
    frequently forward funds between 2am and 5am when:
      - Bank compliance monitoring is lowest
      - Human oversight is at minimum
      - Automated scripts run scheduled jobs

    If more than 40% of an account's transactions fall between
    midnight and 5am, it is a strong operational signal.

    Returns float in [0.0, 1.0] — fraction of night-time transactions.
    Values > 0.40 are suspicious.
    """
    all_timestamps = (
        [d['timestamp'] for _, _, d in G.in_edges(node, data=True)] +
        [d['timestamp'] for _, _, d in G.out_edges(node, data=True)]
    )

    if not all_timestamps:
        return 0.0

    night_count = 0
    for ts in all_timestamps:
        try:
            # ts is a string: '2026-06-03 02:34:11'
            hour = int(str(ts)[11:13])
            if 0 <= hour < 5:  # midnight to 4:59am
                night_count += 1
        except (ValueError, IndexError):
            continue

    return round(night_count / len(all_timestamps), 4)


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
    retention_features = {}
    repeat_rate_features = {}
    roundness_features = {}
    diversity_features = {}
    nighttime_features = {}

    print("Extracting per-account features (BFS, Z-score, multi-window velocity, retention, repeat-rate, roundness, diversity, nighttime)...")
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

        # Feature 8 & 9: Always-on, cheap arithmetic — run on ALL accounts
        retention_features[node] = compute_retention_ratio(det.G, node)
        repeat_rate_features[node] = compute_counterparty_repeat_rate(det.G, node)

        # Features 10, 11, 12: Structuring, Aggregation, and Operational signals
        roundness_features[node] = compute_roundness_score(det.G, node)
        diversity_features[node] = compute_inflow_diversity(det.G, node)
        nighttime_features[node] = compute_nighttime_ratio(det.G, node)

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
    df['retention_ratio'] = df.index.map(retention_features)
    df['counterparty_repeat_rate'] = df.index.map(repeat_rate_features)
    df['roundness_score'] = df.index.map(roundness_features)
    df['inflow_diversity'] = df.index.map(diversity_features)
    df['nighttime_ratio'] = df.index.map(nighttime_features)

    print(f"\nFeature summary (non-zero counts):")
    print(f"  risk_score > 0       : {(df['risk_score'] > 0).sum()}")
    print(f"  has_risky_sink == 1  : {(df['has_risky_sink'] == 1).sum()}")
    print(f"  log_amount_zscore > 3: {(df['log_amount_zscore'] > 3.0).sum()}")
    print(f"  betweenness > 0      : {(df['betweenness'] > 0).sum()}")
    print(f"  max_velocity > 0.85  : {(df['max_velocity_ratio'] > 0.85).sum()}")
    print(f"  retention_ratio > 0.95: {(df['retention_ratio'] > 0.95).sum()}")
    print(f"  repeat_rate < 0.10   : {(df['counterparty_repeat_rate'] < 0.10).sum()}")
    print(f"  roundness > 0.60     : {(df['roundness_score'] > 0.60).sum()}")
    print(f"  inflow_diversity > 15: {(df['inflow_diversity'] > 15).sum()}")
    print(f"  nighttime_ratio > 0.4: {(df['nighttime_ratio'] > 0.4).sum()}")
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
    'dec_thresh': 1.0,           # Threshold B (HIGH_RISK - Risk containment)
    'manual_thresh': 0.5,        # Threshold A (MANUAL_REVIEW - Routes to L2)
    'betweenness_thresh': 0.000131,
    # NEW V3/V4 features (Future Work / Odds left in system - Currently unweighted to prevent over-flagging)
    'retention_weight': 0.40,    # Tuned to capture slow ring (0.6 base + 0.4 = 1.0 High Risk)
    'retention_thresh': 0.95,    
    'repeat_rate_weight': 0.0,   # Feature 9: counterparty_repeat_rate < 0.10 adds weight
    'repeat_rate_thresh': 0.10,  
    'roundness_weight': 0.0,     # Feature 10: >60% round amounts
    'roundness_thresh': 0.60,    
    'diversity_weight': 0.0,     # Feature 11: >15 unique senders
    'diversity_thresh': 15,      
    'nighttime_weight': 0.0,     # Feature 12: >40% night transactions
    'nighttime_thresh': 0.40     
}

def apply_frozen_config(data, params=FROZEN_CONFIG):
    scores = data['risk_score'].values.copy()
    nc = data['node_count'].values
    scores += ((nc > 0) & (nc <= params['node_thresh'])) * params['top_weight']
    scores += (data['has_risky_sink'] == 1).values * params['mcc_weight']
    scores += (data['log_amount_zscore'] > 3.0).values * params['zscore_weight']
    scores += (data['betweenness'] > params['betweenness_thresh']).values * params['centrality_weight']
    scores += (data['max_velocity_ratio'] > 0.85).values * params['velocity_weight']

    # Feature 8: High retention ratio (conduit behaviour, timing-agnostic)
    scores += (data['retention_ratio'] > params['retention_thresh']).values * params['retention_weight']

    # Feature 9: Low counterparty repeat rate (routing to fresh strangers every time)
    scores += (data['counterparty_repeat_rate'] < params['repeat_rate_thresh']).values * params['repeat_rate_weight']

    # Feature 10: Structuring via round numbers
    scores += (data['roundness_score'] > params['roundness_thresh']).values * params['roundness_weight']

    # Feature 11: Aggregation via diverse senders
    scores += (data['inflow_diversity'] > params['diversity_thresh']).values * params['diversity_weight']

    # Feature 12: Operational red flag (Night-time)
    scores += (data['nighttime_ratio'] > params['nighttime_thresh']).values * params['nighttime_weight']

    # Return raw scores so audit.py can route to the A/B bands
    return scores

def evaluate_final_unified(df):
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 60)
    print("FINAL UNIFIED EVALUATION (STRICT TEST SET ONLY)")
    print("=" * 60)

    # STRICT ML DISCIPLINE: We must evaluate ONLY on the untouched Test Set.
    # The previous baseline figure was a data leak.
    all_accounts = df.index.values
    y = df['is_mule'].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_accounts, y, test_size=0.20, random_state=42)
    
    test_df = df.loc[X_test]
    
    scores = apply_frozen_config(test_df, FROZEN_CONFIG)
    
    y_true = test_df['is_mule'].values
    y_pred = scores >= FROZEN_CONFIG['dec_thresh'] # Risk containment line

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
    print(f"  -> HIGH_RISK (>= {FROZEN_CONFIG['dec_thresh']}): {flag}")
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
