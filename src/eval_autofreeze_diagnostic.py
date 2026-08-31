"""
DIAGNOSTIC: Decompose the 17,657 auto-freeze false positives.
Where do they come from? Is Tier 0 too loose, or Tier 1 (≥4 hops)?
"""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== DIAGNOSTIC: WHY ARE 17,657 LEGITIMATE ACCOUNTS AUTO-FROZEN? ===\n")

# Parse Patterns
pattern_nodes_all = set()
with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('BEGIN') and not line.startswith('END'):
            parts = line.split(',')
            if len(parts) >= 5:
                pattern_nodes_all.add(parts[2])
                pattern_nodes_all.add(parts[4])

print("Loading CSV...")
df = pd.read_csv(csv_path, usecols=['Account', 'Account.1', 'Amount Paid', 'Payment Format'])
df['is_safe_format'] = df['Payment Format'].isin(['Cheque', 'Credit Card'])
df['safe_amount'] = df['Amount Paid'] * df['is_safe_format']

out_stats = df.groupby('Account').agg(
    tot_out=('Amount Paid', 'sum'),
    out_tx_count=('Amount Paid', 'count'),
    out_unique_dest=('Account.1', 'nunique')
)
in_stats = df.groupby('Account.1').agg(
    tot_in=('Amount Paid', 'sum'),
    safe_in=('safe_amount', 'sum')
)

accounts_df = out_stats.join(in_stats, how='inner')
accounts_df['tot_in'] = accounts_df['tot_in'].fillna(0.0)
accounts_df['tot_out'] = accounts_df['tot_out'].fillna(0.0)
accounts_df['safe_in'] = accounts_df['safe_in'].fillna(0.0)
accounts_df['pass_through_ratio'] = np.minimum(accounts_df['tot_out'] / np.maximum(accounts_df['tot_in'], 1e-6), 1.0)
accounts_df['safe_ratio'] = accounts_df['safe_in'] / np.maximum(accounts_df['tot_in'], 1e-6)
accounts_df['dest_concentration_ratio'] = accounts_df['out_unique_dest'] / np.maximum(accounts_df['out_tx_count'], 1)
accounts_df['is_true_mule'] = accounts_df.index.isin(pattern_nodes_all)

# Same split
all_idx = accounts_df.index.values
y = accounts_df['is_true_mule'].values
X_train_val, X_test, y_train_val, y_test = train_test_split(all_idx, y, test_size=0.20, random_state=42, stratify=y)
test_df = accounts_df.loc[X_test].copy()
test_mules_count = test_df['is_true_mule'].sum()
test_legit_count = len(test_df) - test_mules_count

print(f"Test Split: {len(test_df):,} accounts ({test_mules_count} mules, {test_legit_count:,} legit)\n")

# ============================================================
# STEP 1: HOW MANY ACCOUNTS PASS TIER 0?
# ============================================================
# Tier 0 Rule A: pass_through > 0.90
tier0_a = test_df['pass_through_ratio'] > 0.90
# Tier 0 Rule B: pass_through > 0.50 AND dest_concentration <= 0.70 AND out_tx >= 3
tier0_b = (test_df['pass_through_ratio'] > 0.50) & (test_df['dest_concentration_ratio'] <= 0.70) & (test_df['out_tx_count'] >= 3)
tier0_gate = tier0_a | tier0_b

t0_total = tier0_gate.sum()
t0_mules = (tier0_gate & test_df['is_true_mule']).sum()
t0_legit = (tier0_gate & ~test_df['is_true_mule']).sum()

print("=" * 70)
print("TIER 0 FUNNEL ANALYSIS")
print("=" * 70)
print(f"Total accounts passing Tier 0:    {t0_total:,} / {len(test_df):,} ({t0_total/len(test_df)*100:.1f}%)")
print(f"  - True Mules passing Tier 0:    {t0_mules:,} / {test_mules_count} ({t0_mules/test_mules_count*100:.1f}% mule recall at T0)")
print(f"  - Legit accounts passing Tier 0: {t0_legit:,} / {test_legit_count:,} ({t0_legit/test_legit_count*100:.1f}%)")
print()

# Sub-breakdown: which Tier 0 rule is the problem?
t0a_total = tier0_a.sum()
t0a_legit = (tier0_a & ~test_df['is_true_mule']).sum()
t0a_mules = (tier0_a & test_df['is_true_mule']).sum()

t0b_only = tier0_b & ~tier0_a  # accounts that pass ONLY via rule B
t0b_total = t0b_only.sum()
t0b_legit = (t0b_only & ~test_df['is_true_mule']).sum()
t0b_mules = (t0b_only & test_df['is_true_mule']).sum()

print("  Rule A (pass_through > 0.90):")
print(f"    Total: {t0a_total:,}  |  Mules: {t0a_mules:,}  |  Legit: {t0a_legit:,}")
print(f"  Rule B (pass_through > 0.50 AND dest_conc <= 0.70 AND out_tx >= 3) [EXCLUSIVE]:")
print(f"    Total: {t0b_total:,}  |  Mules: {t0b_mules:,}  |  Legit: {t0b_legit:,}")
print()

# Distribution of pass_through_ratio for legit accounts passing Tier 0
legit_pass = test_df[tier0_gate & ~test_df['is_true_mule']]
print("Pass-Through Ratio distribution for LEGIT accounts passing Tier 0:")
for pct in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]:
    cnt = (legit_pass['pass_through_ratio'] >= pct).sum()
    print(f"  pass_through >= {pct:.2f}: {cnt:,} legit accounts ({cnt/test_legit_count*100:.1f}%)")

print()

# ============================================================
# STEP 2: BUILD GRAPH AND CHECK TIER 1 HOP THRESHOLD
# ============================================================
print("Building Graph & Computing 15-Hop Traversal...")
G = nx.from_pandas_edgelist(df, 'Account', 'Account.1', create_using=nx.DiGraph())

deep_out_counts = {}
deep_in_counts = {}

for i, node in enumerate(test_df.index):
    if i % 20000 == 0:
        print(f"  BFS progress: {i:,} / {len(test_df):,}")
    if not G.has_node(node): continue
    
    v_down = set()
    q = [(node, 0)]
    while q:
        if len(v_down) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for succ in G.successors(c):
            if succ not in v_down:
                v_down.add(succ)
                q.append((succ, d + 1))
    
    v_up = set()
    q = [(node, 0)]
    while q:
        if len(v_up) >= 25: break
        c, d = q.pop(0)
        if d >= 15: continue
        for pred in G.predecessors(c):
            if pred not in v_up:
                v_up.add(pred)
                q.append((pred, d + 1))
                
    deep_out_counts[node] = len(v_down)
    deep_in_counts[node] = len(v_up)

test_df['deep_out'] = test_df.index.map(lambda x: deep_out_counts.get(x, 0))
test_df['deep_in'] = test_df.index.map(lambda x: deep_in_counts.get(x, 0))
test_df['max_reach'] = test_df[['deep_out', 'deep_in']].max(axis=1)

# Apply safe_ratio filter
safe_filter = ~(test_df['safe_ratio'] > 0.50)

# Current auto-freeze: Tier 0 AND (deep_out >= 4 OR deep_in >= 4) AND safe_ratio <= 0.50
auto_freeze_current = tier0_gate & ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) & safe_filter
fp_current = (auto_freeze_current & ~test_df['is_true_mule']).sum()
tp_current = (auto_freeze_current & test_df['is_true_mule']).sum()

print()
print("=" * 70)
print("TIER 1 HOP THRESHOLD ANALYSIS")
print("=" * 70)
print(f"\nCurrent Config (hops >= 4): TP={tp_current}, FP={fp_current:,}, Precision={tp_current/(tp_current+fp_current)*100:.2f}%")

# Sweep hop thresholds to see the trade-off
print("\nSweep: What if we require MORE hops for auto-freeze?")
print(f"{'Min Hops':>10} {'TP (Mules)':>12} {'FP (Legit)':>12} {'Precision':>10} {'Recall':>10}")
print("-" * 60)
for min_hops in [4, 5, 6, 7, 8, 9, 10, 12, 15]:
    auto = tier0_gate & ((test_df['deep_out'] >= min_hops) | (test_df['deep_in'] >= min_hops)) & safe_filter
    tp = (auto & test_df['is_true_mule']).sum()
    fp = (auto & ~test_df['is_true_mule']).sum()
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / test_mules_count * 100
    print(f"{min_hops:>10} {tp:>12,} {fp:>12,} {prec:>9.2f}% {rec:>9.2f}%")

# ============================================================
# STEP 3: SWEEP TIER 0 TIGHTENING
# ============================================================
print()
print("=" * 70)
print("TIER 0 TIGHTENING ANALYSIS")
print("=" * 70)
print("\nWhat if we raise the pass-through threshold (instead of 0.90)?")
print(f"{'PT Threshold':>13} {'T0 Legit Pass':>14} {'Auto-FP':>10} {'Auto-TP':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 72)
for pt_thresh in [0.90, 0.92, 0.95, 0.97, 0.99]:
    t0 = (test_df['pass_through_ratio'] > pt_thresh) | tier0_b
    auto = t0 & ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) & safe_filter
    tp = (auto & test_df['is_true_mule']).sum()
    fp = (auto & ~test_df['is_true_mule']).sum()
    t0_l = (t0 & ~test_df['is_true_mule']).sum()
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / test_mules_count * 100
    print(f"{pt_thresh:>13.2f} {t0_l:>14,} {fp:>10,} {tp:>10,} {prec:>9.2f}% {rec:>9.2f}%")

# ============================================================
# STEP 4: PERCENTILE-BASED TIER 0 (population-relative)
# ============================================================
print()
print("=" * 70)
print("PERCENTILE-BASED TIER 0 (population-relative)")
print("=" * 70)
pt_vals = test_df['pass_through_ratio']
for pctile in [90, 95, 97, 99]:
    cutoff = np.percentile(pt_vals, pctile)
    t0 = (pt_vals > cutoff) | tier0_b
    auto = t0 & ((test_df['deep_out'] >= 4) | (test_df['deep_in'] >= 4)) & safe_filter
    tp = (auto & test_df['is_true_mule']).sum()
    fp = (auto & ~test_df['is_true_mule']).sum()
    prec = tp / max(tp + fp, 1) * 100
    rec = tp / test_mules_count * 100
    print(f"  P{pctile} cutoff={cutoff:.4f} -> TP={tp}, FP={fp:,}, Precision={prec:.2f}%, Recall={rec:.2f}%")

# ============================================================
# STEP 5: COMBINED SWEEP: TIGHTER TIER 0 + HIGHER HOP THRESHOLD
# ============================================================
print()
print("=" * 70)
print("COMBINED SWEEP: TIGHTER TIER 0 + HIGHER HOP THRESHOLD")
print("=" * 70)
print(f"{'PT Thresh':>10} {'Min Hops':>10} {'TP':>8} {'FP':>8} {'Precision':>10} {'Recall':>10} {'Review TP':>10} {'Total Recall':>12}")
print("-" * 90)
for pt_thresh in [0.90, 0.95, 0.97]:
    for min_hops in [4, 6, 8, 10]:
        t0 = (test_df['pass_through_ratio'] > pt_thresh) | tier0_b
        auto = t0 & ((test_df['deep_out'] >= min_hops) | (test_df['deep_in'] >= min_hops)) & safe_filter
        tp = (auto & test_df['is_true_mule']).sum()
        fp = (auto & ~test_df['is_true_mule']).sum()
        
        # Review tier stays: extreme topology D>=15 not auto-frozen
        review = ((test_df['deep_out'] >= 15) | (test_df['deep_in'] >= 15)) & safe_filter & ~auto
        tp_rev = (review & test_df['is_true_mule']).sum()
        
        total_rec = (tp + tp_rev) / test_mules_count * 100
        prec = tp / max(tp + fp, 1) * 100
        rec = tp / test_mules_count * 100
        print(f"{pt_thresh:>10.2f} {min_hops:>10} {tp:>8} {fp:>8,} {prec:>9.2f}% {rec:>9.2f}% {tp_rev:>10} {total_rec:>11.2f}%")

print()
print("=== DIAGNOSTIC COMPLETE ===")
