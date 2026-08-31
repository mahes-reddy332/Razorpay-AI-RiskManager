import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

csv_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'
patterns_path = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Patterns.txt'

print("=== PART A: PARSING GROUND TRUTH LABELS ===")
# Parse all laundering transactions and patterns
launder_txns = set()
pattern_nodes_all = set()
pattern_nodes_intermediate = set()
pattern_nodes_source = set()
pattern_nodes_sink = set()
cycle_nodes = set()

current_header = ""
current_chain = []
all_chains = []

with open(patterns_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT'):
            current_header = line
            current_chain = []
        elif line.startswith('END LAUNDERING ATTEMPT'):
            if current_chain:
                all_chains.append((current_header, current_chain))
            current_chain = []
        elif current_header and line:
            parts = line.split(',')
            src = parts[2]
            tgt = parts[4]
            current_chain.append((src, tgt, float(parts[7]), parts[0], parts[9]))

for header, txns in all_chains:
    srcs = set(t[0] for t in txns)
    tgts = set(t[1] for t in txns)
    pattern_nodes_all.update(srcs.union(tgts))
    inter = srcs.intersection(tgts)
    pattern_nodes_intermediate.update(inter)
    pattern_nodes_source.update(srcs - tgts)
    pattern_nodes_sink.update(tgts - srcs)
    
    if "CYCLE" in header:
        cycle_nodes.update(srcs.union(tgts))

print(f"Total Unique Nodes in Patterns: {len(pattern_nodes_all):,}")
print(f"  - Intermediate Conduits (In & Out in pattern): {len(pattern_nodes_intermediate):,}")
print(f"  - Pattern Initiator Nodes (Source only): {len(pattern_nodes_source):,}")
print(f"  - Pattern Terminal Sinks (Sink only): {len(pattern_nodes_sink):,}")
print(f"  - Cycle Specific Nodes: {len(cycle_nodes):,}")

# Build graph from transactions
print("\nLoading CSV & Building Graph for evaluation...")
G = nx.DiGraph()
in_flows = {}
out_flows = {}
all_accounts = set()

# Process in chunks
chunk_count = 0
for chunk in pd.read_csv(csv_path, chunksize=1000000):
    for _, row in chunk.iterrows():
        src = row['Account']
        tgt = row['Account.1']
        amt = float(row['Amount Paid'])
        
        all_accounts.add(src)
        all_accounts.add(tgt)
        
        out_flows[src] = out_flows.get(src, 0.0) + amt
        in_flows[tgt] = in_flows.get(tgt, 0.0) + amt
        
        if not G.has_edge(src, tgt):
            G.add_edge(src, tgt, weight=amt)
        else:
            G[src][tgt]['weight'] += amt
            
    chunk_count += 1
    print(f"  Processed chunk {chunk_count}...")

print(f"Graph constructed: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges.")

# Eligible accounts (both in and out)
eligible_accounts = set(n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0)
print(f"Eligible accounts (in_degree > 0 and out_degree > 0): {len(eligible_accounts):,}")

# Ground truth mules for eligible nodes:
# An eligible node is a ground truth mule if it participated as an intermediate conduit or in a cycle
true_mules = eligible_accounts.intersection(pattern_nodes_all)
print(f"True Mule Accounts in eligible population: {len(true_mules):,}")

# PART B: BASELINE EVALUATION (Retention + Fan-out Topology)
print("\n=== PART B: RUNNING CLEAN BASELINE DETECTOR ===")
baseline_flagged = set()

for node in eligible_accounts:
    tot_in = in_flows.get(node, 0.0)
    tot_out = out_flows.get(node, 0.0)
    
    if tot_in > 0:
        retention_ratio = min(tot_out / tot_in, 1.0)
    else:
        retention_ratio = 0.0
        
    out_deg = G.out_degree(node)
    in_deg = G.in_degree(node)
    
    # Baseline logic: High pass-through (>0.95) AND (fan-out >= 4 OR fan-in >= 4)
    if retention_ratio > 0.95 and (out_deg >= 4 or in_deg >= 4):
        baseline_flagged.add(node)

# Metrics for Baseline
tp = len(baseline_flagged.intersection(true_mules))
fp = len(baseline_flagged - true_mules)
fn = len(true_mules - baseline_flagged)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"BASELINE METRICS:")
print(f"  Flagged: {len(baseline_flagged):,}")
print(f"  True Positives : {tp:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall   : {recall*100:.2f}%")
print(f"  F1 Score : {f1:.4f}")

# PART C: CYCLE DETECTION (Incremental Contribution)
print("\n=== PART C: TESTING DIRECTED CYCLE DETECTION ===")
# Find simple cycles in G
# Note: Full simple cycles on 500k graph is expensive, so we run on subgraphs with high retention or SCCs
sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
cycle_detected_nodes = set()
for comp in sccs:
    cycle_detected_nodes.update(comp)

print(f"Nodes participating in Strongly Connected Components / Cycles: {len(cycle_detected_nodes):,}")

# Check overlap with CYCLE ground truth
cycles_caught_by_baseline = len(baseline_flagged.intersection(cycle_nodes))
cycles_caught_by_cycle_engine = len(cycle_detected_nodes.intersection(cycle_nodes))
new_cycles_caught = len((cycle_detected_nodes - baseline_flagged).intersection(cycle_nodes))

print(f"Total True Cycle Nodes: {len(cycle_nodes):,}")
print(f"  - Caught by Baseline (Part B): {cycles_caught_by_baseline:,}")
print(f"  - Caught by Cycle Engine: {cycles_caught_by_cycle_engine:,}")
print(f"  - Incremental New Cycles Caught: {new_cycles_caught:,}")

# Combined Baseline + Cycle metrics
combined_flagged = baseline_flagged.union(cycle_detected_nodes)
tp_comb = len(combined_flagged.intersection(true_mules))
fp_comb = len(combined_flagged - true_mules)
fn_comb = len(true_mules - combined_flagged)
prec_comb = tp_comb / (tp_comb + fp_comb) if (tp_comb + fp_comb) > 0 else 0
rec_comb = tp_comb / (tp_comb + fn_comb) if (tp_comb + fn_comb) > 0 else 0
f1_comb = 2 * prec_comb * rec_comb / (prec_comb + rec_comb) if (prec_comb + rec_comb) > 0 else 0

print(f"\nCOMBINED (Baseline + Cycle Detection) METRICS:")
print(f"  Precision: {prec_comb*100:.2f}%")
print(f"  Recall   : {rec_comb*100:.2f}%")
print(f"  F1 Score : {f1_comb:.4f}")
