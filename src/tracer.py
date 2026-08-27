import sys
import os
import pandas as pd
import networkx as nx
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import MVPDetector

def get_chain_metrics(G, node, max_hours=24):
    """
    Performs a bidirectional BFS to reconstruct the continuous temporal chain
    passing through this account, and identifies final sink nodes.
    """
    in_txns = sorted(G.in_edges(node, data=True), key=lambda x: x[2]['timestamp'])
    if not in_txns: 
        return 0, 0, set()
    
    trigger_time = in_txns[-1][2]['timestamp']
    
    visited_edges = set()
    visited_nodes = set([node])
    fwd_sinks = set()
    
    fwd_queue = [(node, trigger_time, 0)]
    max_fwd_depth = 0
    while fwd_queue:
        curr, curr_time, depth = fwd_queue.pop(0)
        max_fwd_depth = max(max_fwd_depth, depth)
        
        # Check out edges within temporal window
        valid_out_edges = []
        for _, tgt, data in G.out_edges(curr, data=True):
            if data['txn_id'] not in visited_edges and curr_time <= data['timestamp'] <= curr_time + timedelta(hours=max_hours):
                valid_out_edges.append((tgt, data))
                
        if not valid_out_edges:
            fwd_sinks.add(curr)
        else:
            for tgt, data in valid_out_edges:
                visited_edges.add(data['txn_id'])
                visited_nodes.add(tgt)
                fwd_queue.append((tgt, data['timestamp'], depth + 1))
                    
    bwd_queue = [(node, trigger_time, 0)]
    max_bwd_depth = 0
    while bwd_queue:
        curr, curr_time, depth = bwd_queue.pop(0)
        max_bwd_depth = max(max_bwd_depth, depth)
        for src, _, data in G.in_edges(curr, data=True):
            if data['txn_id'] not in visited_edges:
                if curr_time - timedelta(hours=max_hours) <= data['timestamp'] <= curr_time:
                    visited_edges.add(data['txn_id'])
                    visited_nodes.add(src)
                    bwd_queue.append((src, data['timestamp'], depth + 1))
                    
    total_hops = max_fwd_depth + max_bwd_depth
    return total_hops, len(visited_nodes), fwd_sinks

def run_phase4():
    print("Running Phase 4 & Phase 8: Multi-hop Graph Tracing with Metadata...")
    det = MVPDetector('data/accounts.csv', 'data/transactions.csv')
    det.build_graph()
    det.score_accounts()
    df = det.df_results
    
    all_accounts = df['account_id'].unique()
    train_ids, test_ids = train_test_split(all_accounts, test_size=0.2, random_state=42)
    test_df = df[df['account_id'].isin(test_ids)].copy()
    
    threshold = 0.5
    y_true = test_df['is_mule']
    y_pred_mvp = test_df['risk_score'] > threshold
    
    base_p = precision_score(y_true, y_pred_mvp, zero_division=0)
    base_r = recall_score(y_true, y_pred_mvp, zero_division=0)
    base_f1 = f1_score(y_true, y_pred_mvp, zero_division=0)
    base_fp = confusion_matrix(y_true, y_pred_mvp).ravel()[1]
    
    print("\n--- BASELINE (MVP Phase 3) ---")
    print(f"Precision: {base_p:.3f} | Recall: {base_r:.3f} | F1: {base_f1:.3f}")
    print(f"False Positives: {base_fp}")
    
    print("\nTracing flagged accounts to reconstruct chains and applying MCC metadata...")
    new_scores = []
    
    for _, row in test_df.iterrows():
        node = row['account_id']
        score = row['risk_score']
        
        if score > threshold:
            total_hops, node_count, fwd_sinks = get_chain_metrics(det.G, node)
            
            # TRACER RULE (Topology): Massive Fan-in/Fan-out = Safe
            if node_count > 8:
                score = 0.0 
            else:
                # PHASE 8 METADATA RULE: Check the MCC of the terminal sinks
                # If the money safely lands in a registered trusted merchant, un-flag.
                for sink in fwd_sinks:
                    mcc = det.G.nodes[sink].get('mcc_code', 'NONE')
                    if mcc in ["HOSPITAL", "EDUCATION", "UTILITIES", "WHOLESALE"]:
                        score = 0.0
                        break
                
        new_scores.append(score)
        
    test_df['tracer_score'] = new_scores
    y_pred_trace = test_df['tracer_score'] > threshold
    
    trace_p = precision_score(y_true, y_pred_trace, zero_division=0)
    trace_r = recall_score(y_true, y_pred_trace, zero_division=0)
    trace_f1 = f1_score(y_true, y_pred_trace, zero_division=0)
    trace_fp = confusion_matrix(y_true, y_pred_trace).ravel()[1]
    
    print("\n--- AFTER GRAPH TRACING & METADATA (Phases 4 + 8) ---")
    print(f"Precision: {trace_p:.3f} | Recall: {trace_r:.3f} | F1: {trace_f1:.3f}")
    print(f"Precision Lift: +{(trace_p - base_p)*100:.1f}%")
    print(f"False Positives: {trace_fp} (Down from {base_fp})")

if __name__ == "__main__":
    run_phase4()
