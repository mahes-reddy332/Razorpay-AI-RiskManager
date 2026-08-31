import pandas as pd
import networkx as nx
import numpy as np
from datetime import timedelta
import json
import os
import sys

# Append local path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracer import get_chain_metrics

class FraudRiskAuditor:
    def __init__(self, accounts_path, txns_path):
        self.df_acc = pd.read_csv(accounts_path)
        self.df_txn = pd.read_csv(txns_path)
        
        self.df_acc['creation_date'] = pd.to_datetime(self.df_acc['creation_date'])
        self.df_txn['timestamp'] = pd.to_datetime(self.df_txn['timestamp'])
        
        self.G = nx.MultiDiGraph()
        self.audit_log = []
        
    def inject_failure_cases(self):
        """Deliberately injects broken data to test graceful degradation."""
        print("Injecting deliberate failure cases for Phase 5 demo...")
        
        # Case 1: Missing Creation Date (NaT)
        self.df_acc.loc[len(self.df_acc)] = {
            'account_id': 'ACC_FAIL_NODATE', 'creation_date': pd.NaT, 'device_id': 'DEV_000',
            'activity_level': 'low', 'kyc_tier': 'tier1', 'is_mule': False, 'behavior_type': 'normal'
        }
        self.df_txn.loc[len(self.df_txn)] = {
            'txn_id': 'TXN_F1', 'timestamp': pd.Timestamp('2026-06-15 10:00:00'),
            'source_account': 'EXT_BANK_1', 'target_account': 'ACC_FAIL_NODATE', 'amount': 50000,
            'is_mule_chain': False, 'chain_id': 'CHAIN_F1'
        }
        self.df_txn.loc[len(self.df_txn)] = {
            'txn_id': 'TXN_F2', 'timestamp': pd.Timestamp('2026-06-15 11:00:00'),
            'source_account': 'ACC_FAIL_NODATE', 'target_account': 'EXT_BANK_2', 'amount': 49000,
            'is_mule_chain': False, 'chain_id': 'CHAIN_F1'
        }
        
        # Case 2: Missing Transaction Timestamp (NaT)
        self.df_acc.loc[len(self.df_acc)] = {
            'account_id': 'ACC_FAIL_NOTIME', 'creation_date': pd.Timestamp('2026-01-01'), 'device_id': 'DEV_001',
            'activity_level': 'low', 'kyc_tier': 'tier1', 'is_mule': False, 'behavior_type': 'normal'
        }
        self.df_txn.loc[len(self.df_txn)] = {
            'txn_id': 'TXN_F3', 'timestamp': pd.NaT, # BROKEN DATA
            'source_account': 'EXT_BANK_1', 'target_account': 'ACC_FAIL_NOTIME', 'amount': 50000,
            'is_mule_chain': False, 'chain_id': 'CHAIN_F2'
        }
        
    def build_graph(self):
        accounts_dict = self.df_acc.set_index('account_id').to_dict('index')
        for acc_id, attrs in accounts_dict.items():
            self.G.add_node(acc_id, creation_date=attrs['creation_date'], is_mule=attrs['is_mule'])
            
        edges = []
        for _, row in self.df_txn.iterrows():
            src, tgt = row['source_account'], row['target_account']
            if src not in self.G: self.G.add_node(src, creation_date=pd.Timestamp.min, is_mule=False)
            if tgt not in self.G: self.G.add_node(tgt, creation_date=pd.Timestamp.min, is_mule=False)
                
            edges.append((src, tgt, {
                'amount': row['amount'], 'timestamp': row['timestamp'], 'txn_id': row['txn_id']
            }))
        self.G.add_edges_from(edges)
        
    def audit_account(self, node):
        record = {
            "account_id": node,
            "status": "OK",
            "score": 0.0,
            "decision": "PASS",
            "explanation": "Normal transaction activity.",
            "signals": {}
        }
        
        try:
            # 1. DATA INTEGRITY CHECK (Graceful Degradation)
            creation_date = self.G.nodes[node].get('creation_date')
            if pd.isnull(creation_date):
                raise ValueError("Missing account creation date.")
                
            in_txns = self.G.in_edges(node, data=True)
            out_txns = self.G.out_edges(node, data=True)
            
            in_txns_sorted = []
            for _, _, d in in_txns:
                if pd.isnull(d.get('timestamp')) or pd.isnull(d.get('amount')):
                    raise ValueError(f"Corrupt transaction record found: {d.get('txn_id')}")
                in_txns_sorted.append(d)
                
            out_txns_sorted = []
            for _, _, d in out_txns:
                if pd.isnull(d.get('timestamp')) or pd.isnull(d.get('amount')):
                    raise ValueError(f"Corrupt transaction record found: {d.get('txn_id')}")
                out_txns_sorted.append(d)
                    
            if not in_txns_sorted or not out_txns_sorted:
                return record # No complete flow
                
            in_txns_sorted.sort(key=lambda x: x['timestamp'])
            out_txns_sorted.sort(key=lambda x: x['timestamp'])
            
            # 2. CORE MVP SIGNALS
            first_txn_time = in_txns_sorted[0]['timestamp']
            dormancy_days = (first_txn_time - creation_date).days
            record["signals"]["dormancy_days"] = dormancy_days
            is_dormant = dormancy_days > 30
            
            pass_through = False
            pt_ratio = 0.0
            trigger_amt = 0.0
            
            for in_txn in in_txns_sorted:
                in_amt, in_time = in_txn['amount'], in_txn['timestamp']
                if in_amt < 1000: continue
                
                window_out = sum(out_txn['amount'] for out_txn in out_txns_sorted 
                                 if in_time <= out_txn['timestamp'] <= in_time + timedelta(hours=24))
                
                ratio = window_out / in_amt
                if 0.90 <= ratio <= 1.05:
                    pass_through = True
                    pt_ratio = ratio
                    trigger_amt = in_amt
                    break
                    
            record["signals"]["pass_through_ratio"] = round(pt_ratio, 2)
            
            # TIER 0 BASE SCORE
            base_score = 0.0
            if pass_through: base_score += 0.6
            if is_dormant: base_score += 0.4
            record["score"] = base_score
            
            # --- TIER 1 ADVANCED FEATURE EXTRACTION & SCORING ---
            from model_v2 import FROZEN_CONFIG
            
            # 1. Always-on immediate MCC check (Tier 0 integration)
            is_risky = 0
            for succ in self.G.successors(node):
                mcc = self.G.nodes[succ].get('mcc_code', 'NONE')
                if mcc in ["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"]:
                    is_risky = 1
                    break
            
            final_score = base_score
            final_score += is_risky * FROZEN_CONFIG['mcc_weight']
            
            # 2. Expensive Graph Trace (Tier 1) - only if velocity tripwire fired
            if base_score > 0:
                hops, node_count, sinks = get_chain_metrics(self.G, node)
                record["signals"]["chain_nodes"] = node_count
                record["signals"]["chain_hops"] = hops
                
                if node_count > 0 and node_count <= FROZEN_CONFIG['node_thresh']:
                    final_score += FROZEN_CONFIG['top_weight']
                    
                # We also check deep sinks in Tier 1
                for sink in sinks:
                    mcc = self.G.nodes[sink].get('mcc_code', 'NONE')
                    if mcc in ["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"]:
                        if is_risky == 0:  # Only add if it wasn't already caught by immediate check
                            final_score += FROZEN_CONFIG['mcc_weight']
                            is_risky = 1
                        break
            
            record["score"] = final_score
            
            # 4. FINAL FLAG DECISION BANDS
            B_FLAG = FROZEN_CONFIG['dec_thresh']     # Default: 0.5
            A_REV  = FROZEN_CONFIG['manual_thresh']  # Default: 0.3
            
            exp = f"{pt_ratio*100:.0f}% of inflow ({trigger_amt} INR) was forwarded within 24 hours. "
            if is_dormant:
                exp += f"Account was dormant for {dormancy_days} days. "
            if is_risky:
                exp += f"Routed to risky terminal MCC. "
                
            if final_score >= B_FLAG:
                record["decision"] = "HIGH_RISK"
                record["explanation"] = "Flagged: " + exp + f"(Composite Score: {final_score:.2f} >= {B_FLAG})"
            elif final_score >= A_REV:
                record["decision"] = "MANUAL_REVIEW_REQUIRED"
                record["explanation"] = "Borderline: " + exp + f"(Composite Score: {final_score:.2f}). Requires manual review."
            else:
                record["decision"] = "SAFE"
                record["explanation"] = f"Cleared: Activity did not exceed safety thresholds (Composite Score: {final_score:.2f})."
            
        except ValueError as e:
            record["status"] = "DEGRADED"
            record["score"] = -1.0
            record["decision"] = "MANUAL_REVIEW_REQUIRED"
            record["explanation"] = f"INSUFFICIENT DATA: {str(e)} System bypassed automated scoring to prevent silent misclassification."
            record["signals"]["error"] = str(e)
            
        return record

    def generate_audit_log(self):
        print("Generating audit log for all accounts...")
        for node in self.G.nodes():
            if not str(node).startswith("ACC_"): continue
            audit_record = self.audit_account(node)
            self.audit_log.append(audit_record)
            
        os.makedirs('outputs', exist_ok=True)
        with open('outputs/audit_log.json', 'w') as f:
            json.dump(self.audit_log, f, indent=4)
        print("Saved detailed audit log to outputs/audit_log.json")

    def demo_cases(self):
        print("\n--- PHASE 5 DEMO ---")
        
        failed_date = next((r for r in self.audit_log if r["account_id"] == "ACC_FAIL_NODATE"), None)
        failed_time = next((r for r in self.audit_log if r["account_id"] == "ACC_FAIL_NOTIME"), None)
        
        if failed_date:
            print("\n1. ENGINEERED FAILURE 1: Missing Account Creation Date")
            print(f"Account: {failed_date['account_id']}")
            print(f"Decision: {failed_date['decision']} (Score: {failed_date['score']})")
            print(f"Explanation: {failed_date['explanation']}")
        
        if failed_time:
            print("\n2. ENGINEERED FAILURE 2: Corrupted Transaction Timestamp")
            print(f"Account: {failed_time['account_id']}")
            print(f"Decision: {failed_time['decision']} (Score: {failed_time['score']})")
            print(f"Explanation: {failed_time['explanation']}")


if __name__ == "__main__":
    auditor = FraudRiskAuditor('data/accounts.csv', 'data/transactions.csv')
    auditor.inject_failure_cases()
    auditor.build_graph()
    auditor.generate_audit_log()
    auditor.demo_cases()
