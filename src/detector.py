import pandas as pd
import networkx as nx
import numpy as np
from datetime import timedelta

class MVPDetector:
    def __init__(self, accounts_path, txns_path):
        print("Loading data...")
        self.df_acc = pd.read_csv(accounts_path)
        self.df_txn = pd.read_csv(txns_path)
        
        self.df_acc['creation_date'] = pd.to_datetime(self.df_acc['creation_date'])
        self.df_txn['timestamp'] = pd.to_datetime(self.df_txn['timestamp'])
        
        self.accounts_dict = self.df_acc.set_index('account_id').to_dict('index')
        self.G = nx.MultiDiGraph()
        
    def build_graph(self):
        print("Building transaction graph...")
        # Add nodes with creation date
        for acc_id, attrs in self.accounts_dict.items():
            self.G.add_node(acc_id, creation_date=attrs['creation_date'], is_mule=attrs['is_mule'], mcc_code=attrs.get('mcc_code', 'NONE'))
            
        # Add edges and any missing external nodes
        edges = []
        for _, row in self.df_txn.iterrows():
            src = row['source_account']
            tgt = row['target_account']
            
            if src not in self.G:
                self.G.add_node(src, creation_date=pd.Timestamp.min, is_mule=False, mcc_code='NONE')
            if tgt not in self.G:
                self.G.add_node(tgt, creation_date=pd.Timestamp.min, is_mule=False, mcc_code='NONE')
                
            edges.append((src, tgt, {
                'amount': row['amount'],
                'timestamp': row['timestamp'],
                'txn_id': row['txn_id']
            }))
        self.G.add_edges_from(edges)
        print(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        
    def score_accounts(self):
        print("Scoring accounts (MVP Rule Engine)...")
        results = []
        
        for node in self.G.nodes():
            in_edges = self.G.in_edges(node, data=True)
            out_edges = self.G.out_edges(node, data=True)
            
            if not in_edges or not out_edges:
                continue # Ignore accounts that don't both receive and send
                
            in_txns = sorted([d for _, _, d in in_edges], key=lambda x: x['timestamp'])
            out_txns = sorted([d for _, _, d in out_edges], key=lambda x: x['timestamp'])
            
            # --- Rule 1: Pass-through (Velocity + Ratio) ---
            pass_through_score = 0.0
            # Look for any significant inbound transfer that is forwarded within 24 hours
            for in_txn in in_txns:
                in_amt = in_txn['amount']
                in_time = in_txn['timestamp']
                
                if in_amt < 1000: continue # Ignore tiny txns for pass-through analysis
                
                # Sum outbounds within 24h
                window_out = sum(out_txn['amount'] for out_txn in out_txns 
                                 if in_time <= out_txn['timestamp'] <= in_time + timedelta(hours=24))
                
                ratio = window_out / in_amt
                if 0.90 <= ratio <= 1.05:
                    pass_through_score = 0.6
                    break # Found a strong pass-through pattern
                    
            # --- Rule 2: Dormancy ---
            dormancy_score = 0.0
            creation_date = self.G.nodes[node]['creation_date']
            first_txn_time = in_txns[0]['timestamp'] if in_txns else None
            
            if first_txn_time and (first_txn_time - creation_date).days > 30:
                dormancy_score = 0.4
                
            # Final 0-1 Score
            total_score = pass_through_score
            # Only apply dormancy penalty if it also exhibits pass-through (or else dormant legit users get flagged)
            if pass_through_score > 0 and dormancy_score > 0:
                total_score += dormancy_score
                
            results.append({
                'account_id': node,
                'is_mule': self.G.nodes[node]['is_mule'],
                'pass_through_flag': pass_through_score > 0,
                'dormancy_flag': dormancy_score > 0,
                'risk_score': total_score
            })
            
        self.df_results = pd.DataFrame(results)
        
    def evaluate_informally(self):
        threshold = 0.5 # Anyone with score > 0.5 is flagged
        self.df_results['flagged'] = self.df_results['risk_score'] > threshold
        
        tp = len(self.df_results[(self.df_results['flagged'] == True) & (self.df_results['is_mule'] == True)])
        fp = len(self.df_results[(self.df_results['flagged'] == True) & (self.df_results['is_mule'] == False)])
        fn = len(self.df_results[(self.df_results['flagged'] == False) & (self.df_results['is_mule'] == True)])
        tn = len(self.df_results[(self.df_results['flagged'] == False) & (self.df_results['is_mule'] == False)])
        
        print("\n--- INFORMAL MVP RESULTS ---")
        print(f"True Positives (Mules Caught): {tp}")
        print(f"False Positives (Legit Blocked): {fp}")
        print(f"False Negatives (Mules Missed): {fn}")
        
        print("\nSample of Flagged Accounts (True Positives):")
        print(self.df_results[(self.df_results['flagged']) & (self.df_results['is_mule'])].head(5).to_string(index=False))
        
        print("\nSample of Flagged Accounts (False Positives):")
        print(self.df_results[(self.df_results['flagged']) & (~self.df_results['is_mule'])].head(5).to_string(index=False))

if __name__ == "__main__":
    detector = MVPDetector('data/accounts.csv', 'data/transactions.csv')
    detector.build_graph()
    detector.score_accounts()
    detector.evaluate_informally()
