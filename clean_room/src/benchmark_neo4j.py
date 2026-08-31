"""
Neo4j Graph Database Prototype & Benchmark

This script demonstrates migrating the in-memory NetworkX graph to a distributed
Neo4j graph database. It ingests the CSV data, runs a corrected Cypher query 
to find mule rings without cartesian explosion, and benchmarks it against NetworkX.
"""

import os
import time
import statistics
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "hackathon123"

ACCOUNTS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "accounts.csv")
TXNS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")


class Neo4jMuleTracer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        print("Loading CSVs into memory for NetworkX benchmark...")
        self.df_acc = pd.read_csv(ACCOUNTS_CSV)
        self.df_txn = pd.read_csv(TXNS_CSV)
        self.accounts_dict = self.df_acc.set_index('account_id').to_dict('index')

    def close(self):
        self.driver.close()

    def find_mule_fan_out_cypher(self):
        """
        Corrected Cypher query. Separates inbound and outbound aggregations 
        to prevent cartesian cross-multiplication of counts.
        """
        query = """
        MATCH (mule:Account)-[out:TRANSFERRED_TO]->(sink:Account)
        WHERE sink.mcc_code IN ['CRYPTO_EXCHANGE', 'GAMBLING', 'UNREGISTERED_P2P']
        WITH mule, sink, sum(out.amount) AS total_forwarded
        MATCH (src:Account)-[in:TRANSFERRED_TO]->(mule)
        WHERE in.amount > 1000
        WITH mule, sink, total_forwarded, count(in) AS inbound_count
        WHERE inbound_count >= 2
        RETURN mule.id AS MuleAccount, sink.mcc_code AS SinkType, inbound_count, total_forwarded
        ORDER BY total_forwarded DESC LIMIT 5
        """
        
        times = []
        records = []
        with self.driver.session() as session:
            # Run 3 times for median benchmark
            for _ in range(3):
                start = time.perf_counter()
                result = session.run(query)
                records = list(result) # consume cursor
                times.append(time.perf_counter() - start)
                
        median_time = statistics.median(times)
        
        print("\n" + "="*60)
        print(f"CYPHER QUERY RESULTS (Median Time: {median_time*1000:.2f} ms)")
        print("="*60)
        for record in records:
            print(f"Mule: {record['MuleAccount']:<15} | Sent to: {record['SinkType']:<18} "
                  f"| Inbound Links: {record['inbound_count']} "
                  f"| Forwarded Vol: {record['total_forwarded']:.2f}")
                  
        return median_time

    def find_mule_fan_out_networkx(self):
        """
        Equivalent logic in Python/NetworkX for benchmarking.
        """
        print("\nBuilding NetworkX graph...")
        G = nx.MultiDiGraph()
        
        # Add nodes
        all_accounts = set(self.df_acc['account_id']).union(
            set(self.df_txn['source_account']), set(self.df_txn['target_account'])
        )
        for acc_id in all_accounts:
            mcc = self.accounts_dict.get(acc_id, {}).get('mcc_code', 'NONE')
            G.add_node(acc_id, mcc_code=mcc)
            
        # Add edges
        edges = []
        for _, row in self.df_txn.iterrows():
            edges.append((row['source_account'], row['target_account'], {'amount': row['amount']}))
        G.add_edges_from(edges)
        
        print("Running NetworkX equivalent query 3 times...")
        risky_sinks = {'CRYPTO_EXCHANGE', 'GAMBLING', 'UNREGISTERED_P2P'}
        times = []
        
        for _ in range(3):
            start = time.perf_counter()
            results = []
            
            for node in G.nodes():
                # Check outbound to risky sinks
                out_edges = G.out_edges(node, data=True)
                sink_totals = {}
                for u, v, d in out_edges:
                    sink_mcc = G.nodes[v].get('mcc_code', 'NONE')
                    if sink_mcc in risky_sinks:
                        sink_totals[v] = sink_totals.get(v, 0) + d['amount']
                
                if not sink_totals:
                    continue
                    
                # Check inbound > 1000
                in_edges = G.in_edges(node, data=True)
                inbound_count = sum(1 for u, v, d in in_edges if d['amount'] > 1000)
                
                if inbound_count >= 2:
                    for sink_node, total_fwd in sink_totals.items():
                        sink_mcc = G.nodes[sink_node].get('mcc_code', 'NONE')
                        results.append({
                            'MuleAccount': node,
                            'SinkType': sink_mcc,
                            'inbound_count': inbound_count,
                            'total_forwarded': total_fwd
                        })
            
            # Sort and limit 5
            results.sort(key=lambda x: x['total_forwarded'], reverse=True)
            top_5 = results[:5]
            times.append(time.perf_counter() - start)
            
        median_time = statistics.median(times)
        print("\n" + "="*60)
        print(f"NETWORKX QUERY RESULTS (Median Time: {median_time*1000:.2f} ms)")
        print("="*60)
        for r in top_5:
             print(f"Mule: {r['MuleAccount']:<15} | Sent to: {r['SinkType']:<18} "
                  f"| Inbound Links: {r['inbound_count']} "
                  f"| Forwarded Vol: {r['total_forwarded']:.2f}")
                  
        return median_time

if __name__ == "__main__":
    tracer = Neo4jMuleTracer(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    try:
        # Note: We assume ingest_data() ran previously. 
        # If not, the graph is already stored in Neo4j from the last script run.
        tracer.find_mule_fan_out_cypher()
        tracer.find_mule_fan_out_networkx()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracer.close()
