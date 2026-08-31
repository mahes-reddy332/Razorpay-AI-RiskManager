"""
Neo4j Graph Database Prototype

This script demonstrates migrating the in-memory NetworkX graph to a distributed
Neo4j graph database. It ingests the CSV data and runs a Cypher query to find
mule rings based on rapid fan-out topologies.
"""

import os
import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "hackathon123"

ACCOUNTS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "accounts.csv")
TXNS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")


class Neo4jMuleTracer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def wipe_database(self):
        """Clear existing data to ensure a fresh prototype run."""
        print("Wiping existing Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ingest_data(self):
        """Read CSVs and ingest Nodes (Accounts) and Edges (Transactions)."""
        print("Loading CSVs...")
        df_acc = pd.read_csv(ACCOUNTS_CSV)
        df_txn = pd.read_csv(TXNS_CSV)
        
        # We process transactions first to find all unique accounts 
        # (including external banks not in accounts.csv)
        all_accounts = set(df_acc['account_id']).union(
            set(df_txn['source_account']), set(df_txn['target_account'])
        )

        acc_dict = df_acc.set_index('account_id').to_dict('index')

        print(f"Ingesting {len(all_accounts)} Account nodes in bulk...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
            
            nodes_data = [
                {"id": acc_id, 
                 "mcc": acc_dict.get(acc_id, {}).get('mcc_code', 'NONE'),
                 "is_mule": bool(acc_dict.get(acc_id, {}).get('is_mule', False))}
                for acc_id in all_accounts
            ]
            
            session.run(
                """
                UNWIND $nodes AS node
                MERGE (a:Account {id: node.id})
                SET a.mcc_code = node.mcc, a.is_mule = node.is_mule
                """,
                nodes=nodes_data
            )

        print(f"Ingesting {len(df_txn)} Transaction edges in bulk...")
        with self.driver.session() as session:
            edges_data = [
                {"src": row['source_account'], "tgt": row['target_account'],
                 "amount": row['amount'], "txn_id": row['txn_id'], "timestamp": str(row['timestamp'])}
                for _, row in df_txn.iterrows()
            ]
            
            # Neo4j handles 20k edges easily in one UNWIND
            session.run(
                """
                UNWIND $edges AS edge
                MATCH (src:Account {id: edge.src})
                MATCH (tgt:Account {id: edge.tgt})
                CREATE (src)-[:TRANSFERRED_TO {
                    amount: edge.amount, 
                    txn_id: edge.txn_id, 
                    timestamp: edge.timestamp
                }]->(tgt)
                """,
                edges=edges_data
            )

    def find_mule_fan_out(self):
        """
        Cypher query equivalent of our NetworkX graph trace.
        Finds accounts that receive money from multiple sources, 
        then quickly forward it to a high-risk sink.
        """
        print("\n" + "="*60)
        print("RUNNING CYPHER QUERY: Mule Fan-Out to Risky Sinks")
        print("="*60)
        
        query = """
        MATCH (src:Account)-[in:TRANSFERRED_TO]->(mule:Account)-[out:TRANSFERRED_TO]->(sink:Account)
        WHERE sink.mcc_code IN ['CRYPTO_EXCHANGE', 'GAMBLING', 'UNREGISTERED_P2P']
        AND in.amount > 1000
        WITH mule, sink, count(in) as inbound_count, sum(out.amount) as total_forwarded
        WHERE inbound_count >= 2
        RETURN mule.id AS MuleAccount, sink.mcc_code AS SinkType, inbound_count, total_forwarded
        ORDER BY total_forwarded DESC
        LIMIT 5
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            if not records:
                print("No active mule rings found matching pattern.")
            else:
                for record in records:
                    print(f"Mule: {record['MuleAccount']:<15} | Sent to: {record['SinkType']:<18} "
                          f"| Inbound Links: {record['inbound_count']} "
                          f"| Forwarded Vol: {record['total_forwarded']}")


if __name__ == "__main__":
    tracer = Neo4jMuleTracer(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    try:
        tracer.wipe_database()
        tracer.ingest_data()
        tracer.find_mule_fan_out()
    except Exception as e:
        print(f"Neo4j Error: {e}")
        print("Make sure Docker Neo4j is running: docker-compose up -d")
    finally:
        tracer.close()
