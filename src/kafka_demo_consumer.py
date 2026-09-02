import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import time
import networkx as nx
from confluent_kafka import Consumer, KafkaError
from src.incremental_graph_engine import incremental_update

def main():
    print("Starting Kafka Consumer (confluent_kafka)...")
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'incremental-scorer-group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe(['upi-transactions'])
    
    # Initialize baseline graph
    graph = nx.DiGraph()
    print("Consumer subscribed to 'upi-transactions'. Waiting for incoming Kafka messages...")
    
    count = 0
    start_time = None
    last_msg_time = time.time()
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                if count > 0 and (time.time() - last_msg_time > 3):
                    print(f"\nStream paused. Total transactions processed via Kafka: {count}")
                    break
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break
                    
            if count == 0:
                start_time = time.time()
                
            last_msg_time = time.time()
            txn = json.loads(msg.value().decode('utf-8'))
            
            sender = txn.get('sender_account', 'ACC_DEFAULT_SENDER')
            receiver = txn.get('receiver_account', 'ACC_DEFAULT_RECV')
            amount = float(txn.get('amount', 100))
            
            # CALL THE EXISTING INCREMENTAL GRAPH ENGINE
            incremental_update(graph, sender, receiver, amount)
            count += 1
            
            if count % 200 == 0:
                elapsed = time.time() - start_time
                tps = count / elapsed if elapsed > 0 else 0
                print(f"Consumed & Incremental Scored {count} transactions | Velocity: {tps:.1f} TPS")
                
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        if start_time and count > 0:
            total_time = last_msg_time - start_time
            print(f"\n=== REAL KAFKA STREAMING BENCHMARK ===")
            print(f"Total Events Consumed from Redpanda: {count}")
            print(f"Total Execution Time: {total_time:.3f} seconds")
            print(f"End-to-End Event Ingestion Velocity: {(count/total_time):.1f} TPS")

if __name__ == '__main__':
    main()
