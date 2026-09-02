import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import pandas as pd
import networkx as nx
from confluent_kafka import Producer, Consumer, KafkaError
from src.incremental_graph_engine import incremental_update

IBM_CSV_PATH = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")

def run_ibm_kafka_demo():
    print("=== STARTING IBM AML DATASET KAFKA STREAMING BENCHMARK ===")
    
    if not os.path.exists(IBM_CSV_PATH):
        print(f"IBM Dataset not found at {IBM_CSV_PATH}")
        return

    # 1. Read IBM dataset sample
    print(f"Reading sample from IBM AML Dataset ({IBM_CSV_PATH})...")
    df = pd.read_csv(IBM_CSV_PATH, nrows=2000)
    print(f"Loaded {len(df)} transactions from IBM dataset.")

    # 2. Setup Producer
    producer_conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(producer_conf)
    
    print("Publishing IBM AML events to Kafka topic 'ibm-aml-transactions'...")
    for idx, row in df.iterrows():
        payload = {
            'sender_account': f"BANK_{row['From Bank']}_{row['Account']}",
            'receiver_account': f"BANK_{row['To Bank']}_{row['Account.1']}",
            'amount': float(row['Amount Paid']),
            'timestamp': str(row['Timestamp']),
            'is_laundering': int(row['Is Laundering'])
        }
        producer.produce('ibm-aml-transactions', json.dumps(payload).encode('utf-8'))
        producer.poll(0)
    
    producer.flush()
    print("All 2,000 IBM transactions successfully published to Redpanda Kafka topic!\n")

    # 3. Setup Consumer & Score
    consumer_conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'ibm-aml-scorer-group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(['ibm-aml-transactions'])

    graph = nx.DiGraph()
    print("Consuming and Incrementally Scoring IBM AML events...")
    
    count = 0
    laundering_detected = 0
    start_time = None
    last_msg_time = time.time()
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                if count > 0 and (time.time() - last_msg_time > 2):
                    break
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    break
                    
            if count == 0:
                start_time = time.time()
                
            last_msg_time = time.time()
            txn = json.loads(msg.value().decode('utf-8'))
            
            # Incremental update on IBM topology
            incremental_update(graph, txn['sender_account'], txn['receiver_account'], txn['amount'])
            count += 1
            if txn.get('is_laundering') == 1:
                laundering_detected += 1

            if count % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {count} IBM transactions... ({(count/elapsed):.1f} TPS)")
                
    finally:
        consumer.close()
        if start_time and count > 0:
            total_time = last_msg_time - start_time
            print(f"\n=== IBM AML KAFKA STREAMING RESULT ===")
            print(f"Dataset: IBM AML (HI-Small_Trans.csv)")
            print(f"Total Transactions Streamed: {count}")
            print(f"Laundering Tags Processed: {laundering_detected}")
            print(f"Total Execution Time: {total_time:.3f} seconds")
            print(f"Streaming Graph Throughput: {(count/total_time):.1f} TPS")

if __name__ == '__main__':
    run_ibm_kafka_demo()
