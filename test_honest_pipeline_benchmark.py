import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import threading
import pandas as pd
import networkx as nx
from confluent_kafka import Producer, Consumer, KafkaError
from src.incremental_graph_engine import incremental_update

IBM_CSV_PATH = r'C:\Users\PC-ACER\.cache\kagglehub\datasets\ealtman2019\ibm-transactions-for-anti-money-laundering-aml\versions\8\HI-Small_Trans.csv'

def run_honest_streaming_benchmark(dataset_name, dataset_type, num_records):
    topic = f"honest-bench-{int(time.time()*1000)}"
    conf_p = {'bootstrap.servers': 'localhost:9092'}
    conf_c = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': f'honest-group-{int(time.time()*1000)}',
        'auto.offset.reset': 'earliest'
    }

    producer = Producer(conf_p)
    consumer = Consumer(conf_c)
    consumer.subscribe([topic])

    # 1. Warm up Consumer connection
    print(f"\n=======================================================")
    print(f"BENCHMARK: {dataset_name} ({num_records} records)")
    print("=======================================================")
    print("Warming up consumer connection...")
    for _ in range(30):
        producer.produce(topic, b'warmup')
        producer.flush()
        msg = consumer.poll(timeout=0.1)
        if consumer.assignment():
            break
        time.sleep(0.05)

    # Drain warmup messages
    while consumer.poll(timeout=0.1) is not None:
        pass

    # Load dataset
    if dataset_type == 'upi':
        df = pd.read_csv('data/transactions.csv').head(num_records)
        records = []
        for _, r in df.iterrows():
            records.append({
                'sender': str(r['source_account']),
                'receiver': str(r['target_account']),
                'amount': float(r['amount'])
            })
    else:
        df = pd.read_csv(IBM_CSV_PATH, nrows=num_records)
        records = []
        for _, r in df.iterrows():
            records.append({
                'sender': f"BANK_{r['From Bank']}_{r['Account']}",
                'receiver': f"BANK_{r['To Bank']}_{r['Account.1']}",
                'amount': float(r['Amount Paid'])
            })

    graph = nx.DiGraph()
    consumed_count = [0]
    all_done = threading.Event()

    def consumer_thread_func():
        while not all_done.is_set() or consumed_count[0] < num_records:
            msg = consumer.poll(timeout=0.05)
            if msg is None or msg.error():
                continue
            val = msg.value().decode('utf-8')
            if 'sender' not in val:
                continue
            txn = json.loads(val)
            incremental_update(graph, txn['sender'], txn['receiver'], txn['amount'])
            consumed_count[0] += 1

    # Start consumer in thread
    c_thread = threading.Thread(target=consumer_thread_func)
    c_thread.start()

    # --- TRUE END-TO-END TIMING ---
    # Clock starts BEFORE Producer produces message 1
    t_start = time.perf_counter()

    # Producer streams records
    for rec in records:
        producer.produce(topic, json.dumps(rec).encode('utf-8'))
        producer.poll(0)

    # Flush all messages out of producer buffer to broker
    producer.flush()

    # Signal producer finished
    all_done.set()

    # Wait for consumer thread to finish consuming & scoring all records
    c_thread.join(timeout=15.0)

    # Clock stops ONLY after all messages are consumed and scored
    t_end = time.perf_counter()
    consumer.close()

    total_wall_time = t_end - t_start
    real_tps = consumed_count[0] / total_wall_time if total_wall_time > 0 else 0
    avg_lat_per_event = (total_wall_time / consumed_count[0]) * 1000 if consumed_count[0] > 0 else 0

    print(f"\n--- REAL END-TO-END STREAMING BENCHMARK RESULTS ---")
    print(f"Dataset:                             {dataset_name}")
    print(f"Total Events Produced & Consumed:   {consumed_count[0]}")
    print(f"Total End-to-End Wall-Clock Time:    {total_wall_time:.4f} seconds")
    print(f"Average Pipeline Latency per Event:  {avg_lat_per_event:.3f} ms")
    print(f"REAL Dynamic Throughput:             {real_tps:.1f} TPS")
    return real_tps, total_wall_time, avg_lat_per_event

if __name__ == '__main__':
    run_honest_streaming_benchmark("Synthetic UPI Dataset", "upi", 1000)
    run_honest_streaming_benchmark("IBM AML Dataset (HI-Small)", "ibm", 2000)
