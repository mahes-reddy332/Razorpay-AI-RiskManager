import time
import json
import networkx as nx
from confluent_kafka import Producer, Consumer
from src.incremental_graph_engine import incremental_update

def test_single_message_latency():
    conf_p = {'bootstrap.servers': 'localhost:9092'}
    conf_c = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'single-msg-test-group',
        'auto.offset.reset': 'latest'
    }
    
    producer = Producer(conf_p)
    consumer = Consumer(conf_c)
    consumer.subscribe(['single-msg-topic'])
    
    # Warm up consumer connection
    consumer.poll(timeout=1.0)
    
    graph = nx.DiGraph()
    test_event = {
        'sender_account': 'ACC_TEST_SENDER',
        'receiver_account': 'ACC_TEST_RECV',
        'amount': 500.0
    }
    
    # Measure exact wall-clock start
    t0 = time.perf_counter()
    
    # Send & flush producer
    producer.produce('single-msg-topic', json.dumps(test_event).encode('utf-8'))
    producer.flush()
    
    # Poll consumer until received
    while True:
        msg = consumer.poll(timeout=0.1)
        if msg is not None and not msg.error():
            t1 = time.perf_counter()
            data = json.loads(msg.value().decode('utf-8'))
            incremental_update(graph, data['sender_account'], data['receiver_account'], data['amount'])
            t2 = time.perf_counter()
            break
            
    consumer.close()
    
    network_broker_lat = (t1 - t0) * 1000  # ms
    scoring_lat = (t2 - t1) * 1000          # ms
    total_lat = (t2 - t0) * 1000            # ms
    
    print("=== SINGLE MESSAGE ROUND-TRIP SANITY TEST ===")
    print(f"Network + Redpanda Broker Latency: {network_broker_lat:.3f} ms")
    print(f"Incremental Graph Scoring Latency: {scoring_lat:.3f} ms")
    print(f"Total Wall-Clock Round-Trip Latency: {total_lat:.3f} ms")

if __name__ == '__main__':
    test_single_message_latency()
