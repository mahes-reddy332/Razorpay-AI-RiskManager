import time
import json
import threading
import sys
import networkx as nx
from confluent_kafka import Producer, Consumer
from src.incremental_graph_engine import incremental_update

def run_honest_single_msg_test():
    topic = 'honest-single-msg-topic'
    conf_p = {'bootstrap.servers': 'localhost:9092'}
    conf_c = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'honest-test-group-1',
        'auto.offset.reset': 'latest'
    }

    producer = Producer(conf_p)
    consumer = Consumer(conf_c)
    consumer.subscribe([topic])

    # 1. Warmup: Poll until consumer group assignment is ACTIVE and settled
    print("Warming up consumer group assignment...")
    ready = False
    for _ in range(50):
        msg = consumer.poll(timeout=0.1)
        # Produce a dummy message to force topic creation and partition assignment
        producer.produce(topic, b'dummy')
        producer.flush()
        if consumer.assignment():
            ready = True
            break
        time.sleep(0.1)

    print("Consumer ready and assigned to partitions!")
    # Clear out any dummy messages
    while consumer.poll(timeout=0.2) is not None:
        pass

    # 2. Perform Single Message Latency Test
    graph = nx.DiGraph()
    test_event = {
        'sender_account': 'ACC_HONEST_SENDER',
        'receiver_account': 'ACC_HONEST_RECV',
        'amount': 750.0
    }

    # Start timer RIGHT BEFORE produce
    t_send = time.perf_counter()
    producer.produce(topic, json.dumps(test_event).encode('utf-8'))
    producer.flush()  # Ensures message is confirmed written to broker socket

    # Poll until consumer gets the real message
    received_time = None
    scored_time = None

    while True:
        msg = consumer.poll(timeout=0.05)
        if msg is not None and not msg.error():
            val = msg.value().decode('utf-8')
            if 'ACC_HONEST_SENDER' in val:
                received_time = time.perf_counter()
                txn = json.loads(val)
                incremental_update(graph, txn['sender_account'], txn['receiver_account'], txn['amount'])
                scored_time = time.perf_counter()
                break

    consumer.close()

    net_broker_lat = (received_time - t_send) * 1000
    scoring_lat = (scored_time - received_time) * 1000
    total_roundtrip = (scored_time - t_send) * 1000

    print("\n=== HONEST SINGLE MESSAGE LATENCY TEST ===")
    print(f"Producer Send + Flush -> Broker -> Consumer Poll Latency: {net_broker_lat:.3f} ms")
    print(f"Incremental Scoring Execution Latency:                    {scoring_lat:.3f} ms")
    print(f"Total Real Wall-Clock Round-Trip:                          {total_roundtrip:.3f} ms")

if __name__ == '__main__':
    run_honest_single_msg_test()
