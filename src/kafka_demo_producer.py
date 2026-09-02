import json
import time
from datetime import datetime
from kafka import KafkaProducer

def get_producer():
    return KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def run_demo():
    producer = get_producer()
    topic = 'upi-transactions'
    
    print("=== STARTING TIER 0 EVENT-DRIVEN INGESTION DEMO ===")
    
    # Sequence of transactions for the same account
    transactions = [
        {"txn_id": "T1", "sender": "ACC_DEMO_99", "receiver": "ACC_LEGIT_1", "amount": 500, "format": "UPI", "timestamp": datetime.utcnow().isoformat()},
        {"txn_id": "T2", "sender": "ACC_DEMO_99", "receiver": "ACC_LEGIT_2", "amount": 1200, "format": "UPI", "timestamp": datetime.utcnow().isoformat()},
        {"txn_id": "T3", "sender": "ACC_DEMO_99", "receiver": "ACC_LEGIT_3", "amount": 800, "format": "UPI", "timestamp": datetime.utcnow().isoformat()},
        # The spike that pushes it over the threshold (e.g. > 50,000 velocity or fast succession)
        {"txn_id": "T4", "sender": "ACC_DEMO_99", "receiver": "ACC_SHADY_X", "amount": 95000, "format": "UPI", "timestamp": datetime.utcnow().isoformat()},
    ]
    
    for txn in transactions:
        print(f"[{datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]}] PRODUCER: Sending TXN {txn['txn_id']} | Amount: {txn['amount']} | Sender: {txn['sender']}")
        producer.send(topic, txn)
        producer.flush()
        # Sleep to simulate real timing, not an instant batch dump
        time.sleep(2)
        
    print("=== PRODUCER DEMO COMPLETE ===")

if __name__ == "__main__":
    run_demo()
