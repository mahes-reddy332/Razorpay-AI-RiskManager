import time
import json
import sys
import pandas as pd
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

def main():
    print("Starting Kafka Producer (confluent_kafka)...")
    conf = {'bootstrap.servers': 'localhost:9092'}
    
    try:
        producer = Producer(conf)
    except Exception as e:
        print(f"Failed to create producer: {e}")
        sys.exit(1)

    print("Loading data/transactions.csv...")
    df = pd.read_csv('data/transactions.csv')
    subset = df.head(1000)
    
    print(f"Publishing {len(subset)} transactions to 'upi-transactions' topic...")
    start_time = time.time()
    
    for idx, row in subset.iterrows():
        payload = row.to_dict()
        producer.produce('upi-transactions', json.dumps(payload).encode('utf-8'), callback=delivery_report)
        producer.poll(0)
        time.sleep(0.001) # 1ms delay
        
        if (idx + 1) % 200 == 0:
            print(f"Published {idx + 1} transactions...")
            
    producer.flush()
    elapsed = time.time() - start_time
    print(f"Finished streaming {len(subset)} transactions in {elapsed:.2f} seconds!")

if __name__ == '__main__':
    main()
