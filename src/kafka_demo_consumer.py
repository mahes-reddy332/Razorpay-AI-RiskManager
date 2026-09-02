import json
from datetime import datetime
from kafka import KafkaConsumer

# Mocking the Tier 0 State Store (in production, this is Redis or Flink State)
account_state_store = {}

# Frozen Tier 0 Thresholds
TIER_0_VELOCITY_THRESHOLD = 50000

def process_transaction(txn):
    sender = txn['sender']
    amount = float(txn['amount'])
    
    # Initialize state if brand new
    if sender not in account_state_store:
        account_state_store[sender] = {
            "total_sent_24h": 0.0,
            "txn_count": 0,
            "flagged": False
        }
        
    state = account_state_store[sender]
    
    # 1. Incrementally update Tier 0 aggregates
    state["total_sent_24h"] += amount
    state["txn_count"] += 1
    
    current_velocity = state["total_sent_24h"]
    
    # 2. Re-score against the frozen Tier 0 gate
    if current_velocity > TIER_0_VELOCITY_THRESHOLD and not state["flagged"]:
        state["flagged"] = True
        return True, current_velocity
        
    return False, current_velocity

def run_consumer():
    consumer = KafkaConsumer(
        'upi-transactions',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    print("=== CONSUMER: LISTENING FOR TIER 0 EVENTS ===")
    
    for message in consumer:
        txn = message.value
        arrival_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
        
        flagged, velocity = process_transaction(txn)
        
        print(f"[{arrival_time}] CONSUMER: Received TXN {txn['txn_id']} | Total Sender Velocity: {velocity}")
        
        if flagged:
            # 3. Log promotion territory decision at ingestion time
            print(f"  --> [{arrival_time}] [ALERT] Account {txn['sender']} crossed Tier 0 velocity gate ({velocity} > {TIER_0_VELOCITY_THRESHOLD}).")
            print(f"  --> [{arrival_time}] [ACTION] Routing to RISK_CONTAINMENT_REQUIRED / L2 Review instantly!")

if __name__ == "__main__":
    run_consumer()
