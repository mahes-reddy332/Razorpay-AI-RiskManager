import time
from datetime import datetime

print('=== STARTING TIER 0 EVENT-DRIVEN INGESTION DEMO ===')
print('=== CONSUMER: LISTENING FOR TIER 0 EVENTS ===\n')

transactions = [
    {'txn_id': 'T1', 'sender': 'ACC_DEMO_99', 'amount': 500},
    {'txn_id': 'T2', 'sender': 'ACC_DEMO_99', 'amount': 1200},
    {'txn_id': 'T3', 'sender': 'ACC_DEMO_99', 'amount': 800},
    {'txn_id': 'T4', 'sender': 'ACC_DEMO_99', 'amount': 95000},
]

velocity = 0
for txn in transactions:
    t1 = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{t1}] PRODUCER: Sending TXN {txn['txn_id']} | Amount: {txn['amount']} | Sender: {txn['sender']}")
    time.sleep(0.01)
    
    velocity += txn['amount']
    t2 = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{t2}] CONSUMER: Received TXN {txn['txn_id']} | Total Sender Velocity: {velocity}")
    
    if velocity > 50000:
        print(f"  --> [{t2}] [ALERT] Account {txn['sender']} crossed Tier 0 velocity gate ({velocity} > 50000).")
        print(f"  --> [{t2}] [ACTION] Routing to RISK_CONTAINMENT_REQUIRED / L2 Review instantly!")
    
    time.sleep(1)

print('\n=== PRODUCER DEMO COMPLETE ===')
