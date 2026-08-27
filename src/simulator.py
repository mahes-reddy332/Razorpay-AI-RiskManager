import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import os

class UPISimulator:
    def __init__(self, start_date=datetime(2026, 5, 1), end_date=datetime(2026, 7, 30)):
        self.start_date = start_date
        self.end_date = end_date
        self.accounts = []
        self.transactions = []
        
        # --- EXPLICIT RANDOMIZATION RANGES (For Audit) ---
        # Mule Chains
        self.MULE_AMT_RANGE = (10000, 200000)      # Large sums being moved
        self.MULE_HOP_RANGE = (2, 6)               # 2 to 6 intermediate hops
        self.MULE_FAST_DELAY_MINS = (5, 60)        # Minutes between hops (Fast)
        self.MULE_SLOW_DELAY_MINS = (360, 1440)    # 6-24 hours between hops (Slow)
        self.MULE_COMMISSION = (0.02, 0.08)        # 2-8% cut per hop
        self.MULE_SPLIT_WAYS = (2, 5)              # Splitting into 2-5 streams
        
        # Hard Negatives
        self.SALARY_EMPLOYEES = (5, 50)            # 5 to 50 employees per batch
        self.SALARY_AMT = (20000, 150000)          # Typical monthly salary
        self.SMB_CUSTOMERS = (10, 60)              # SMB receives 10-60 payments/day
        self.SMB_TXN_AMT = (100, 3000)             # Normal consumer purchases
        self.DORMANT_DAYS = (45, 120)              # Inactive for 1.5 to 4 months
        self.DORMANT_PURCHASE = (0.85, 0.98)       # Spends 85-98% of lump sum
        self.BILL_SPLIT_FRIENDS = (2, 8)           # Group dinner sizes
        
        # Background Legit Traffic
        self.NORMAL_TXN_PER_DAY = (0.1, 2.0)       # Average txns per day per user
        
    def generate_account(self, is_mule=False, behavior="normal", dormant_days=0, shared_device=None):
        acc_id = f"ACC_{uuid.uuid4().hex[:8].upper()}"
        creation_date = self.start_date - timedelta(days=random.randint(dormant_days, dormant_days + 60))
        device_id = shared_device if shared_device else f"DEV_{uuid.uuid4().hex[:8].upper()}"
        
        mcc_code = "NONE"
        if behavior == "merchant": 
            if random.random() < 0.90: mcc_code = random.choice(["HOSPITAL", "EDUCATION", "UTILITIES"])
            else: mcc_code = random.choice(["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"])
        elif behavior == "supplier": 
            if random.random() < 0.90: mcc_code = "WHOLESALE"
            else: mcc_code = random.choice(["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"])
        elif behavior == "mule_cashout": 
            if random.random() < 0.80: mcc_code = random.choice(["CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"])
            else: mcc_code = random.choice(["HOSPITAL", "EDUCATION", "UTILITIES", "WHOLESALE"])
        
        self.accounts.append({
            "account_id": acc_id,
            "creation_date": creation_date,
            "device_id": device_id,
            "activity_level": "low" if dormant_days > 0 else random.choice(["low", "medium", "high"]),
            "kyc_tier": random.choice(["tier1", "tier2", "tier3"]),
            "mcc_code": mcc_code,
            "is_mule": is_mule,
            "behavior_type": behavior
        })
        return acc_id

    def add_transaction(self, source, target, amount, timestamp, chain_id=None, is_mule_chain=False):
        if timestamp > self.end_date:
            return # Cap at end date
        self.transactions.append({
            "txn_id": f"TXN_{uuid.uuid4().hex[:10].upper()}",
            "timestamp": timestamp,
            "source_account": source,
            "target_account": target,
            "amount": round(amount, 2),
            "is_mule_chain": is_mule_chain,
            "chain_id": chain_id
        })

    def random_timestamp(self):
        delta = self.end_date - self.start_date
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return self.start_date + timedelta(seconds=random_seconds)

    # --- Mule Generators (using explicit ranges) ---
    def inject_mule_chain_linear(self, is_slow=False, use_commission=False):
        num_hops = random.randint(*self.MULE_HOP_RANGE)
        base_amount = random.uniform(*self.MULE_AMT_RANGE)
        delay_range = self.MULE_SLOW_DELAY_MINS if is_slow else self.MULE_FAST_DELAY_MINS
        comm_range = self.MULE_COMMISSION if use_commission else (0.0, 0.0)
        
        chain_id = f"CHAIN_MULE_LIN_{uuid.uuid4().hex[:6].upper()}"
        current_time = self.random_timestamp() - timedelta(days=2) # ensure fits in window
        
        nodes = [self.generate_account(is_mule=True, behavior="mule_hop") for _ in range(num_hops)]
        prev_node = f"EXT_VICTIM_{uuid.uuid4().hex[:4].upper()}"
        current_amount = base_amount
        
        for node in nodes:
            self.add_transaction(prev_node, node, current_amount, current_time, chain_id, True)
            current_time += timedelta(minutes=random.randint(*delay_range))
            current_amount = current_amount * (1.0 - random.uniform(*comm_range))
            prev_node = node
            
        cashout = self.generate_account(is_mule=True, behavior="mule_cashout")
        self.add_transaction(prev_node, cashout, current_amount, current_time, chain_id, True)

    def inject_mule_chain_split_reconverge(self):
        total_amount = random.uniform(*self.MULE_AMT_RANGE)
        num_splits = random.randint(*self.MULE_SPLIT_WAYS)
        chain_id = f"CHAIN_MULE_SPLIT_{uuid.uuid4().hex[:6].upper()}"
        ext_source = f"EXT_VICTIM_{uuid.uuid4().hex[:4].upper()}"
        start_time = self.random_timestamp() - timedelta(days=1)
        
        split_nodes = [self.generate_account(is_mule=True, behavior="mule_hop") for _ in range(num_splits)]
        aggregator = self.generate_account(is_mule=True, behavior="mule_hop")
        cashout = self.generate_account(is_mule=True, behavior="mule_cashout")
        
        split_amount = total_amount / num_splits
        
        for node in split_nodes:
            jitter_amt = split_amount * random.uniform(0.9, 1.1)
            jitter_time = start_time + timedelta(minutes=random.randint(0, 30))
            self.add_transaction(ext_source, node, jitter_amt, jitter_time, chain_id, True)
            
            fwd_time = jitter_time + timedelta(minutes=random.randint(*self.MULE_FAST_DELAY_MINS))
            fwd_amt = jitter_amt * (1.0 - random.uniform(*self.MULE_COMMISSION))
            self.add_transaction(node, aggregator, fwd_amt, fwd_time, chain_id, True)
            
        max_fwd_time = start_time + timedelta(minutes=180)
        agg_received = sum([t['amount'] for t in self.transactions if t['target_account'] == aggregator and t['chain_id'] == chain_id])
        self.add_transaction(aggregator, cashout, agg_received, max_fwd_time, chain_id, True)

    def inject_mule_chain_structuring(self):
        total_amount = random.uniform(*self.MULE_AMT_RANGE)
        num_splits = random.randint(*self.MULE_SPLIT_WAYS)
        chain_id = f"CHAIN_MULE_STRUCT_{uuid.uuid4().hex[:6].upper()}"
        start_time = self.random_timestamp() - timedelta(days=1)
        
        node1 = self.generate_account(is_mule=True, behavior="mule_hop")
        ext_source = f"EXT_VICTIM_{uuid.uuid4().hex[:4].upper()}"
        self.add_transaction(ext_source, node1, total_amount, start_time, chain_id, True)
        
        split_amount = (total_amount / num_splits) * random.uniform(0.95, 0.99)
        for _ in range(num_splits):
            cashout = self.generate_account(is_mule=True, behavior="mule_cashout")
            out_time = start_time + timedelta(minutes=random.randint(*self.MULE_SLOW_DELAY_MINS))
            self.add_transaction(node1, cashout, split_amount * random.uniform(0.9, 1.1), out_time, chain_id, True)

    # --- Hard Negative Generators ---
    def inject_salary_disbursement(self):
        num_emps = random.randint(*self.SALARY_EMPLOYEES)
        salary = random.uniform(*self.SALARY_AMT)
        employer = self.generate_account(is_mule=False, behavior="salary_disburser")
        ext_source = f"EXT_CORP_{uuid.uuid4().hex[:4].upper()}"
        start_time = self.random_timestamp()
        chain_id = f"CHAIN_HN_SALARY_{uuid.uuid4().hex[:6].upper()}"
        
        self.add_transaction(ext_source, employer, salary * num_emps, start_time, chain_id, False)
        for _ in range(num_emps):
            emp = self.generate_account(is_mule=False, behavior="normal")
            self.add_transaction(employer, emp, salary, start_time + timedelta(minutes=random.randint(1, 15)), chain_id, False)
            
    def inject_small_business(self):
        num_custs = random.randint(*self.SMB_CUSTOMERS)
        smb = self.generate_account(is_mule=False, behavior="small_business")
        chain_id = f"CHAIN_HN_SMB_{uuid.uuid4().hex[:6].upper()}"
        current_time = self.random_timestamp() - timedelta(days=2)
        total_collected = 0
        for _ in range(num_custs):
            cust = self.generate_account(is_mule=False, behavior="normal")
            amt = random.uniform(*self.SMB_TXN_AMT)
            self.add_transaction(cust, smb, amt, current_time, chain_id, False)
            total_collected += amt
            current_time += timedelta(minutes=random.randint(5, 30))
        supplier = self.generate_account(is_mule=False, behavior="supplier")
        self.add_transaction(smb, supplier, total_collected * 0.9, current_time + timedelta(hours=2), chain_id, False)

    def inject_bill_split(self):
        num_friends = random.randint(*self.BILL_SPLIT_FRIENDS)
        payer = self.generate_account(is_mule=False, behavior="normal")
        restaurant = f"EXT_REST_{uuid.uuid4().hex[:4].upper()}"
        start_time = self.random_timestamp()
        chain_id = f"CHAIN_HN_SPLIT_{uuid.uuid4().hex[:6].upper()}"
        total_bill = random.uniform(1000, 10000)
        split = total_bill / (num_friends + 1)
        self.add_transaction(payer, restaurant, total_bill, start_time, chain_id, False)
        for _ in range(num_friends):
            friend = self.generate_account(is_mule=False, behavior="normal")
            self.add_transaction(friend, payer, split, start_time + timedelta(minutes=random.randint(5, 60)), chain_id, False)

    def inject_dormant_legit_large_purchase(self):
        dormant_days = random.randint(*self.DORMANT_DAYS)
        acc = self.generate_account(is_mule=False, behavior="dormant_legit", dormant_days=dormant_days)
        start_time = self.start_date + timedelta(days=dormant_days + random.randint(1, 10))
        if start_time > self.end_date: start_time = self.end_date - timedelta(days=2)
        chain_id = f"CHAIN_HN_DORMANT_PURCHASE_{uuid.uuid4().hex[:6].upper()}"
        
        deposit = random.uniform(*self.MULE_AMT_RANGE)
        self.add_transaction(f"EXT_BANK_{uuid.uuid4().hex[:4].upper()}", acc, deposit, start_time, chain_id, False)
        
        merchant = self.generate_account(is_mule=False, behavior="merchant")
        spend = deposit * random.uniform(*self.DORMANT_PURCHASE)
        self.add_transaction(acc, merchant, spend, start_time + timedelta(minutes=random.randint(30, 180)), chain_id, False)

    # --- Full Generation ---
    def generate_full_dataset(self, num_accounts=5000, mule_prevalence=0.03):
        target_mules = int(num_accounts * mule_prevalence) # ~150 mules
        
        # 1. Generate Mule Chains
        mule_accs = 0
        while mule_accs < target_mules:
            choice = random.random()
            if choice < 0.4: self.inject_mule_chain_linear(is_slow=False, use_commission=False)
            elif choice < 0.6: self.inject_mule_chain_linear(is_slow=True, use_commission=False)
            elif choice < 0.8: self.inject_mule_chain_linear(is_slow=False, use_commission=True)
            elif choice < 0.9: self.inject_mule_chain_split_reconverge()
            else: self.inject_mule_chain_structuring()
            mule_accs = sum(1 for a in self.accounts if a['is_mule'])
            
        # 2. Generate Hard Negatives
        for _ in range(30): self.inject_salary_disbursement()
        for _ in range(30): self.inject_small_business()
        for _ in range(40): self.inject_dormant_legit_large_purchase()
        for _ in range(30): self.inject_bill_split()
        
        # 3. Generate Normal Users (Fill up to num_accounts)
        current_accs = len(self.accounts)
        normal_accs_needed = max(0, num_accounts - current_accs)
        normal_ids = []
        
        # Create some shared devices
        for _ in range(int(normal_accs_needed * 0.05)): # 5% pairs share device
            dev = f"DEV_FAMILY_{uuid.uuid4().hex[:8].upper()}"
            normal_ids.append(self.generate_account(shared_device=dev))
            normal_ids.append(self.generate_account(shared_device=dev))
            
        while len(self.accounts) < num_accounts:
            normal_ids.append(self.generate_account(is_mule=False, behavior="normal"))
            
        # 4. Generate Background Legitimate Traffic
        print("Generating background traffic...")
        days = (self.end_date - self.start_date).days
        for i, acc_id in enumerate(normal_ids):
            txns_to_make = random.randint(1, 20)
            for _ in range(txns_to_make):
                target = random.choice(normal_ids)
                if target == acc_id: continue
                amt = np.random.lognormal(mean=6, sigma=1.2) # ~400 median, heavily right skewed
                self.add_transaction(acc_id, target, amt, self.random_timestamp())
                
    def export(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        pd.DataFrame(self.accounts).to_csv(os.path.join(dir_path, 'accounts.csv'), index=False)
        pd.DataFrame(self.transactions).to_csv(os.path.join(dir_path, 'transactions.csv'), index=False)
        print(f"Exported {len(self.accounts)} accounts and {len(self.transactions)} transactions to {dir_path}")

if __name__ == "__main__":
    sim = UPISimulator()
    sim.generate_full_dataset(num_accounts=5000, mule_prevalence=0.03)
    sim.export('data')
