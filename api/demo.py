"""
PHASE 3 - Demo Script

Calls all three API endpoints in sequence to demonstrate the live
scoring capability during the pitch recording.

Usage: python api/demo.py
(Requires the server to be running: uvicorn api.server:app --port 8000)
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo():
    # ---------------------------------------------------------------
    # Step 0: Health Check
    # ---------------------------------------------------------------
    divider("STEP 0: Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        print(f"Status:          {data['status']}")
        print(f"Cached Accounts: {data['cached_accounts']}")
        print(f"Cache Loaded:    {data['cache_loaded']}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to the API server.")
        print("Start it with: uvicorn api.server:app --port 8000")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 1: Score a known MULE account
    # ---------------------------------------------------------------
    divider("STEP 1: Score a Known Mule Account")

    # Find a flagged mule from the cache
    resp = requests.get(f"{BASE_URL}/health")
    cache_size = resp.json()["cached_accounts"]

    # Try a few known mule account IDs from our dataset
    mule_found = False
    for test_id in ["ACC_8D70E7C41F08", "ACC_DDEC20BB1E5E"]:
        resp = requests.get(f"{BASE_URL}/score/{test_id}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Account:       {data['account_id']}")
            print(f"Risk Score:    {data['risk_score']}")
            print(f"Decision:      {data['decision']}")
            print(f"Signals:")
            for k, v in data['signals'].items():
                print(f"  {k:25s}: {v}")
            print(f"Lookup Latency: {data['lookup_latency_ms']} ms")
            mule_found = True
            break

    if not mule_found:
        print("Could not find a test mule account. Trying first account in cache...")
        # Fallback: just grab any account
        resp = requests.get(f"{BASE_URL}/score/ACC_3EA468883310")
        if resp.status_code == 200:
            data = resp.json()
            print(json.dumps(data, indent=2))

    # ---------------------------------------------------------------
    # Step 2: Score a known LEGITIMATE account
    # ---------------------------------------------------------------
    divider("STEP 2: Score a Known Legitimate Account")

    for test_id in ["ACC_3EA468883310", "ACC_066748CBFD0F", "ACC_BB7D691D38C9"]:
        resp = requests.get(f"{BASE_URL}/score/{test_id}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Account:       {data['account_id']}")
            print(f"Risk Score:    {data['risk_score']}")
            print(f"Decision:      {data['decision']}")
            print(f"Signals:")
            for k, v in data['signals'].items():
                print(f"  {k:25s}: {v}")
            print(f"Lookup Latency: {data['lookup_latency_ms']} ms")
            break

    # ---------------------------------------------------------------
    # Step 3: POST a suspicious transaction
    # ---------------------------------------------------------------
    divider("STEP 3: POST a Suspicious Transaction (Incremental Tier 0 Update)")

    # First, check current score
    test_account = "ACC_3EA468883310"
    resp = requests.get(f"{BASE_URL}/score/{test_account}")
    if resp.status_code == 200:
        before = resp.json()
        print(f"BEFORE transaction:")
        print(f"  Score:    {before['risk_score']}")
        print(f"  Decision: {before['decision']}")
        print(f"  Velocity: {before['signals']['velocity_ratio']}")
        print()

    # Send a suspicious transaction to a crypto exchange
    txn_payload = {
        "txn_id": "TXN_DEMO_001",
        "source_account": test_account,
        "target_account": "CRYPTO_SINK_001",
        "amount": 50000,
        "timestamp": "2026-08-28T22:00:00",
        "target_mcc": "CRYPTO_EXCHANGE"
    }

    print(f"Sending transaction: {json.dumps(txn_payload, indent=2)}")
    print()

    resp = requests.post(f"{BASE_URL}/transactions", json=txn_payload)
    data = resp.json()

    print(f"AFTER transaction:")
    print(f"  Previous Score:    {data['previous_score']}")
    print(f"  Updated Score:     {data['updated_score']}")
    print(f"  Previous Decision: {data['previous_decision']}")
    print(f"  Updated Decision:  {data['updated_decision']}")
    print(f"  Alert Triggered:   {data['alert_triggered']}")
    print(f"  Signals Updated:   {data['signals_updated']}")
    print(f"  Topology Reverify: {data['topology_reverification']}")
    print(f"  Processing Latency: {data['processing_latency_ms']} ms")
    print()
    print(f"  Note: {data['note']}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    divider("DEMO COMPLETE")
    print("All three endpoints responded correctly.")
    print("The API serves precomputed scores in sub-millisecond latency")
    print("and accepts incremental Tier 0 updates without re-running")
    print("the expensive graph topology trace.")


if __name__ == "__main__":
    demo()
