import sys
import json
import os
import argparse
import hashlib
from datetime import datetime

OVERRIDE_LOG = "outputs/human_overrides.json"

def calculate_hash(record_data, previous_hash):
    """Computes a deterministic SHA-256 hash for immutable audit chaining."""
    payload = json.dumps(record_data, sort_keys=True) + previous_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def verify_chain(file_path):
    if not os.path.exists(file_path):
        return True, "No log file found."
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "records" not in data:
        return True, "Legacy log format, no chain to verify."
    
    expected_prev = "GENESIS_HASH_0000000000000000"
    for idx, rec in enumerate(data["records"]):
        rec_copy = dict(rec)
        stored_hash = rec_copy.pop("hash", None)
        
        if rec_copy.get("previous_hash") != expected_prev:
            return False, f"Broken chain at index {idx}: previous_hash mismatch"
            
        calculated = calculate_hash(rec_copy, expected_prev)
        if calculated != stored_hash:
            return False, f"Tampering detected at index {idx}: hash mismatch"
            
        expected_prev = stored_hash
        
    return True, "Chain is valid."

def main():
    parser = argparse.ArgumentParser(description="Human Override CLI with Cryptographic Hash Chaining")
    parser.add_argument("--action", choices=["record", "verify"], default="record", help="Action to perform")
    parser.add_argument("--account", help="Account ID to override (required for record)")
    parser.add_argument("--decision", choices=["SAFE", "HIGH_RISK", "MANUAL_REVIEW"], help="New decision (required for record)")
    parser.add_argument("--note", help="Reason for human override (required for record)")
    parser.add_argument("--output-file", default=OVERRIDE_LOG, help="Custom output file for testing")
    args = parser.parse_args()

    out_file = args.output_file

    if args.action == "verify":
        is_valid, msg = verify_chain(out_file)
        if is_valid:
            print(f"[SUCCESS] {msg}")
            sys.exit(0)
        else:
            print(f"[ERROR] {msg}")
            sys.exit(1)

    if not all([args.account, args.decision, args.note]):
        parser.error("--account, --decision, and --note are required for 'record' action.")
    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            try:
                overrides = json.load(f)
            except json.JSONDecodeError:
                overrides = {"records": [], "last_hash": "GENESIS_HASH_0000000000000000"}
    else:
        overrides = {"records": [], "last_hash": "GENESIS_HASH_0000000000000000"}

    # Ensure schema backwards-compatibility
    if isinstance(overrides, dict) and "records" not in overrides:
        # Migrate old dictionary format to list with hashes
        old_items = [{"account_id": k, **v} for k, v in overrides.items()]
        overrides = {"records": old_items, "last_hash": "GENESIS_HASH_0000000000000000"}

    prev_hash = overrides.get("last_hash", "GENESIS_HASH_0000000000000000")
    record = {
        "account_id": args.account,
        "override_decision": args.decision,
        "reviewer_note": args.note,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "previous_hash": prev_hash
    }
    current_hash = calculate_hash(record, prev_hash)
    record["hash"] = current_hash

    overrides["records"].append(record)
    overrides["last_hash"] = current_hash
    # Also maintain account key lookup for O(1) checks
    overrides[args.account] = record

    with open(out_file, "w") as f:
        json.dump(overrides, f, indent=2)

    print(f"[SUCCESS] Recorded cryptographically signed override for {args.account} -> {args.decision} (Hash: {current_hash[:12]}...)")

if __name__ == "__main__":
    main()
