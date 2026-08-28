import json
import os
import argparse
from datetime import datetime

OVERRIDE_LOG = "outputs/human_overrides.json"

def main():
    parser = argparse.ArgumentParser(description="Human Override CLI for Customer Due Process")
    parser.add_argument("--account", required=True, help="Account ID to override")
    parser.add_argument("--decision", required=True, choices=["SAFE", "FLAG_MULE", "MANUAL_REVIEW"], help="New decision")
    parser.add_argument("--note", required=True, help="Reason for human override")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if os.path.exists(OVERRIDE_LOG):
        with open(OVERRIDE_LOG, "r") as f:
            overrides = json.load(f)
    else:
        overrides = {}

    overrides[args.account] = {
        "account_id": args.account,
        "override_decision": args.decision,
        "reviewer_note": args.note,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(OVERRIDE_LOG, "w") as f:
        json.dump(overrides, f, indent=2)

    print(f"[SUCCESS] Recorded human override for {args.account} -> {args.decision}")

if __name__ == "__main__":
    main()
