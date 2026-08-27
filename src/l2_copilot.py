"""
Level 2 LLM Copilot — reads a local audit_log.json (produced by the L1
graph engine for MANUAL_REVIEW_REQUIRED accounts) and asks an LLM for a
structured Fraud/Safe recommendation.

Standalone and additive by design: does not import from or modify
anything in the existing L1 pipeline. Run it separately, after audit.py
has produced its JSON log.

Setup:
    pip install google-genai pydantic
    export GEMINI_API_KEY=your_key_here   # free key: https://aistudio.google.com/apikey

Usage:
    python src/l2_copilot.py --input outputs/audit_log.json --output outputs/l2_decisions.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from google import genai
except ImportError:
    print("Missing dependency. Run: pip install google-genai pydantic", file=sys.stderr)
    sys.exit(1)


class L2Decision(BaseModel):
    decision: Literal["FRAUD", "SAFE", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="2-3 sentence plain-language explanation")
    key_signal: str = Field(description="The single strongest signal driving this decision")


SYSTEM_PROMPT = """You are a Level 2 fraud review assistant for a UPI mule-account
detection system. The Level 1 graph engine has already flagged this account as
MANUAL_REVIEW_REQUIRED because deterministic rules (transaction ratio, dormancy,
topology, MCC category) could not confidently classify it.

You will be given the account's audit record: graph topology, transaction
history, and metadata. Make a final FRAUD / SAFE / UNCERTAIN call using the
kind of judgment the deterministic rules can't apply — for example, whether
the combination of signals tells a plausible legitimate story (a returning
customer, a business paying suppliers) versus a plausible mule story (funds
passing through with no economic purpose).

Rules:
- Base your decision only on the data provided. Do not assume facts not present.
- If the record is genuinely ambiguous even to you, say UNCERTAIN rather than
  guessing — a wrong confident answer is worse than an honest "still needs a
  human."
- Never fabricate transaction details, dates, or amounts not in the input.
"""


def call_l2(client: "genai.Client", record: dict, model: str) -> L2Decision:
    prompt = f"{SYSTEM_PROMPT}\n\nAccount audit record:\n{json.dumps(record, indent=2, default=str)}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": L2Decision,
        },
    )
    return response.parsed


def main():
    parser = argparse.ArgumentParser(description="L2 LLM Copilot for MANUAL_REVIEW_REQUIRED accounts")
    parser.add_argument("--input", required=True, help="Path to audit_log.json from the L1 pipeline")
    parser.add_argument("--output", required=True, help="Path to write L2 decisions JSON")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        audit_records = json.load(f)

    # ONLY PROCESS ACCOUNTS FLAGGED FOR MANUAL REVIEW
    review_records = [r for r in audit_records if r.get("decision") == "MANUAL_REVIEW_REQUIRED"]

    # Optional: If using a free-tier key, uncomment the line below to test on a small sample.
    # review_records = review_records[:5]

    if not review_records:
        print("No accounts flagged for MANUAL_REVIEW_REQUIRED found in the input file.")
        sys.exit(0)

    print(f"Found {len(review_records)} accounts requiring L2 manual review. Processing...")

    client = genai.Client(api_key=api_key)
    results = []

    for record in review_records:
        account_id = record.get("account_id", "UNKNOWN")
        outcome = {"account_id": account_id, "input_record": record}

        for attempt in range(args.max_retries + 1):
            try:
                decision = call_l2(client, record, args.model)
                outcome["l2_decision"] = decision.model_dump()
                outcome["status"] = "success"
                break
            except Exception as e:
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                # Graceful degradation, matching the L1 pipeline's own
                # philosophy: an API failure must never silently become a
                # FRAUD or SAFE verdict.
                outcome["l2_decision"] = None
                outcome["status"] = "failed"
                outcome["error"] = str(e)
                outcome["fallback"] = "MANUAL_REVIEW_REQUIRED"

        results.append(outcome)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"Processed {len(results)} accounts: {succeeded} succeeded, "
          f"{len(results) - succeeded} fell back to manual review.")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
