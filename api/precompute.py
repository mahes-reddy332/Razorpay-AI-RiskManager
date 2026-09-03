"""
PHASE 1 — Precompute Cache

Runs the full frozen L1 pipeline (Tier 0 + Tier 1) across the entire
account population and writes results to a JSON cache file.

In production, this script would run periodically (e.g., every hour via
a cron job or triggered by a transaction-ingestion webhook), NOT
synchronously per API request.
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_v2 import extract_features, apply_frozen_config, FROZEN_CONFIG

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'cache', 'scores.json')


def precompute():
    print("=" * 60)
    print("PRECOMPUTE CACHE — Running full L1 pipeline")
    print("=" * 60)

    df = extract_features()
    scores = apply_frozen_config(df, FROZEN_CONFIG)

    cache = {}
    for i, acc in enumerate(df.index):
        score = float(scores[i])
        if score >= FROZEN_CONFIG['dec_thresh']:
            decision = "HIGH_RISK"
        elif score >= FROZEN_CONFIG['manual_thresh']:
            decision = "MANUAL_REVIEW"
        else:
            decision = "SAFE"

        cache[acc] = {
            "account_id": acc,
            "risk_score": round(score, 4),
            "decision": decision,
            "is_mule": bool(df.loc[acc, 'is_mule']),
            "signals": {
                "base_risk_score": round(float(df.loc[acc, 'risk_score']), 4),
                "velocity_ratio": round(float(df.loc[acc, 'max_velocity_ratio']), 4),
                "has_risky_sink": int(df.loc[acc, 'has_risky_sink']),
                "topology_node_count": int(df.loc[acc, 'node_count']),
                "log_amount_zscore": round(float(df.loc[acc, 'log_amount_zscore']), 4),
                "betweenness": round(float(df.loc[acc, 'betweenness']), 6),
                "pagerank": round(float(df.loc[acc, 'pagerank']), 6),
            },
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    flagged = sum(1 for v in cache.values() if v['decision'] == 'HIGH_RISK')
    review = sum(1 for v in cache.values() if v['decision'] == 'MANUAL_REVIEW')
    safe = sum(1 for v in cache.values() if v['decision'] == 'SAFE')

    print("\nPrecomputation Complete!")
    print(f"Total accounts cached: {len(cache)}")
    print(f"  HIGH_RISK:      {flagged}")
    print(f"  MANUAL_REVIEW:  {review}")
    print(f"  SAFE:           {safe}")
    print("Done.")


if __name__ == "__main__":
    precompute()
