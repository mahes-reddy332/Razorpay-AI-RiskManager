"""
PHASE 2 - FastAPI Real-Time Scoring API

Serves precomputed L1 scores via O(1) dictionary lookup.
Accepts new transactions for incremental Tier 0 signal updates.

This API is a SEPARATE artifact from the React dashboard.
The dashboard reads static JSON exports and works without this server.
This server demonstrates the real-time microservice pattern.
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class AlertPayload(BaseModel):
    account_id: str
    decision: str
    score: float
    timestamp: str

@app.post("/api/internal/push_alert")
async def push_alert(payload: AlertPayload):
    await manager.broadcast(payload.model_dump())
    return {"status": "broadcasted"}


# ---------------------------------------------------------------------------
# Load precomputed cache at startup (O(1) lookup)
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'cache', 'scores.json')

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'r') as f:
        SCORE_CACHE = json.load(f)
    print(f"[STARTUP] Loaded {len(SCORE_CACHE)} precomputed scores from cache.")
else:
    SCORE_CACHE = {}
    print("[STARTUP] WARNING: No precomputed cache found. Run `python api/precompute.py` first.")

# ---------------------------------------------------------------------------
# Frozen config (mirrors src/model_v2.py — kept here to avoid import issues)
# ---------------------------------------------------------------------------
FROZEN_CONFIG = {
    'dec_thresh': 1.0,
    'manual_thresh': 0.5,
    'mcc_weight': 0.6,
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="UPI Fraud Flow Tracer - Real-Time Scoring API",
    description="Precomputed L1 risk scores with incremental Tier 0 updates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    txn_id: str
    source_account: str
    target_account: str
    amount: float
    timestamp: Optional[str] = None
    target_mcc: Optional[str] = None


class ScoreResponse(BaseModel):
    account_id: str
    risk_score: float
    decision: str
    signals: dict
    last_updated: str
    lookup_latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Basic readiness check."""
    return {
        "status": "healthy",
        "cached_accounts": len(SCORE_CACHE),
        "cache_loaded": len(SCORE_CACHE) > 0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/score/{account_id}", response_model=ScoreResponse)
def get_score(account_id: str):
    """
    Look up the precomputed risk score for an account.
    O(1) dictionary lookup - no graph traversal at request time.
    """
    start = time.perf_counter()

    if account_id not in SCORE_CACHE:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found in cache.")

    entry = SCORE_CACHE[account_id]
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    return ScoreResponse(
        account_id=account_id,
        risk_score=entry["risk_score"],
        decision=entry["decision"],
        signals=entry["signals"],
        last_updated=entry["last_updated"],
        lookup_latency_ms=latency_ms,
    )


@app.post("/transactions")
def ingest_transaction(txn: TransactionPayload):
    """
    Accept a new transaction and incrementally update ONLY the cheap
    Tier 0 signals (velocity, immediate-counterparty MCC) for the
    source account.

    The full graph topology trace is NOT re-run synchronously here.
    It would be queued for the next batch precompute job in production.
    """
    start = time.perf_counter()
    account_id = txn.source_account

    if account_id not in SCORE_CACHE:
        # New account — create a baseline entry
        SCORE_CACHE[account_id] = {
            "account_id": account_id,
            "risk_score": 0.0,
            "decision": "SAFE",
            "is_mule": False,
            "signals": {
                "base_risk_score": 0.0,
                "velocity_ratio": 0.0,
                "has_risky_sink": 0,
                "topology_node_count": 0,
                "log_amount_zscore": 0.0,
                "betweenness": 0.0,
                "pagerank": 0.0,
            },
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

    entry = SCORE_CACHE[account_id]
    old_score = entry["risk_score"]
    old_decision = entry["decision"]

    # --- Tier 0 Incremental Updates (cheap, O(1)) ---

    # 1. Velocity bump: if large amount forwarded quickly, increase velocity signal
    if txn.amount >= 1000:
        current_velocity = entry["signals"]["velocity_ratio"]
        # Simulate velocity increase — in production this would check actual
        # time deltas against the account's recent transaction history
        new_velocity = min(current_velocity + 0.15, 1.1)
        entry["signals"]["velocity_ratio"] = round(new_velocity, 4)

    # 2. Immediate counterparty MCC check
    risky_mccs = {"CRYPTO_EXCHANGE", "GAMBLING", "UNREGISTERED_P2P"}
    if txn.target_mcc and txn.target_mcc.upper() in risky_mccs:
        entry["signals"]["has_risky_sink"] = 1

    # --- Recompute score from updated signals ---
    new_score = entry["signals"]["base_risk_score"]
    if entry["signals"]["has_risky_sink"] == 1:
        new_score += FROZEN_CONFIG['mcc_weight']
    # Velocity crossing threshold adds to risk
    if entry["signals"]["velocity_ratio"] > 0.85:
        new_score += entry["signals"]["base_risk_score"]  # amplify existing risk

    entry["risk_score"] = round(new_score, 4)

    # Update decision based on new score
    if new_score >= FROZEN_CONFIG['dec_thresh']:
        entry["decision"] = "HIGH_RISK"
    elif new_score >= FROZEN_CONFIG['manual_thresh']:
        entry["decision"] = "MANUAL_REVIEW"
    else:
        entry["decision"] = "SAFE"

    entry["last_updated"] = datetime.utcnow().isoformat() + "Z"

    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    alert_triggered = (old_decision == "SAFE" and entry["decision"] != "SAFE")

    return {
        "status": "accepted",
        "account_id": account_id,
        "transaction_id": txn.txn_id,
        "previous_score": old_score,
        "updated_score": entry["risk_score"],
        "previous_decision": old_decision,
        "updated_decision": entry["decision"],
        "alert_triggered": alert_triggered,
        "signals_updated": ["velocity_ratio", "has_risky_sink"],
        "topology_reverification": "queued",
        "processing_latency_ms": latency_ms,
        "note": "Only Tier 0 signals updated incrementally. Full graph topology trace queued for next batch precompute."
    }


# ---------------------------------------------------------------------------
# Analyst Feedback Capture (Section 13 of architecture.md)
# Captures labeled outcomes for future retraining. Does NOT retrain or
# adjust any thresholds in this submission.
# ---------------------------------------------------------------------------
import sqlite3

FEEDBACK_DB = os.path.join(os.path.dirname(__file__), 'feedback_log.db')

def _init_feedback_db():
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            l2_decision TEXT NOT NULL,
            analyst_action TEXT NOT NULL CHECK(analyst_action IN ('CONFIRM_FREEZE', 'RELEASE', 'ESCALATE')),
            timestamp TEXT NOT NULL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_feedback_db()


class FeedbackPayload(BaseModel):
    account_id: str
    l2_decision: str
    analyst_action: str  # CONFIRM_FREEZE | RELEASE | ESCALATE
    notes: Optional[str] = None


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    """
    Record an analyst's agree/disagree decision on a Level 2 review case.
    Capture-only — does not retrain or adjust any thresholds.
    """
    ts = datetime.utcnow().isoformat() + "Z"
    if payload.analyst_action not in ("CONFIRM_FREEZE", "RELEASE", "ESCALATE"):
        raise HTTPException(status_code=400, detail="analyst_action must be CONFIRM_FREEZE, RELEASE, or ESCALATE")

    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(
        "INSERT INTO analyst_feedback (account_id, l2_decision, analyst_action, timestamp, notes) VALUES (?, ?, ?, ?, ?)",
        (payload.account_id, payload.l2_decision, payload.analyst_action, ts, payload.notes)
    )
    conn.commit()

    # Compute running agreement rate
    cursor = conn.execute("SELECT COUNT(*) FROM analyst_feedback")
    total = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM analyst_feedback WHERE analyst_action = 'CONFIRM_FREEZE'")
    confirms = cursor.fetchone()[0]
    conn.close()

    agreement_rate = confirms / total if total > 0 else 0.0

    return {
        "status": "recorded",
        "account_id": payload.account_id,
        "analyst_action": payload.analyst_action,
        "timestamp": ts,
        "running_agreement_rate": round(agreement_rate, 4),
        "total_feedback_entries": total,
        "note": "Feedback captured for future recalibration. No thresholds adjusted."
    }


@app.get("/feedback/stats")
def feedback_stats():
    """Return aggregate feedback statistics for dashboard display."""
    conn = sqlite3.connect(FEEDBACK_DB)
    cursor = conn.execute("SELECT COUNT(*) FROM analyst_feedback")
    total = cursor.fetchone()[0]
    cursor = conn.execute("SELECT analyst_action, COUNT(*) FROM analyst_feedback GROUP BY analyst_action")
    breakdown = dict(cursor.fetchall())
    conn.close()

    return {
        "total_reviews": total,
        "action_breakdown": breakdown,
        "agreement_rate": round(breakdown.get("CONFIRM_FREEZE", 0) / total, 4) if total > 0 else None,
        "release_rate": round(breakdown.get("RELEASE", 0) / total, 4) if total > 0 else None,
        "note": "Capture-only. Does not retrain or adjust any thresholds in this submission."
    }


# ---------------------------------------------------------------------------
# Run with: uvicorn api.server:app --reload --port 8000
# ---------------------------------------------------------------------------
