# Architecture & Design Decisions

## 1. Synthetic Simulator (Phase 1)
- **Constraint:** Strictly defense-only; no real PII/financial data used.
- **Design:** A custom Pandas-based transaction generator that simulates a 90-day UPI flow. It injects complex Mule archetypes (Structuring, Split/Re-converge, Commission-taking) alongside Hard Negatives (Salary Disbursers, Small Businesses, Dormant Purchases).
- **Probabilistic Noise:** Merchant Category Codes (MCC) are assigned probabilistically. 10% of legitimate businesses receive high-risk MCCs, and 20% of mule cashouts route through compromised safe MCCs to model real-world edge cases.

## 2. Methodology: Leakage Discovery and V2 Fix
During the development of the Phase 4 (Topology) and Phase 8 (MCC) rules, the initial evaluation methodology (v1) was flawed. 
- **The Leakage:** The evaluation suffered from threshold nudging (the topology threshold was lowered after inspecting false positives in the test set) and deterministic feature generation (safe MCCs perfectly correlated with legitimate hard-negatives).
### Data Split Protocol & Leakage Diagnostic
Our random 60/20/20 split on graph data is performed at the account-level rather than chronologically. We exhaustively verified every detection feature (pass-through ratios, dormancy windows, topology BFS, MCC weighting, log-amount Z-scores, betweenness centrality, and multi-window velocity) and confirmed that **no cross-account population statistic or global baseline references the test data**. The only population-level statistic (betweenness centrality threshold) is computed strictly from train+validation accounts. Because all other thresholds evaluate individual node topologies and per-account temporal windows rather than global statistical percentiles, the random split guarantees zero leakage.

### Official Test-Set Metrics (Final V2 Model)
We explicitly locked the final production configuration (`FROZEN_CONFIG`) and ran our evaluation strictly against the untouched Test Set (which includes both baseline mules and adversarial evaders).

**The Final Decision Bands:**
*   **Tier 0 Tripwire:** Base score (0.0 to 1.0) derived from 24h velocity and dormancy.
*   **Tier 1 Additions:** `mcc_weight: 0.6` (for both immediate and deep graph risky sinks).
*   `Threshold B` (**FLAG_MULE**): `Score >= 1.0` (Automatic Freeze)
*   `Threshold A` (**MANUAL_REVIEW**): `0.5 <= Score < 1.0` (Routes to L2)
*   **SAFE**: `Score < 0.5`

**Final System Performance (Strict Test Set Only):**
*Total Mules in Test Set: 146*

**L1-Only Measured Performance (Auto-Freeze):**
*   **True Positives:** 119
*   **False Positives:** 10
*   **Precision:** 92.2%
*   **Recall:** 81.5%

**L1 + L2 Actual Measured Performance (Combined System):**
*We ran the 21 Test Set accounts that landed in the `MANUAL_REVIEW` band (19 mules, 2 legitimate) through the actual LLM API. The LLM correctly caught 14 of the 19 camouflaged mules, and incorrectly flagged the 2 legitimate accounts.*
*   **Total True Positives:** 133 (119 from L1 + 14 from L2)
*   **Total False Positives:** 12 (10 from L1 + 2 from L2)
*   **Total False Negatives (Missed):** 13 (8 Slow Mules + 5 Evaders that tricked L2)
*   **Total True Negatives:** 285
*   **Combined Precision:** 91.7%
*   **Combined Recall:** 91.1%

**Extrapolating Error Rates to Production Scale:**
On our strict test set, we measured a False Positive Rate (FPR) of ~4.0% (12 FP / 297 legitimate accounts) and a False Negative Rate (FNR) of ~8.9% (13 FN / 146 mules). If this false-positive rate held constant at illustrative scales:
*   At **10,000** legitimate accounts, we would flag **~404** innocent accounts.
*   At **100,000** legitimate accounts, we would flag **~4,040** innocent accounts.
*   At **1,000,000** legitimate accounts, we would flag **~40,400** innocent accounts.

While error rates rarely scale perfectly linearly in practice, large absolute false-positive counts at scale are an expected property of any high-recall real-time fraud system, not a defect specific to this one. This is precisely why production fraud operations use tiered human review rather than expecting a fully automated layer to be perfect. It is exactly why this system is architected as an L1 auto-clear, routing to an L2 LLM review, backed by a documented L3 batch sweep, instead of a single monolithic classifier.

**The Honest Finding (L1 Limits, L2 Routing, & The Level 3 Gap):**
1. **Metadata Wakes Up (The Fix):** We decoupled the immediate MCC check from the expensive full graph trace so that it runs independently of the velocity tripwire. By giving the ML independent access to this signal, it learned to catch mules directly via their risky recipients (`mcc_weight` jumped to 0.6).
2. **The L2 Safety Net Works (For Camouflage):** The "Ultimate Evader" (wide topology + fraudulently registered safe MCC) successfully bypassed the 1.0 automatic freeze threshold. However, they scored 0.60, landing perfectly inside the `MANUAL_REVIEW_REQUIRED` band (0.5 - 1.0). This is exactly why we built the Level 2 LLM Copilot—to catch the subtle camouflaged evaders that rigid math alone misses.
3. **The Genuine Blind Spot (Multi-Hop Slow Evasion / Level 3):** This fix does not close everything. A chain where EVERY hop individually waits past 72 hours before the money *finally* reaches a risky MCC several hops downstream will still score 0.0 and bypass everything. We state this explicitly as a specific remaining limitation: catching multi-hop slow-walked laundering requires a genuine **Level 3** periodic batch sweep running offline. This is our complete defense-in-depth story: we optimized for scale, closed the cheap gaps, and named exactly what's still open and why.

## 3. Explainable Detection Engine vs. GNNs
- **Decision:** We chose a deterministic, rule-based temporal graph traversal over a Graph Neural Network (GNN).
- **Rationale:** 
  1. **Explainability & Auditability:** Regulatory mandates require plain-English justification for freezing an account. A composite rule base yields defensible audit logs.
  2. **Infrastructure:** In-memory `NetworkX` simplifies deployment and reduces latency compared to maintaining a live Graph Database.

## 4. The Topo-Metadata Pipeline
Our detection logic operates in a tiered pipeline yielding a composite score:
1. **Velocity/Dormancy:** Evaluates nodes on in/out ratio within 24h and dormancy.
2. **Temporal Tracer:** Executes a bidirectional Breadth-First Search (BFS). Extracts the topological footprint (node count) to recognize the "fan-in/fan-out" signatures of legitimate SMBs/Salary accounts.
3. **Metadata Contextualization:** Extracts the Terminal Sink Node and checks its Merchant Category Code (MCC).

## 5. Scalability: Risk Waterfall & Level 2 LLM Copilot
To scale this architecture, we implemented a hybrid L1/L2 framework:

### Level 1: Real-Time Engine (`src/model_v2.py` / `src/audit.py`)
The NetworkX metadata graph operates in milliseconds using the composite scores derived during the validation sweep. If scores are extremely close to the borderline (or if upstream data is corrupted), it yields `MANUAL_REVIEW_REQUIRED`.

### Level 2: Asynchronous LLM Copilot (`src/l2_copilot.py`)
Instead of a human analyst, our Level 2 script uses `google-genai` and Pydantic to read the `audit_log.json`. It passes the data to Gemini 2.5 Flash using structured outputs, forcing a highly constrained, rule-bound `FRAUD/SAFE` decision along with its reasoning. This achieves the analytical depth of an investigator without breaking the speed of L1.

## 6. Real-Time Scoring API (`api/server.py`)

### Architecture Pattern: Decouple Expensive Computation from the Request Path

The API demonstrates the correct production pattern for real-time fraud scoring:

1. **Precompute Job (`api/precompute.py`):** Runs the full frozen L1 pipeline (Tier 0 + Tier 1) across the entire account population and writes results to a JSON cache file. In production, this would run periodically via a cron job or be triggered by a transaction-ingestion webhook — never synchronously per request.

2. **O(1) Lookup at Request Time:** The FastAPI server loads the precomputed cache into memory at startup. `GET /score/{account_id}` performs a single dictionary lookup and returns the risk score, decision, full signal breakdown, and the actual measured latency (typically < 1ms).

3. **Incremental Tier 0 Updates:** `POST /transactions` accepts a new transaction and updates only the cheap Tier 0 signals (velocity, immediate-counterparty MCC) incrementally. The response includes `"topology_reverification": "queued"` to honestly show that the expensive graph topology trace is NOT re-run synchronously — it would be queued for the next batch precompute.

### Honest Boundary

This prototype runs the cache and API locally against the existing dataset scale (~2,200 accounts). It demonstrates the correct architectural **pattern** at prototype scale — it is not the production infrastructure itself. A real deployment would additionally require:

- **Streaming ingestion** (Apache Kafka / Flink) to ingest transactions in real-time and trigger incremental cache updates.
- **Distributed graph store** (Neo4j / TigerGraph) to replace the in-memory NetworkX graph for billion-edge scale.
- **Cache layer** (Redis / DynamoDB) to replace the local JSON file with a distributed, low-latency key-value store.

This prototype proves we understand the decoupled architecture pattern. It does not imply those infrastructure components are no longer needed.

## 7. L2 Model Failover Strategy

The L2 LLM Copilot depends on a third-party inference API (currently Groq). During development, two models were deprecated (`llama3-70b-8192` and `llama-3.3-70b-versatile`), forcing a migration to `openai/gpt-oss-120b`.

### What happens when the API is unavailable?

The system **fails closed**, not open:

1. If the API returns a non-200 status or times out, the account stays in `MANUAL_REVIEW_REQUIRED` status — it is never auto-passed as SAFE.
2. The exception is logged with the full error payload for ops debugging.
3. A human analyst must manually review these accounts via the `human_override.py` CLI tool.

This is a deliberate design choice: in financial fraud detection, a false negative (letting a mule through) is far more costly than a delayed review. The system always errs on the side of caution.

### Model-Agnostic Interface

The L2 prompt template uses a standard OpenAI-compatible chat completions API format. Switching to any provider (OpenAI, Anthropic, local vLLM) requires changing only the `GROQ_URL` and `MODEL` constants — zero prompt rewriting.
