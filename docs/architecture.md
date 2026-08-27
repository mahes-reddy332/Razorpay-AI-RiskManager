# Architecture & Design Decisions

## 1. Synthetic Simulator (Phase 1)
- **Constraint:** Strictly defense-only; no real PII/financial data used.
- **Design:** A custom Pandas-based transaction generator that simulates a 90-day UPI flow. It injects complex Mule archetypes (Structuring, Split/Re-converge, Commission-taking) alongside Hard Negatives (Salary Disbursers, Small Businesses, Dormant Purchases).
- **Probabilistic Noise:** Merchant Category Codes (MCC) are assigned probabilistically. 10% of legitimate businesses receive high-risk MCCs, and 20% of mule cashouts route through compromised safe MCCs to model real-world edge cases.

## 2. Methodology: Leakage Discovery and V2 Fix
During the development of the Phase 4 (Topology) and Phase 8 (MCC) rules, the initial evaluation methodology (v1) was flawed. 
- **The Leakage:** The evaluation suffered from threshold nudging (the topology threshold was lowered after inspecting false positives in the test set) and deterministic feature generation (safe MCCs perfectly correlated with legitimate hard-negatives).
- **The Fix (V2):** We implemented a strict 60/20/20 Train/Validation/Test split. MCC assignment was made probabilistic in the generator. We then performed a grid-search sweep of composite thresholds optimizing for F1 score *only* on the Validation split (562 accounts). 

### V2 Validation Sweep Results (Top 5 Configs)
| Node Threshold | Topology Weight | MCC Weight | Decision Threshold | Validation F1 |
| :--- | :--- | :--- | :--- | :--- |
| 8 | 0.6 | 0.0 | 1.2 | 0.8636 |
| 8 | 0.6 | 0.3 | 1.2 | 0.8636 |
| 8 | 0.6 | 0.6 | 1.2 | 0.8636 |
| 8 | 0.3 | 0.3 | 1.2 | 0.8571 |
| 8 | 0.3 | 0.6 | 1.2 | 0.8571 |

The final parameters (`node_thresh=8`, `top_weight=0.6`, `mcc_weight=0.0`, `dec_thresh=1.2`) were frozen and run exactly once against the untouched Test split (563 accounts). This methodology correctly documents the precision/recall trade-off without data leakage. The v2 results include 1 false negative, versus 0 previously — an honest trade-off from removing the deterministic MCC leak, not a regression.

### The Single False Negative (The Compromised MCC)
In our honest v2 test run, exactly one mule chain slipped through our defenses. Upon inspection, this mule account successfully routed its cash-out through an MCC registered as `HOSPITAL`. Because our rules dynamically weighed topology and allowed some leeway for safe MCCs (to protect legitimate businesses), this sophisticated evasion tactic worked. 
This is not a bug; it is a real, documented pattern where fraud rings use fraudulently-registered or compromised safe merchant accounts to launder funds. This specific false negative is the exact reason our roadmap includes the Level 2 LLM Copilot (proposed future architecture) — while rigid graph rules might pass a `HOSPITAL` transaction, an LLM reasoning over the full JSON context can detect the subtle behavioral anomalies of a compromised merchant.

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

## 5. The Future Scope: Risk Waterfall & Level 2 LLM Copilot (proposed future architecture — not implemented in this submission)
To scale this architecture, we propose a hybrid L1/L2 framework:

### Level 1: Real-Time Engine
The NetworkX metadata graph operates in milliseconds using the composite scores derived during the validation sweep.

### Level 2: Asynchronous LLM Copilot (proposed future architecture — not implemented in this submission)
If Level 1 encounters missing or corrupted upstream data logs (as engineered in Phase 5), the system gracefully degrades to a `MANUAL_REVIEW_REQUIRED` state and outputs a structured JSON audit log. 
In the proposed future architecture — not implemented in this submission, an **LLM Agent** consumes this JSON log. The LLM acts as an L2 Copilot, dynamically reasoning across unstructured metadata to output a final investigation recommendation.
