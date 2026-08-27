# UPI Fraud Flow Tracer (Mule Account Detector)

**Razorpay AI Buildathon 2026 - Track 2 (AI Risk Manager)**

This repository contains a defense-only UPI Fraud Flow Tracer. It detects mule account networks by reconstructing multi-hop transaction flows over a temporal graph. Designed for real-world compliance (e.g., RBI's MuleHunter.AI), it prioritizes explainability and the strict minimization of False Positives (protecting legitimate businesses from unfair account freezes).

---

## Tech Stack
*   **Core Logic:** Python 3.11
*   **Data Generation:** Pandas, Faker (Synthetic realistic data generation)
*   **Graph Engine:** NetworkX (In-memory topological tracing)
*   **Visualizations:** Matplotlib
*   **Testing:** Pytest
*   **No Black Boxes:** Explicitly avoided opaque Graph Neural Networks (GNNs) or heavy Graph Databases (Neo4j) to ensure regulatory explainability and low operational overhead.

---

## Project Memory: Phase-by-Phase Breakdown

This project was built incrementally:

### Phase 1: Synthetic Data Simulation (`src/simulator.py`)
*   **Implementation:** Generated 5,000 accounts and 29,000+ transactions over 90 days.
*   **Focus:** Realistic class imbalance (~3% mules). We engineered complex mule archetypes (Structuring, Split/Re-converge) and Hard Negatives (Salary Disbursers, Small Businesses, Dormant legit purchases).

### Phase 2 & 3: MVP Detector & Formal Evaluation (`src/evaluate.py`)
*   **Implementation:** A baseline rule engine checking for high "pass-through" velocity (>90% within 24h) and account dormancy (>30 days).
*   **Result:** High recall, but high False Positives, as it mistakenly froze legitimate Small Businesses and Payroll accounts.

### Phase 4: Temporal Graph Tracer (`src/tracer.py`)
*   **Implementation:** Added a Bidirectional Breadth-First Search (BFS). Reconstructs the entire chain within a 24-hour window. 
*   **Result:** By analyzing graph topology, the system recognized that legitimate SMBs have massive "fan-in/fan-out" topologies.

### Phase 5: Graceful Degradation & Audit (`src/audit.py`)
*   **Implementation:** Wrapped the engine in a `FraudRiskAuditor`. Injected corrupted data (missing timestamps/dates).
*   **Result:** The system degrades gracefully, outputting a `MANUAL_REVIEW_REQUIRED` decision inside a structured JSON audit log (`outputs/audit_log.json`) with plain-English explanations.

### Phase 6: Visualizations (`src/visualize.py`)
*   **Implementation:** Auto-generated the topological flow diagram and exported evaluation metric tables.

### Phase 7: Packaging & Testing (`tests/test_detector.py`)
*   **Implementation:** Wrote a `pytest` suite validating the core BFS tracer.

### Phase 8: Metadata Contextualization & Honest Evaluation (`src/model_v2.py`)
*   **Implementation:** Addressed methodological flaws in early evaluation versions. Re-generated data with probabilistic Merchant Category Codes (MCC) to simulate noise (e.g., compromised legitimate merchants). Re-split data into a strict 60/20/20 Train/Validation/Test split. Tuned weights exclusively on validation data.

---

## Evaluation & Metrics (Final Phase 8)

We present two sets of numbers. The v1 numbers represent our initial, flawed evaluation methodology. The v2 numbers represent the final, honest evaluation.

**Test Set Composition (v2):** Evaluated on a strictly held-out test split of 563 accounts (21 Mules, 542 Legitimate). *Note: Accounts with deliberately corrupted/missing data (Phase 5) degrade to a `MANUAL_REVIEW_REQUIRED` state and are excluded from the binary TP/FP/TN/FN metrics as an "abstain" category.*

| Model | Precision | Recall | F1 Score | False Positives |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 3 (MVP Baseline)** | 54.5% | 85.7% | 0.667 | 15 |
| **Phase 8 v1 (Invalidated - Tuned on Test Set)** | 85.2% | 100.0% | 0.920 | 4 |
| **Phase 8 v2 (Final - Honest 60/20/20 Split)** | 74.1% | 95.2% | 0.833 | 7 |

*Methodology Note:* The v1 model achieved high performance due to data leakage (thresholds were nudged based on test-set visibility) and deterministic feature generation (MCCs perfectly correlated with labels). The v2 model fixes this by applying probabilistic noise to MCCs and tuning strictly on a validation set. The v2 results include 1 false negative, versus 0 previously — an honest trade-off from removing the deterministic MCC leak, not a regression.

### Extrapolating the False Positive Rate (FPR)
Our final v2 model achieved a 1.29% False Positive Rate on the test set (7 false positives out of 542 legitimate users). If this false-positive rate held constant in a production environment compared to a standard rule engine:

*   **At 10,000 legitimate users:** We flag ~129 innocent accounts (Baseline MVP would flag ~270).
*   **At 100,000 legitimate users:** We flag ~1,290 innocent accounts (Baseline MVP would flag ~2,700).

*Note: This is a linear extrapolation from a small sample. In practice, error rates rarely scale perfectly linearly due to changing distributions and long-tail behaviors at scale.*

### Known Limitations: Synthetic Data
The results reported above are evaluated on synthetic data generated for this Buildathon. Results on synthetic data will inherently look cleaner than production data because the same engineering team designed both the generator (`simulator.py`) and the detector. The model benefits from knowing the precise structural boundaries (e.g., 90-day windows, fixed archetypes) defined by the generator. Performance in a live environment would require re-calibration against organic noise.

---

## How to Run the Pipeline

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Generate Data:** `python src/simulator.py`
3. **Execute V2 Pipeline:** `python src/model_v2.py`
4. **Generate Audit Logs:** `python src/audit.py`
5. **Generate Visualizations:** `python src/visualize.py`
6. **Run Tests:** `pytest tests/test_detector.py`

---

## Future Scope & Scalability: The Risk Waterfall & Level 2 LLM Copilot

To scale this engine, we implemented a hybrid Level 1 / Level 2 Architecture:

**Level 1: The Real-Time Graph Switch (Implemented in `src/model_v2.py`)**
Executes in milliseconds using NetworkX and a composite weighted score (Velocity, Topology, MCC).

**Level 2: The LLM Copilot (Implemented in `src/l2_copilot.py`)**
For the ambiguous edge cases or corrupted data logs (as engineered in Phase 5), Level 1 marks the account as `MANUAL_REVIEW_REQUIRED`. Instead of a human wasting time, our standalone `l2_copilot.py` asynchronously reads the structured `audit_log.json`, reasons across unstructured metadata combinations, and generates a strict, Pydantic-validated `FRAUD/SAFE` recommendation using Google's Gemini 2.5 Flash.
