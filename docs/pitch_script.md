# Pitch Script Outline: UPI Fraud Flow Tracer

**Speaking Time Estimate:** ~4.5 to 5 minutes (approx. 660 words)
**Shot List / Section Outline:**
*   0:00-0:30 — **The Problem:** The operational bottleneck of false positives in mule detection.
*   0:30-1:00 — **Our Approach:** Explainable, in-memory temporal graphs over GNNs.
*   1:00-1:30 — **Live Demo:** Visualizing a Split-and-Reconverge mule flow.
*   1:30-2:30 — **Honest Metrics & Methodology:** The v1 leakage audit and the v2 fix.
*   2:30-3:00 — **The False Negative Story:** How a compromised merchant slipped through, proving the need for L2.
*   3:00-3:45 — **Graceful Failure:** The Phase 5 audit log demo.
*   3:45-4:30 — **Future Scope:** The Risk Waterfall & Level 2 LLM Copilot proposal.
*   4:30-4:40 — **Close.**

---

## 1. The Problem
*   "Hi everyone. As UPI scales, so does the sophistication of mule account networks. The problem isn't just catching mules; it's catching them without blocking legitimate users."
*   "A naive rule engine might boast high recall, but if its false positive rate is high, you freeze innocent small businesses and payroll accounts. That's an operational bottleneck."

## 2. Our Approach
*   "For Track 2, we built the UPI Fraud Flow Tracer. We chose a deterministic, in-memory graph approach using NetworkX rather than a black-box GNN."
*   "Regulators require explainability. We process transactions into a temporal graph, flag suspicious velocity, and then recursively trace the money hop-by-hop through time to understand the topology."

## 3. Live Demo (The Flow)
*   *(Show `mule_flow.png` on screen)*
*   "Here is a detected Split-and-Reconverge topology. Victim funds fan out across first-hop mules, before reconverging at an aggregator and cashing out. Our BFS tracer reconstructs this flow strictly by following temporal edge constraints."

## 4. Honest Methodology and Metrics
*   *(Show `metrics_summary.png` on screen)*
*   "We want to be transparent about our methodology. During early development, we hit 85% precision, but an internal audit revealed this was artificially high due to test-set threshold tuning and deterministic feature generation."
*   "We corrected this in our v2 model. We injected probabilistic noise into the Merchant Category Codes (modeling compromised safe merchants) and strictly tuned our composite weights on a validation split, evaluating only once on a blind test set."
*   "The honest result: our v2 model achieves 74.1% Precision and 95.2% Recall. We made a deliberate trade-off, accepting slightly lower recall to aggressively protect legitimate users, reducing False Positives down to just 7 in the test set."
*   "It is important to note that these results are on synthetic data. Because we designed both the generator and the detector, performance in a live environment would require re-calibration against organic noise."

## 5. The False Negative (Understanding our limits)
*   "In our honest test run, exactly one mule chain slipped through our defenses—resulting in 1 false negative, versus 0 previously. This is an honest trade-off from removing the deterministic leak, not a regression."
*   "Upon inspection, this mule account successfully routed its cash-out through an MCC registered as a Hospital. Because our rules dynamically weigh topology and allow leeway for safe MCCs to protect legitimate businesses, this sophisticated evasion tactic worked."
*   "This isn't a bug; it's a real-world pattern where fraud rings use compromised safe merchant accounts to launder funds. And this specific false negative is exactly why we designed our future roadmap."

## 6. Graceful Failure (The Audit Log)
*   *(Show terminal output of `python src/audit.py`)*
*   "Before talking about that roadmap, I want to show resilience. What happens if upstream logs break and we get a transaction with a missing timestamp?"
*   "Instead of crashing, our FraudRiskAuditor traps the exception, aborts the score, and generates a JSON audit log marking the account for `MANUAL_REVIEW_REQUIRED` due to `INSUFFICIENT_DATA`."

## 7. The Risk Waterfall & Level 2 LLM Copilot
*   "If our false-positive rate held constant at 100,000 legitimate users, we would flag around 1,290 innocent accounts. While error rates rarely scale perfectly linearly in practice, this is a 50% improvement over our baseline MVP rules."
*   "To handle those remaining edge cases—like the compromised Hospital cash-out or the corrupted timestamps—we built a Level 1 / Level 2 Architecture."
*   "Level 1 is what you've seen: The Real-Time Graph executing in milliseconds."
*   "Level 2 is our LLM Copilot (`l2_copilot.py`). Instead of a human analyst, we implemented a standalone Pydantic script that uses Gemini 2.5 Flash to asynchronously read our structured JSON audit logs, reasoning through conflicting metadata to catch subtle behavioral anomalies that rigid rules miss."

## 8. Close
*   "Explainable, operationally safe, methodologically honest, and powered by a working LLM Copilot. Thank you."
