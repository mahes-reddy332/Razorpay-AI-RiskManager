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
*   "The honest result: our v2 model achieves 81.5% L1 Recall: 91.1% Combined Recall on the test set. We swept 4,860 parameter combinations across six features — topology, MCC, Z-scores, betweenness centrality, and multi-window velocity — and the grid search honestly converged on topology as the single dominant signal. The other features are implemented as production infrastructure but did not improve separation on our synthetic data."
*   "It is important to note that these results are on synthetic data. Performance in a live environment would require re-calibration against organic noise."

## 5. Adversarial Testing & Feature Infrastructure
*   "We didn't just test against naive synthetic mules; we ran a Round 2 stress test. We injected adversarial mules designed specifically to camouflage their amounts, disguise their topologies, or delay transfers up to 120 hours to evade our windows."
*   "When we ran our 4,860-combination grid search against these evaders, something fascinating happened. Our velocity and Z-score signals stayed at zero weight because the fraudsters successfully bypassed the thresholds. But our **MCC metadata feature** woke up, jumping from 0.0 to a heavy 0.6 weight."
*   "When topology failed against these evaders, the machine learning algorithm dynamically shifted its reliance to metadata. We decoupled our MCC risk check from our expensive graph trace so that it acts as a cheap, always-on check. The algorithm learned to catch slow mules directly via their risky counterparties."
*   "We even took this further by stress-testing our deterministic engine against a massive 5-million transaction, 400,000-account real-world benchmark (the IBM AML dataset). When we extended our graph traversal to a 15-hop horizon and decoupled extreme topologies ($D \ge 15$), our total detection rate surged to **73.8%**."
*   "Crucially, under real-world banking safeguards, high graph connectivity alone is never grounds for an immediate account freeze. We architected a strict operational boundary: only accounts with full volume-retention corroboration are contained for risk (catching 49.0% of mules), while the additional 24.8% of mules caught by extreme topology are safely routed to the **Level 2 Review Queue**. Exactly **0%** of the 11,327 high-degree legitimate clearing accounts were contained for risk. High recall via decoupled topology feeds the review queue, not the risk containment path."
*   "We also tested an 'Ultimate Evader' combining wide topology camouflage with a fraudulently registered safe MCC. While they successfully evaded our L1 automatic freeze, they scored 0.60 — landing perfectly in our Manual Review band. This is exactly why we built the Level 2 LLM Copilot."
*   "But we want to be fully transparent about a genuine blind spot we discovered. If a fraud ring uses a multi-hop chain where *every single node* delays transfers past 72 hours, and the risky MCC is hidden several hops deep, our cheap immediate check will miss it. Catching multi-hop slow-walked laundering requires a genuine Level 3: a periodic batch sweep that runs heavy graph traces offline. We optimized for scale, closed the cheap gaps, and know exactly what is still open and why."

## 6. Graceful Failure (The Audit Log)
*   *(Show terminal output of `python src/audit.py`)*
*   "Before talking about that roadmap, I want to show resilience. What happens if upstream logs break and we get a transaction with a missing timestamp?"
*   "Instead of crashing, our FraudRiskAuditor traps the exception, aborts the score, and generates a JSON audit log marking the account for `MANUAL_REVIEW_REQUIRED` due to `INSUFFICIENT_DATA`."

## 7. The Risk Waterfall & Level 2 LLM Copilot
*   "On our test set, we have a ~4.0% False Positive rate. If this rate held constant at scale, we would flag ~404 innocent accounts per 10,000. At 100,000, it's ~4,040. At 1 million legitimate users, it's over 40,000 false positives. While error rates rarely scale perfectly linearly in practice, large absolute false-positive counts at scale are an expected property of any high-recall real-time fraud system, not a defect specific to ours."
*   "This is precisely why production fraud operations use tiered human review rather than expecting a fully automated layer to be perfect. And it's exactly why this system is architected as an L1 auto-clear, routing to an L2 LLM review, backed by a documented L3 batch sweep, instead of a single monolithic classifier."
*   "To handle those remaining edge cases—like the compromised Hospital cash-out or the corrupted timestamps—we built a Level 1 / Level 2 Architecture."
*   "Level 1 is what you've seen: The Real-Time Graph executing in milliseconds."
*   "Level 2 is our LLM Copilot (`l2_copilot.py`). Instead of a human analyst, we implemented a standalone Pydantic script that uses Gemini 2.5 Flash to asynchronously read our structured JSON audit logs, reasoning through conflicting metadata to catch subtle behavioral anomalies that rigid rules miss."

## 8. Close
*   "Explainable, operationally safe, methodologically honest, and powered by a working LLM Copilot. Thank you."
