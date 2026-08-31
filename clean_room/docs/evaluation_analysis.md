# Evaluation & False-Positive Cost Analysis

## Phase 3 MVP Results (Held-out Test Set)
An 80/20 strict account-level split was used to evaluate the MVP rules, ensuring no data leakage between train and test scenarios. 

**Test Set Composition:**
- Total Scorable Nodes: 808
- Legit Accounts: 782
- True Mule Accounts: 26

**Performance (Threshold > 0.5):**
- **Recall:** 1.000 (26/26 mules caught)
- **Precision:** 0.788 (26 true positives, 7 false positives)
- **F1-Score:** 0.881

*(The Precision-Recall curve has been generated and saved to `outputs/pr_curve.png`).*

## The Business Reality of False Positives
While an F1-score of 0.88 is theoretically strong, blindly optimizing F1 obscures the asymmetric real-world costs of our errors.

### The Cost of a False Negative (FN)
When a mule chain successfully evades detection, the primary costs are:
1. **Regulatory Sanctions:** Failure to comply with mandates like RBI's MuleHunter.AI attracts steep fines and limits the institution's ability to onboard new users.
2. **Fraud Losses & Liability:** Victim funds are lost to the final cash-out point, potentially leading to chargebacks and reputational damage.
3. **Network Contagion:** A successful mule network often scales up, utilizing the same topologies repeatedly until caught.

### The Cost of a False Positive (FP)
Our MVP flagged **7 legitimate users** as mules. In a real-world system processing 10 million active accounts, a false positive rate matching this test set (~0.9%) means **90,000 legitimate users are unjustly blocked**.

The real cost here includes:
1. **Business Interruption:** Small Businesses (SMBs) consolidating daily earnings to pay suppliers suddenly have frozen capital. Missing a vendor payment damages their business and fractures their trust in our platform.
2. **Human Suffering:** Salary disbursers flagged for "structuring" or "fan-out" mean everyday employees do not get their paychecks on time, which inevitably leads to severe RBI ombudsman complaints.
3. **Operational Overhead (L1 Support):** 90,000 falsely blocked users result in 90,000 urgent support tickets. If manual review takes 15 minutes per case, that is 22,500 hours of L1 investigation time—an unsustainable operational expense.

## Conclusion: Do We Accept the MVP?
The MVP's 100% recall proves the baseline rules (velocity + dormancy) are effective detectors, but the 78.8% precision requires improvement. We need a way to distinguish an SMB from a mule aggregator without dropping our recall.

**Proposed Next Step:** Context-aware multi-hop tracing (Phase 4). By walking the graph to see if the funds *eventually* land in a known cash-out signature vs. a legitimate merchant, we can safely un-flag those SMBs and push Precision up towards 95% without sacrificing Recall.
