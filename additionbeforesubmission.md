# Final Review Checklist (Pending)

## 1. Ground-Truth Accuracy Measurement (Agentic L2)

**Why:** The L2 LLM now acts as an agent with bounded tool access (it can query counterparties). We must rigorously prove whether this tool access actually improves ground-truth accuracy or just increases LLM confidence. 

**What is built:** We created the script `measure_real_accuracy.py`. It pulls the actual `is_mule` labels from `data/accounts.csv`, pulls real transaction counterparties from `data/transactions.csv`, and runs a 20-account sample of `MANUAL_REVIEW_REQUIRED` accounts through both the baseline prompt (no tools) and the agentic prompt (with tools).

**Action (Post-Submission / Live Pitch Prep):**
- [ ] Ensure the Gemini API key has quota available (or use a paid tier to bypass the 15 RPM / daily limits).
- [ ] Run the evaluation script: `python measure_real_accuracy.py`
- [ ] Record the breakdown output to see the exact accuracy percentage of cases where the LLM used the tool versus when it did not.
