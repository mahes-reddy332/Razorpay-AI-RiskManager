import json
import os
import time
import requests
from sklearn.model_selection import train_test_split
from model_v2 import extract_features, apply_frozen_config, FROZEN_CONFIG
from audit import FraudRiskAuditor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-api-key-here")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

print("Extracting features for L2 measurement...")
df = extract_features()

all_accounts = df.index.values
y = df['is_mule'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_accounts, y, test_size=0.20, random_state=42)
test_df = df.loc[X_test].copy()

scores = apply_frozen_config(test_df, FROZEN_CONFIG)
test_df['final_score'] = scores

manual_review_df = test_df[(test_df['final_score'] >= FROZEN_CONFIG['manual_thresh']) & 
                           (test_df['final_score'] < FROZEN_CONFIG['dec_thresh'])]

print(f"\nFound {len(manual_review_df)} accounts in MANUAL_REVIEW band in the strictly split Test Set.")
print(f"  Mules in review: {sum(manual_review_df['is_mule'])}")
print(f"  Legitimate in review: {len(manual_review_df) - sum(manual_review_df['is_mule'])}")

auditor = FraudRiskAuditor('data/accounts.csv', 'data/transactions.csv')
auditor.build_graph()

SYSTEM_PROMPT = """You are a Level 2 fraud review assistant for a UPI mule-account detection system.
You will be given the account's audit record: graph topology, transaction history, and metadata.
Make a final FRAUD / SAFE / UNCERTAIN call using the kind of judgment the deterministic rules can't apply.
Reply ONLY with a JSON array of objects, exactly in this format: [{"account_id": "ACC_XYZ", "decision": "FRAUD"}]. No markdown, just raw JSON."""

results = {}
batch_size = 5
account_ids = manual_review_df.index.tolist()

print("\nQuerying L2 (Groq/LLaMA3-70b) in batches...")
for i in range(0, len(account_ids), batch_size):
    batch_ids = account_ids[i:i+batch_size]
    batch_records = [auditor.audit_account(acc) for acc in batch_ids]
    
    prompt = f"Evaluate these records:\n{json.dumps(batch_records, indent=2, default=str)}"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    resp = requests.post(GROQ_URL, headers=headers, json=data)
    if resp.status_code == 200:
        try:
            content = resp.json()['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            decisions = json.loads(content)
            for d in decisions:
                results[d["account_id"]] = d["decision"]
            print(f"  Processed batch {i//batch_size + 1}: Success")
        except Exception as e:
            print(f"  Processed batch {i//batch_size + 1}: Parse error. Raw: {content}")
    else:
        print(f"  Processed batch {i//batch_size + 1}: API Error {resp.status_code} - {resp.text}")
    
    time.sleep(2.5) # Groq rate limit safety

# Combined Metric
final_y_pred = []
l2_correct_mules = 0
l2_incorrect_legit = 0

for acc, score in zip(X_test, scores):
    if score >= FROZEN_CONFIG['dec_thresh']:
        final_y_pred.append(1)
    elif score >= FROZEN_CONFIG['manual_thresh']:
        dec = results.get(acc, "UNCERTAIN")
        pred = 1 if dec == "FRAUD" else 0
        final_y_pred.append(pred)
        
        is_actually_mule = df.loc[acc, 'is_mule']
        if is_actually_mule and pred == 1: l2_correct_mules += 1
        if not is_actually_mule and pred == 1: l2_incorrect_legit += 1
    else:
        final_y_pred.append(0)

p = precision_score(y_test, final_y_pred, zero_division=0)
r = recall_score(y_test, final_y_pred, zero_division=0)
f1 = f1_score(y_test, final_y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, final_y_pred).ravel()

print("\n" + "="*50)
print("  ACTUAL MEASURED L1+L2 COMBINED SYSTEM PERFORMANCE (TEST SET ONLY)")
print("="*50)
print(f"L2 Caught {l2_correct_mules} out of {sum(manual_review_df['is_mule'])} manual review mules.")
print(f"L2 Incorrectly flagged {l2_incorrect_legit} out of {len(manual_review_df) - sum(manual_review_df['is_mule'])} legitimate manual review accounts.")
print("-"*50)
print(f"Precision : {p:.3f}  ({p*100:.1f}%)")
print(f"Recall    : {r:.3f}  ({r*100:.1f}%)")
print(f"F1 Score  : {f1:.3f}")
print(f"True Pos  : {tp}")
print(f"False Pos : {fp}")
print(f"True Neg  : {tn}")
print(f"False Neg : {fn}")
print("="*50)
