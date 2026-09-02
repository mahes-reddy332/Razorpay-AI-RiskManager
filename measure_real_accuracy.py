import json
import os
import time
import pandas as pd
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
import random

# Load ground truth
accounts_df = pd.read_csv('data/accounts.csv')
ground_truth = dict(zip(accounts_df['account_id'], accounts_df['is_mule']))

# Load audit logs and filter for MANUAL_REVIEW
with open('outputs/audit_log.json', 'r') as f:
    full_audit = json.load(f)

# Find true MANUAL_REVIEW_REQUIRED accounts (or just pick those that aren't cleanly 1.0 or 0.0)
manual_review_cases = [r for r in full_audit if r.get('decision') == 'MANUAL_REVIEW_REQUIRED']
if not manual_review_cases:
    # Fallback: take some that aren't 1.0
    manual_review_cases = [r for r in full_audit if r.get('score', 0) < 1.0]

# Sample 20 cases exactly
random.seed(42)
sample_cases = random.sample(manual_review_cases, min(20, len(manual_review_cases)))

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

class L2Decision(BaseModel):
    decision: Literal["FRAUD", "SAFE", "UNCERTAIN"]
    confidence: float
    reasoning: str
    key_signal: str

SYSTEM_PROMPT = """You are a Level 2 fraud review assistant for a UPI mule-account detection system.
You will be given the account's audit record. Make a final FRAUD / SAFE / UNCERTAIN call."""

def run_non_agentic(record):
    prompt = f"{SYSTEM_PROMPT}\n\nAccount audit record:\n{json.dumps(record, indent=2, default=str)}\n\nYou must output exactly your final decision JSON."
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": L2Decision}
        )
        time.sleep(4) # Rate limit respect
        return resp.parsed
    except Exception as e:
        time.sleep(10)
        return None

def run_agentic(record):
    account_id = record.get("account_id")
    tool_prompt = """\n\nIf you need to check who an account transacted with, reply EXACTLY and ONLY with this JSON:
{"tool": "query_counterparties", "account_id": "<ID>"}
If you have enough information, reply with your final decision using the exact JSON schema requested."""

    history = f"{SYSTEM_PROMPT}\n\nAccount audit record:\n{json.dumps(record, indent=2, default=str)}" + tool_prompt
    
    tool_used = False
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=history,
                config={"response_mime_type": "application/json"}
            )
            time.sleep(4) # Rate limit
        except Exception:
            time.sleep(10)
            break
            
        try:
            text = resp.text.strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            data = json.loads(text.strip())
        except:
            break
            
        if "tool" in data and data["tool"] == "query_counterparties":
            tool_used = True
            target = data.get("account_id")
            # Mock graph query since we don't have the Neo4j/full CSV topology graph logic here instantly available
            # Wait! Let's get real counterparties from transactions.csv if we want to be 100% honest!
            # Since that's heavy, we'll just simulate a read-only response, but to measure the tool's effectiveness honestly, 
            # we should provide accurate data. Let's load transactions.csv.
            pass
        else:
            try:
                return L2Decision(**data), tool_used
            except:
                break
                
    # Force final
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=history + "\n\nYou must now output your final decision.",
            config={"response_mime_type": "application/json", "response_schema": L2Decision}
        )
        time.sleep(4)
        return resp.parsed, tool_used
    except:
        return None, tool_used


# To be perfectly honest, providing random counterparties ruins the tool's validity. 
# We must load real counterparties.
print("Loading real counterparties from transactions.csv...")
tx_df = pd.read_csv('data/transactions.csv')
def get_real_counterparties(acc_id):
    sent = tx_df[tx_df['sender_account'] == acc_id]['receiver_account'].tolist()
    recv = tx_df[tx_df['receiver_account'] == acc_id]['sender_account'].tolist()
    return list(set(sent + recv))[:5] # cap at 5 for prompt size

results = []
print(f"Evaluating {len(sample_cases)} cases...")
for idx, case in enumerate(sample_cases):
    acc = case['account_id']
    truth = "FRAUD" if ground_truth.get(acc, False) else "SAFE"
    
    # Non agentic
    na_dec = run_non_agentic(case)
    na_val = na_dec.decision if na_dec else "ERROR"
    
    # Agentic
    # We must patch run_agentic to use real counterparties
    # Redefine it here to close over get_real_counterparties
    history = f"{SYSTEM_PROMPT}\n\nAccount audit record:\n{json.dumps(case, indent=2, default=str)}\n\nIf you need to check who an account transacted with, reply EXACTLY and ONLY with this JSON:\n{{\"tool\": \"query_counterparties\", \"account_id\": \"<ID>\"}}\nIf you have enough information, reply with your final decision using the exact JSON schema requested."
    
    tool_used = False
    ag_val = "ERROR"
    for attempt in range(2):
        try:
            resp = client.models.generate_content(model=MODEL, contents=history, config={"response_mime_type": "application/json"})
            time.sleep(4)
            text = resp.text.strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            data = json.loads(text.strip())
            
            if "tool" in data and data["tool"] == "query_counterparties":
                tool_used = True
                target = data.get("account_id")
                partners = get_real_counterparties(target)
                history += f"\n{resp.text}\nTool Result: Counterparties are {partners}."
            else:
                ag_val = L2Decision(**data).decision
                break
        except:
            break
            
    if ag_val == "ERROR" or (tool_used and ag_val == "ERROR"):
        try:
            resp = client.models.generate_content(model=MODEL, contents=history + "\n\nYou must now output your final decision.", config={"response_mime_type": "application/json", "response_schema": L2Decision})
            time.sleep(4)
            ag_val = resp.parsed.decision
        except:
            pass

    results.append({
        'account_id': acc,
        'ground_truth': truth,
        'non_agentic': na_val,
        'agentic': ag_val,
        'tool_used': tool_used
    })
    print(f"[{idx+1}/{len(sample_cases)}] {acc} | Truth: {truth} | NA: {na_val} | Agentic: {ag_val} | Tool: {tool_used}")

# Analysis
print("\n--- RESULTS ---")
na_correct = sum(1 for r in results if r['non_agentic'] == r['ground_truth'])
ag_correct = sum(1 for r in results if r['agentic'] == r['ground_truth'])

print(f"Non-Agentic Accuracy: {na_correct}/{len(results)} ({na_correct/len(results):.1%})")
print(f"Agentic Accuracy:     {ag_correct}/{len(results)} ({ag_correct/len(results):.1%})")

tools = [r for r in results if r['tool_used']]
no_tools = [r for r in results if not r['tool_used']]

if tools:
    t_na_correct = sum(1 for r in tools if r['non_agentic'] == r['ground_truth'])
    t_ag_correct = sum(1 for r in tools if r['agentic'] == r['ground_truth'])
    print(f"\nOf the {len(tools)} cases that USED the tool:")
    print(f"  Non-Agentic was correct on {t_na_correct}")
    print(f"  Agentic was correct on {t_ag_correct}")
else:
    print("\nNo cases used the tool.")

if no_tools:
    nt_na_correct = sum(1 for r in no_tools if r['non_agentic'] == r['ground_truth'])
    nt_ag_correct = sum(1 for r in no_tools if r['agentic'] == r['ground_truth'])
    print(f"\nOf the {len(no_tools)} cases that DID NOT use the tool:")
    print(f"  Non-Agentic was correct on {nt_na_correct}")
    print(f"  Agentic was correct on {nt_ag_correct}")
