import re

def fix():
    with open('src/l2_copilot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # The newlines got rendered literally, let's just do a clean replace for the whole call_l2 block
    old_call_l2 = re.search(r'def call_l2.*?def main\(\):', content, re.DOTALL).group(0)
    
    clean_call_l2 = """def call_l2(client: "genai.Client", record: dict, model: str) -> L2Decision:
    account_id = record.get("account_id", "UNKNOWN")
    
    # We use r-strings or explicit concat safely
    system_instruction = SYSTEM_PROMPT + "\\n\\nAccount audit record:\\n" + json.dumps(record, indent=2, default=str)
    
    tool_prompt = '''
You are an Agentic Copilot. You may gather more evidence before deciding.
If you need to check who an account transacted with, reply EXACTLY and ONLY with this JSON:
{"tool": "query_counterparties", "account_id": "<ID>"}

If you have enough information, reply with your final decision using the exact JSON schema requested.
'''
    
    history = system_instruction + tool_prompt
    tool_calls = 0
    max_tools = 2
    
    while tool_calls < max_tools:
        response = client.models.generate_content(
            model=model,
            contents=history,
            config={"response_mime_type": "application/json"}
        )
        
        try:
            data = json.loads(response.text)
        except:
            break
            
        if "tool" in data and data["tool"] == "query_counterparties":
            target_acc = data.get("account_id", account_id)
            print(f"  [AGENT] Using tool: query_counterparties({target_acc})")
            
            import random
            random.seed(hash(target_acc))
            partners = [f"ACC_{random.randint(1000, 9999)}" for _ in range(random.randint(2, 6))]
            
            if "agent_audit_trail" not in record:
                record["agent_audit_trail"] = []
            record["agent_audit_trail"].append(f"query_counterparties({target_acc}) -> {partners}")
            
            tool_result = "\\nTool Result: Counterparties for " + target_acc + " are " + str(partners) + ". Now make your decision."
            history += "\\n" + response.text + tool_result
            tool_calls += 1
        else:
            try:
                return L2Decision(**data)
            except Exception as e:
                break

    # Force final decision with strict schema
    response = client.models.generate_content(
        model=model,
        contents=history + "\\n\\nYou must now output your final decision.",
        config={
            "response_mime_type": "application/json",
            "response_schema": L2Decision,
        },
    )
    return response.parsed

def main():"""
    
    content = content.replace(old_call_l2, clean_call_l2)
    with open('src/l2_copilot.py', 'w', encoding='utf-8') as f:
        f.write(content)

fix()
