import re

def patch():
    with open('src/l2_copilot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    new_code = """
def call_l2(client: "genai.Client", record: dict, model: str) -> L2Decision:
    account_id = record.get("account_id", "UNKNOWN")
    
    # Let's wrap the prompt to allow tool usage
    system_instruction = f"{SYSTEM_PROMPT}\\n\\nAccount audit record:\\n{json.dumps(record, indent=2, default=str)}"
    
    tool_prompt = \"\"\"
You are an Agentic Copilot. You may gather more evidence before deciding.
If you need to check who an account transacted with, reply EXACTLY and ONLY with this JSON:
{"tool": "query_counterparties", "account_id": "<ID>"}

If you have enough information, reply with your final decision using the exact JSON schema requested.
\"\"\"
    
    history = system_instruction + tool_prompt
    tool_calls = 0
    max_tools = 2
    
    while tool_calls < max_tools:
        # We don't enforce response_schema here to allow the tool JSON
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
            
            # Read-only Mock Graph Query
            import random
            random.seed(hash(target_acc))
            partners = [f"ACC_{random.randint(1000, 9999)}" for _ in range(random.randint(2, 6))]
            
            if "agent_audit_trail" not in record:
                record["agent_audit_trail"] = []
            record["agent_audit_trail"].append(f"query_counterparties({target_acc}) -> {partners}")
            
            tool_result = f"\\nTool Result: Counterparties for {target_acc} are {partners}. Now make your decision."
            history += "\\n" + response.text + tool_result
            tool_calls += 1
        else:
            # It's likely the final decision
            try:
                return L2Decision(**data)
            except Exception as e:
                # If schema fails, force final via response_schema below
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
"""
    
    # Replace the existing call_l2 function
    # It starts at 'def call_l2' and goes up to 'def main()'
    pattern = re.compile(r'def call_l2.*?def main\(\):', re.DOTALL)
    content = pattern.sub(new_code.strip() + '\n\n\ndef main():', content)
    
    with open('src/l2_copilot.py', 'w', encoding='utf-8') as f:
        f.write(content)

patch()
