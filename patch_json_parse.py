import re

def fix():
    with open('src/l2_copilot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    old_parse = """        try:
            data = json.loads(response.text)
        except:
            break"""
            
    new_parse = """        try:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
        except Exception as e:
            # print("JSON Parse Error:", e, response.text)
            break"""

    content = content.replace(old_parse, new_parse)
    with open('src/l2_copilot.py', 'w', encoding='utf-8') as f:
        f.write(content)

fix()
