import re

def patch():
    with open('src/incremental_graph_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    ws_code = """
import requests
import datetime
def push_alert(account_id, decision, score):
    try:
        requests.post('http://localhost:8000/api/internal/push_alert', json={
            'account_id': account_id,
            'decision': decision,
            'score': float(score),
            'timestamp': datetime.datetime.utcnow().isoformat() + "Z"
        }, timeout=1)
    except Exception as e:
        pass
"""
    
    if 'push_alert' not in content:
        content = content.replace('import random', 'import random\n' + ws_code)
        
        old_print = 'print("[ACTION] Routing to RISK_CONTAINMENT_REQUIRED instantly upon arrival.")'
        new_print = 'print("[ACTION] Routing to RISK_CONTAINMENT_REQUIRED instantly upon arrival.")\n        push_alert(new_txn["receiver"], "RISK_CONTAINMENT_REQUIRED", 1.0)'
        content = content.replace(old_print, new_print)

        with open('src/incremental_graph_engine.py', 'w', encoding='utf-8') as f:
            f.write(content)

patch()
