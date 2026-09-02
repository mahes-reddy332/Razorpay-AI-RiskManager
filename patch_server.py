import re

def patch():
    with open('api/server.py', 'r', encoding='utf-8') as f:
        content = f.read()

    ws_code = """
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class AlertPayload(BaseModel):
    account_id: str
    decision: str
    score: float
    timestamp: str

@app.post("/api/internal/push_alert")
async def push_alert(payload: AlertPayload):
    await manager.broadcast(payload.model_dump())
    return {"status": "broadcasted"}
"""
    
    if 'WebSocket' not in content:
        content = content.replace('from pydantic import BaseModel', 'from pydantic import BaseModel\n' + ws_code)
        with open('api/server.py', 'w', encoding='utf-8') as f:
            f.write(content)

patch()
