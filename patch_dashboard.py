import re

def patch():
    with open('dashboard/src/App.jsx', 'r', encoding='utf-8') as f:
        content = f.read()

    ws_code = """
  // WebSocket for real-time alerts
  const [liveAlerts, setLiveAlerts] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/alerts");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLiveAlerts(prev => [data, ...prev].slice(0, 5)); // keep last 5
    };
    return () => ws.close();
  }, []);
"""

    alert_render_code = """
      {/* Live Alerts Notification Toast */}
      <div style={{ position: 'fixed', top: '20px', right: '20px', zIndex: 1000, display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {liveAlerts.map((alert, idx) => (
          <div key={idx} style={{ background: '#ff4d4f', color: 'white', padding: '15px 20px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)', animation: 'slideIn 0.3s ease-out' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>🚨 LIVE ALERT: {alert.decision}</div>
            <div style={{ fontSize: '14px' }}>Account: {alert.account_id}</div>
            <div style={{ fontSize: '12px', opacity: 0.8, marginTop: '5px' }}>{new Date(alert.timestamp).toLocaleTimeString()}</div>
          </div>
        ))}
      </div>
"""
    
    if 'setLiveAlerts' not in content:
        # insert hook
        content = content.replace('const [loading, setLoading] = useState(true);', 'const [loading, setLoading] = useState(true);\n' + ws_code)
        
        # insert render
        content = content.replace('<div className="min-h-screen', alert_render_code + '\n    <div className="min-h-screen')

        with open('dashboard/src/App.jsx', 'w', encoding='utf-8') as f:
            f.write(content)

patch()
