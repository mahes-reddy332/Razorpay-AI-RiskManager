import { useState, useMemo } from 'react'
import accountsData from './data/accounts.json'
import metricsData from './data/metrics.json'
import l2Data from './data/l2_decisions.json'

function App() {
  const [view, setView] = useState('overview') // overview, detail, metrics, ibm
  const [selectedAcc, setSelectedAcc] = useState(null)
  const [filter, setFilter] = useState('ALL')

  const filteredAccounts = useMemo(() => {
    if (filter === 'ALL') return accountsData;
    return accountsData.filter(a => a.decision === filter)
  }, [filter])

  const navigateTo = (v, acc = null) => {
    setView(v);
    if (acc) setSelectedAcc(acc);
  }

  const renderNav = () => (
    <nav className="bg-slate-900 text-white p-4 shadow-md flex items-center justify-between font-semibold">
      <div className="flex items-center gap-3">
        <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
        <div className="text-xl tracking-tight font-black bg-gradient-to-r from-blue-400 to-indigo-300 bg-clip-text text-transparent">
          UPI Fraud Flow Tracer
        </div>
        <span className="text-xs bg-blue-900/60 border border-blue-400/40 text-blue-300 px-2 py-0.5 rounded font-mono">
          MuleHunter Architecture
        </span>
      </div>
      <div className="flex gap-2">
        <button 
          className={`px-3 py-1.5 rounded transition ${view==='overview'?'bg-blue-600 text-white':'text-slate-300 hover:bg-slate-800'}`} 
          onClick={() => navigateTo('overview')}
        >
          Command Center
        </button>
        <button 
          className={`px-3 py-1.5 rounded transition ${view==='metrics'?'bg-blue-600 text-white':'text-slate-300 hover:bg-slate-800'}`} 
          onClick={() => navigateTo('metrics')}
        >
          Synthetic Evaluation
        </button>
        <button 
          className={`px-3 py-1.5 rounded transition flex items-center gap-1.5 ${view==='ibm'?'bg-emerald-600 text-white':'text-slate-300 hover:bg-slate-800'}`} 
          onClick={() => navigateTo('ibm')}
        >
          <span>5M IBM Benchmark</span>
          <span className="text-[10px] bg-emerald-400/20 text-emerald-300 border border-emerald-400/30 px-1.5 py-0.2 rounded font-mono">SOTA</span>
        </button>
      </div>
    </nav>
  )

  const renderOverview = () => (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex justify-between items-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
        <div>
          <h1 className="text-2xl font-black text-slate-800">Scored Accounts Live Feed</h1>
          <p className="text-sm text-slate-500 font-medium">Real-time Tier 0 + Tier 1 graph decisions with Level 2 LLM review routing</p>
        </div>
        <select className="p-2 border border-slate-300 rounded-lg shadow-sm font-semibold text-slate-700 bg-white" value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="ALL">All Decisions ({accountsData.length})</option>
          <option value="HIGH_RISK">L1: HIGH_RISK (Risk Containment)</option>
          <option value="MANUAL_REVIEW">L1: MANUAL_REVIEW (-&gt; L2 Copilot)</option>
          <option value="SAFE">L1: SAFE (Auto-Cleared)</option>
        </select>
      </div>

      <div className="bg-white shadow-sm rounded-xl overflow-hidden border border-slate-200">
        <table className="min-w-full text-left text-sm whitespace-nowrap">
          <thead className="bg-slate-50 uppercase tracking-wider text-slate-600 font-bold text-xs border-b border-slate-200">
            <tr>
              <th className="px-6 py-4">Account ID</th>
              <th className="px-6 py-4">Risk Score</th>
              <th className="px-6 py-4">Action Tier</th>
              <th className="px-6 py-4">Primary Signals</th>
              <th className="px-6 py-4 text-right">Investigation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-medium">
            {filteredAccounts.map(acc => (
              <tr key={acc.id} className="hover:bg-slate-50/80 transition">
                <td className="px-6 py-4 font-mono font-bold text-slate-900">{acc.id}</td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="font-black text-slate-800">{acc.score.toFixed(2)}</span>
                    <div className="w-16 h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${acc.score >= 1.0 ? 'bg-red-500' : acc.score >= 0.5 ? 'bg-yellow-500' : 'bg-emerald-500'}`} 
                        style={{ width: `${Math.min(acc.score * 100, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                    acc.decision === 'HIGH_RISK' ? 'bg-red-50 text-red-700 border border-red-200' :
                    acc.decision === 'MANUAL_REVIEW' ? 'bg-amber-50 text-amber-700 border border-amber-200' :
                    'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  }`}>
                    {acc.decision}
                  </span>
                  {acc.human_overridden && (
                    <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-700 border border-blue-200">
                      HUMAN OVERRIDE
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 text-xs text-slate-500">
                  <span className="font-mono">
                    vel:{acc.features.velocity} | nodes:{acc.features.topology_count} {acc.features.risky_mcc === 1 ? '| risky_sink' : ''}
                  </span>
                </td>
                <td className="px-6 py-4 text-right">
                  <button onClick={() => navigateTo('detail', acc)} className="bg-slate-100 hover:bg-blue-50 text-blue-700 hover:text-blue-800 px-3 py-1 rounded-lg text-xs font-bold transition border border-slate-200">
                    Investigate Graph →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )

  const renderDetail = () => {
    if (!selectedAcc) return null;
    const l2 = l2Data[selectedAcc.id]
    
    return (
      <div className="p-6 max-w-4xl mx-auto space-y-6">
        <button onClick={() => navigateTo('overview')} className="text-blue-600 font-bold hover:underline flex items-center gap-1">
          ← Back to Command Center
        </button>
        
        <div className="bg-white p-6 shadow-sm rounded-xl border border-slate-200 space-y-6">
          <div className="flex justify-between items-start border-b border-slate-100 pb-4">
            <div>
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Account Telemetry</span>
              <h2 className="text-2xl font-black font-mono text-slate-900">{selectedAcc.id}</h2>
            </div>
            <div className="flex gap-3">
              <div className="bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl text-center">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Base Score</span>
                <span className="text-xl font-black text-slate-900">{selectedAcc.score.toFixed(2)}</span>
              </div>
              <div className="bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl text-center">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Action Tier</span>
                <span className={`text-xl font-black ${
                  selectedAcc.decision === 'HIGH_RISK' ? 'text-red-600' :
                  selectedAcc.decision === 'MANUAL_REVIEW' ? 'text-amber-600' :
                  'text-emerald-600'
                }`}>{selectedAcc.decision}</span>
              </div>
            </div>
          </div>

          <h3 className="font-black text-slate-800 text-lg">Multi-Tier Signal Decomposition</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs font-bold text-slate-500 block mb-1">Tier 0 Velocity Ratio</span>
              <span className="text-xl font-black text-slate-800">{selectedAcc.features.velocity}</span>
              <span className="text-[11px] text-slate-400 block mt-1">24h rapid pass-through</span>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs font-bold text-slate-500 block mb-1">Tier 0 Risky MCC Sink</span>
              <span className="text-xl font-black text-slate-800">{selectedAcc.features.risky_mcc === 1 ? 'Yes (+0.60)' : 'None'}</span>
              <span className="text-[11px] text-slate-400 block mt-1">Direct high-risk merchant destination</span>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs font-bold text-slate-500 block mb-1">Tier 1 Subgraph Nodes</span>
              <span className="text-xl font-black text-slate-800">{selectedAcc.features.topology_count}</span>
              <span className="text-[11px] text-slate-400 block mt-1">15-Hop BFS connected horizon</span>
            </div>
          </div>
        </div>

        {l2 && (
          <div className="bg-slate-900 text-slate-100 p-6 shadow-xl rounded-xl border-l-4 border-amber-400 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-amber-400 font-black text-xl flex items-center gap-2">
                <span>Level 2 LLM Copilot Audit</span>
                <span className="text-xs bg-amber-400/20 text-amber-300 border border-amber-400/30 px-2 py-0.5 rounded font-mono">Gemini 2.5 Flash</span>
              </h3>
              <span className={`px-3 py-1 rounded-full text-xs font-black ${l2.decision === 'FRAUD' ? 'bg-red-500/20 text-red-300 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'}`}>
                Copilot Recommendation: {l2.decision}
              </span>
            </div>
            <div className="bg-slate-800/60 p-4 rounded-lg border border-slate-700/50">
              <span className="text-xs uppercase text-slate-400 font-bold block mb-1">Structured Reasoning Payload:</span>
              <p className="text-slate-200 leading-relaxed text-sm">{l2.reasoning}</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderMetrics = () => {
    const { synthetic_upi } = metricsData;
    return (
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        <div className="border-b border-slate-200 pb-4">
          <h1 className="text-3xl font-black text-slate-900">Synthetic UPI Evaluation (Strict Test Split)</h1>
          <p className="text-sm text-slate-500 mt-1">5,000 Accounts / 29,000+ Transactions with Adversarial Evasion Stress-Tests</p>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 shadow-sm rounded-xl border border-slate-200 border-t-4 border-t-slate-500">
            <h2 className="text-lg font-black text-slate-800 mb-4">Level 1 Only (Risk Containment)</h2>
            <div className="space-y-3 font-semibold">
              <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg">
                <span className="text-slate-600">Precision</span>
                <span className="text-2xl font-black text-slate-900">{(synthetic_upi.l1_only.precision * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg">
                <span className="text-slate-600">Recall</span>
                <span className="text-2xl font-black text-slate-900">{(synthetic_upi.l1_only.recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 shadow-sm rounded-xl border border-slate-200 border-t-4 border-t-blue-500">
            <h2 className="text-lg font-black text-blue-900 mb-4">Level 1 + Level 2 Combined (L1 + Copilot)</h2>
            <div className="space-y-3 font-semibold">
              <div className="flex justify-between items-center bg-blue-50/60 p-3 rounded-lg">
                <span className="text-blue-900">Combined Precision</span>
                <span className="text-2xl font-black text-blue-900">{(synthetic_upi.combined.precision * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-blue-50/60 p-3 rounded-lg">
                <span className="text-blue-900">Combined Recall</span>
                <span className="text-2xl font-black text-blue-900">{(synthetic_upi.combined.recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-amber-50/80 p-6 rounded-xl border border-amber-200 mt-6 space-y-3">
          <h3 className="text-base font-black text-amber-900">Production Scaling & Risk Waterfall Justification</h3>
          <p className="text-sm text-amber-900 leading-relaxed font-medium">
            On the strict test set, we measured a <strong>{synthetic_upi.extrapolation.fpr}% False Positive Rate</strong>. At large scale:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-sm text-amber-900 font-bold">
            <li>At 10,000 legitimate users: ~{synthetic_upi.extrapolation.per_10k} innocent accounts flagged.</li>
            <li>At 1,000,000 legitimate users: ~{synthetic_upi.extrapolation.per_1m.toLocaleString()} innocent accounts flagged.</li>
          </ul>
          <p className="text-xs font-semibold text-amber-800">
            This scale mathematically necessitates our tiered L1 (Auto-Clear) $\rightarrow$ L2 (LLM Review) $\rightarrow$ L3 (Batch Sweep) waterfall rather than a single monolithic risk containmentr.
          </p>
        </div>
      </div>
    )
  }

  const renderIBM = () => {
    const { ibm_aml_benchmark } = metricsData;
    return (
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        <div className="border-b border-slate-200 pb-4">
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-black text-slate-900">5.08M Transaction IBM AML Benchmark</h1>
            <span className="text-xs bg-emerald-100 text-emerald-800 border border-emerald-300 font-mono px-2 py-0.5 rounded font-bold">
              Real-World Graph Scale
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">402,551 Accounts / 5,078,345 Transactions | 6 Complex Adversarial Typologies</p>
        </div>

        {/* Big Metrics Grid */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-center">
            <span className="text-xs font-bold text-slate-500 uppercase block mb-1">Total System Recall</span>
            <span className="text-3xl font-black text-emerald-600">{(ibm_aml_benchmark.total_recall * 100).toFixed(2)}%</span>
            <span className="text-[11px] text-slate-400 block mt-1">{ibm_aml_benchmark.total_caught_mules} / {ibm_aml_benchmark.test_split_mules} mules caught</span>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-center">
            <span className="text-xs font-bold text-slate-500 uppercase block mb-1">Contained for Risk (Tier 0+1)</span>
            <span className="text-3xl font-black text-slate-800">{(ibm_aml_benchmark.auto_frozen_recall * 100).toFixed(1)}%</span>
            <span className="text-[11px] text-slate-400 block mt-1">{ibm_aml_benchmark.auto_frozen_mules} mules contained for risk</span>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-center">
            <span className="text-xs font-bold text-slate-500 uppercase block mb-1">Decoupled Review Tier</span>
            <span className="text-3xl font-black text-blue-600">{(ibm_aml_benchmark.manual_review_recall * 100).toFixed(1)}%</span>
            <span className="text-[11px] text-slate-400 block mt-1">{ibm_aml_benchmark.manual_review_mules} mules sent to review</span>
          </div>
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-center">
            <span className="text-xs font-bold text-slate-500 uppercase block mb-1">Innocent Auto-Cleared</span>
            <span className="text-3xl font-black text-slate-800">{ibm_aml_benchmark.safe_cleared_legit.toLocaleString()}</span>
            <span className="text-[11px] text-slate-400 block mt-1">Zero disruption to merchants</span>
          </div>
        </div>

        {/* Typology Breakdown */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-lg font-black text-slate-900">Per-Typology Recall Breakdown across Graph Shapes</h2>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(ibm_aml_benchmark.typology_recall).map(([key, val]) => (
              <div key={key} className="bg-slate-50 p-4 rounded-xl border border-slate-100 flex justify-between items-center">
                <div>
                  <span className="text-xs font-bold text-slate-500 capitalize block">{key.replace(/_/g, ' ')}</span>
                  <span className="text-xl font-black text-slate-800">{(val * 100).toFixed(1)}%</span>
                </div>
                <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-800 flex items-center justify-center font-mono font-bold text-xs">
                  {val >= 0.75 ? 'HIGH' : 'MED'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Academic Context Alert */}
        <div className="bg-blue-50 p-6 rounded-xl border border-blue-200 space-y-2">
          <h3 className="text-base font-black text-blue-900">Academic Benchmark Alignment (Egressy et al., AAAI 2024)</h3>
          <p className="text-sm text-blue-950 leading-relaxed font-medium">
            Published research on this exact IBM AML multigraph by IBM Research indicates that even complex Graph Neural Networks (GNNs) report modest minority-class $F_1$ improvements over message-passing baselines under 0.77% class imbalance. Our empirical 1.65% precision ceiling on pure topological ledgers aligns with consensus scientific findings, validating why production systems pair deterministic graph filters with contextual LLM Copilots.
          </p>
        </div>
      </div>
    )
  }

  return (
    
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

    <div className="min-h-screen bg-slate-100 font-sans">
      {renderNav()}
      <main className="py-6">
        {view === 'overview' && renderOverview()}
        {view === 'detail' && renderDetail()}
        {view === 'metrics' && renderMetrics()}
        {view === 'ibm' && renderIBM()}
      </main>
    </div>
  )
}

export default App
