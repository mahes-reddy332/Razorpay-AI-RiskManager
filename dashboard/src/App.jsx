import { useState, useMemo } from 'react'
import accountsData from './data/accounts.json'
import metricsData from './data/metrics.json'
import l2Data from './data/l2_decisions.json'

function App() {
  const [view, setView] = useState('overview') // overview, detail, metrics
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
    <nav className="bg-slate-900 text-white p-4 shadow-md flex gap-4 font-semibold">
      <div className="text-xl mr-8">UPI Fraud Flow Tracer</div>
      <button className={`hover:text-blue-300 ${view==='overview'?'text-blue-400':''}`} onClick={() => navigateTo('overview')}>Command Center</button>
      <button className={`hover:text-blue-300 ${view==='metrics'?'text-blue-400':''}`} onClick={() => navigateTo('metrics')}>System Metrics</button>
    </nav>
  )

  const renderOverview = () => (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Scored Accounts</h1>
        <select className="p-2 border rounded shadow-sm font-medium" value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="ALL">All Decisions</option>
          <option value="FLAG_MULE">L1: FLAG_MULE (Auto-Freeze)</option>
          <option value="MANUAL_REVIEW">L1: MANUAL_REVIEW (-&gt; L2)</option>
          <option value="SAFE">L1: SAFE</option>
        </select>
      </div>

      <div className="bg-white shadow rounded-lg overflow-hidden border border-slate-200">
        <table className="min-w-full text-left text-sm whitespace-nowrap">
          <thead className="bg-slate-100 uppercase tracking-wider text-slate-600 font-semibold text-xs border-b">
            <tr>
              <th className="px-6 py-4">Account ID</th>
              <th className="px-6 py-4">Score</th>
              <th className="px-6 py-4">Decision</th>
              <th className="px-6 py-4">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filteredAccounts.map(acc => (
              <tr key={acc.id} className="hover:bg-slate-50">
                <td className="px-6 py-4 font-mono font-medium">{acc.id}</td>
                <td className="px-6 py-4 font-bold">{acc.score.toFixed(2)}</td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                    acc.decision === 'FLAG_MULE' ? 'bg-red-100 text-red-700' :
                    acc.decision === 'MANUAL_REVIEW' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {acc.decision}
                  </span>
                  {acc.human_overridden && (
                    <span className="ml-2 px-2 py-1 rounded text-xs font-bold bg-blue-100 text-blue-700">
                      HUMAN OVERRIDE
                    </span>
                  )}
                </td>
                <td className="px-6 py-4">
                  <button onClick={() => navigateTo('detail', acc)} className="text-blue-600 font-semibold hover:underline">
                    Investigate
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
      <div className="p-6 max-w-4xl mx-auto">
        <button onClick={() => navigateTo('overview')} className="text-blue-600 font-bold mb-4 hover:underline">← Back to Overview</button>
        
        <div className="bg-white p-6 shadow-md rounded-lg mb-6 border border-slate-200">
          <h2 className="text-2xl font-bold mb-2 font-mono">{selectedAcc.id}</h2>
          <div className="flex gap-4 mb-6">
            <div className="bg-slate-100 p-3 rounded shadow-inner">
              <span className="text-xs text-slate-500 uppercase font-bold block">Base Score</span>
              <span className="text-xl font-black">{selectedAcc.score.toFixed(2)}</span>
            </div>
            <div className="bg-slate-100 p-3 rounded shadow-inner">
              <span className="text-xs text-slate-500 uppercase font-bold block">Final Decision</span>
              <span className={`text-xl font-black ${
                    selectedAcc.decision === 'FLAG_MULE' ? 'text-red-600' :
                    selectedAcc.decision === 'MANUAL_REVIEW' ? 'text-yellow-600' :
                    'text-green-600'
                  }`}>{selectedAcc.decision}</span>
              {selectedAcc.human_overridden && (
                  <div className="text-xs font-bold text-blue-600 mt-1 uppercase">✓ Human Overridden</div>
              )}
            </div>
          </div>

          <h3 className="font-bold text-lg mb-3 border-b pb-2">Feature Breakdown</h3>
          <ul className="space-y-2 font-medium">
            <li className="flex justify-between p-2 bg-slate-50 rounded"><span>Max Velocity Ratio:</span> <span>{selectedAcc.features.velocity}</span></li>
            <li className="flex justify-between p-2 bg-slate-50 rounded"><span>Risky Sink (MCC):</span> <span>{selectedAcc.features.risky_mcc === 1 ? 'Yes (0.6 penalty)' : 'No'}</span></li>
            <li className="flex justify-between p-2 bg-slate-50 rounded"><span>Topology Nodes:</span> <span>{selectedAcc.features.topology_count}</span></li>
          </ul>
        </div>

        {l2 && (
          <div className="bg-slate-900 text-slate-100 p-6 shadow-lg rounded-lg border-l-4 border-yellow-400">
            <h3 className="text-yellow-400 font-black text-xl mb-2 flex items-center gap-2">
              L2 LLM Copilot Review
            </h3>
            <div className="mb-4">
              <span className="text-sm uppercase text-slate-400 font-bold block">Final Decision:</span>
              <span className={`text-2xl font-black ${l2.decision === 'FRAUD' ? 'text-red-400' : 'text-green-400'}`}>{l2.decision}</span>
            </div>
            <div>
              <span className="text-sm uppercase text-slate-400 font-bold block">Reasoning:</span>
              <p className="text-lg leading-relaxed mt-1">{l2.reasoning}</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderMetrics = () => {
    const { l1_only, combined, extrapolation } = metricsData;
    return (
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        <h1 className="text-3xl font-black text-slate-800 border-b pb-4">Performance Metrics (Strict Test Set)</h1>
        
        <div className="grid grid-cols-2 gap-6">
          {/* L1 Only */}
          <div className="bg-white p-6 shadow rounded-lg border-t-4 border-slate-400">
            <h2 className="text-xl font-bold mb-4">L1 Only (Auto-Freeze)</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-slate-50 p-3 rounded">
                <span className="font-bold text-slate-600">Precision</span>
                <span className="text-2xl font-black text-slate-800">{(l1_only.precision * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-slate-50 p-3 rounded">
                <span className="font-bold text-slate-600">Recall</span>
                <span className="text-2xl font-black text-slate-800">{(l1_only.recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Combined */}
          <div className="bg-white p-6 shadow rounded-lg border-t-4 border-blue-500">
            <h2 className="text-xl font-bold mb-4 text-blue-800">L1 + L2 (Combined System)</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center bg-blue-50 p-3 rounded">
                <span className="font-bold text-slate-600">Precision</span>
                <span className="text-2xl font-black text-blue-800">{(combined.precision * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center bg-blue-50 p-3 rounded">
                <span className="font-bold text-slate-600">Recall</span>
                <span className="text-2xl font-black text-blue-800">{(combined.recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 p-6 shadow rounded-lg border-l-4 border-yellow-500 mt-6">
           <h3 className="text-lg font-black text-yellow-800 mb-3">Scaling & Production Risk (False Positives)</h3>
           <p className="text-yellow-900 mb-4 font-medium leading-relaxed">
             On our strict test set, we measured a <strong>{extrapolation.fpr}% False Positive Rate</strong>. While error rates rarely scale perfectly linearly, here is what that absolute volume looks like at production scale:
           </p>
           <ul className="list-disc pl-6 space-y-2 text-yellow-900 font-bold">
             <li>At 10,000 legitimate users: ~{extrapolation.per_10k} innocent accounts flagged.</li>
             <li>At 1,000,000 legitimate users: ~{extrapolation.per_1m.toLocaleString()} innocent accounts flagged.</li>
           </ul>
           <p className="mt-4 text-sm font-bold text-yellow-800">This volume justifies the L1 -&gt; L2 -&gt; L3 hybrid architecture over a single rigid ruleset.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {renderNav()}
      <main>
        {view === 'overview' && renderOverview()}
        {view === 'detail' && renderDetail()}
        {view === 'metrics' && renderMetrics()}
      </main>
    </div>
  )
}

export default App
