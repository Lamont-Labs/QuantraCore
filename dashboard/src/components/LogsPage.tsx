import { useState, useEffect } from 'react'
import { api, type LogEntry, type ProvenanceRecord } from '../lib/api'

export function LogsPage() {
  const [activeTab, setActiveTab] = useState<'logs' | 'provenance'>('logs')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [provenance, setProvenance] = useState<ProvenanceRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'INFO' | 'WARN' | 'ERROR'>('all')
  const [totalCount, setTotalCount] = useState(0)

  useEffect(() => {
    loadData()
  }, [])

  async function loadData() {
    setIsLoading(true)
    setError(null)
    
    try {
      const [logsData, provenanceData] = await Promise.all([
        api.getSystemLogs({ limit: 200 }),
        api.getProvenanceRecords(50)
      ])
      
      setLogs(logsData.logs)
      setTotalCount(logsData.total_count)
      setProvenance(provenanceData.records)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }

  async function handleRefresh() {
    await loadData()
  }

  function getLevelColor(level: string) {
    switch (level.toUpperCase()) {
      case 'ERROR': return 'text-red-400'
      case 'WARN': case 'WARNING': return 'text-yellow-400'
      case 'DEBUG': return 'text-slate-500'
      default: return 'text-cyan-400'
    }
  }

  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(l => l.level.toUpperCase() === filter)

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
            <span className="text-2xl">≡</span>
            Logs & Provenance
          </h2>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-[#0a0f1a] border border-[#0096ff]/30 text-cyan-400 hover:bg-[#0096ff]/10 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
        <p className="text-slate-400 text-sm">
          Real system logs from engine operations and cryptographic provenance records for audit trail verification.
        </p>
        
        {error && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => setActiveTab('logs')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'logs'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-cyan-300 border border-transparent'
          }`}
        >
          System Logs ({totalCount})
        </button>
        <button
          onClick={() => setActiveTab('provenance')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'provenance'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-cyan-300 border border-transparent'
          }`}
        >
          Provenance Records ({provenance.length})
        </button>
      </div>

      {activeTab === 'logs' && (
        <div className="flex-1 apex-card overflow-hidden flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-300">Real-time Logs</h3>
            <div className="flex gap-2">
              {(['all', 'INFO', 'WARN', 'ERROR'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setFilter(level)}
                  className={`px-3 py-1 rounded text-sm ${
                    filter === level
                      ? 'bg-[#0096ff]/20 text-cyan-400'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>
          
          <div className="flex-1 bg-[#030508] rounded-lg p-4 font-mono text-sm overflow-y-auto">
            {isLoading ? (
              <div className="text-slate-500">Loading logs from server...</div>
            ) : filteredLogs.length === 0 ? (
              <div className="text-slate-500">No logs matching filter</div>
            ) : (
              filteredLogs.map((log, i) => (
                <div key={i} className="flex gap-4 py-1 hover:bg-[#0096ff]/5">
                  <span className="text-slate-500 w-36 flex-shrink-0">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`w-16 flex-shrink-0 ${getLevelColor(log.level)}`}>
                    [{log.level}]
                  </span>
                  <span className="text-slate-400 w-32 flex-shrink-0">
                    {log.component}
                  </span>
                  <span className="text-slate-300 flex-1">{log.message}</span>
                  {log.file && (
                    <span className="text-slate-600 text-xs">{log.file}</span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {activeTab === 'provenance' && (
        <div className="flex-1 apex-card overflow-hidden">
          <h3 className="text-lg font-semibold text-slate-300 mb-4">Provenance Records</h3>
          <p className="text-slate-400 text-sm mb-4">
            Every analysis produces a cryptographic hash for reproducibility verification. Same inputs always produce identical outputs.
          </p>
          
          {isLoading ? (
            <div className="text-slate-500">Loading provenance records...</div>
          ) : provenance.length === 0 ? (
            <div className="h-32 flex items-center justify-center text-slate-500">
              <div className="text-center">
                <div className="text-3xl mb-2 opacity-30">🔐</div>
                <div>No provenance records yet</div>
                <div className="text-sm mt-1">Run scans to generate provenance hashes</div>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-400 border-b border-[#0096ff]/20">
                    <th className="text-left py-2 px-3">Hash</th>
                    <th className="text-left py-2 px-3">Timestamp</th>
                    <th className="text-left py-2 px-3">Symbol</th>
                    <th className="text-right py-2 px-3">QuantraScore</th>
                    <th className="text-right py-2 px-3">Protocols</th>
                    <th className="text-left py-2 px-3">Regime</th>
                    <th className="text-left py-2 px-3">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {provenance.map((record) => (
                    <tr key={record.hash} className="border-b border-[#0096ff]/10 hover:bg-[#0096ff]/5">
                      <td className="py-3 px-3 font-mono text-cyan-400">{record.hash}</td>
                      <td className="py-3 px-3 text-slate-400">
                        {new Date(record.timestamp).toLocaleString()}
                      </td>
                      <td className="py-3 px-3 font-semibold">{record.symbol}</td>
                      <td className="py-3 px-3 text-right">{record.quantrascore.toFixed(1)}</td>
                      <td className="py-3 px-3 text-right">{record.protocols_fired}</td>
                      <td className="py-3 px-3 text-slate-300">{record.regime.replace(/_/g, ' ')}</td>
                      <td className={`py-3 px-3 ${
                        record.risk_tier === 'high' ? 'text-red-400' :
                        record.risk_tier === 'medium' ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {record.risk_tier}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          <div className="mt-4 p-3 bg-[#0a0f1a] border border-[#0096ff]/20 rounded-lg text-slate-400 text-sm">
            <strong className="text-cyan-400">Deterministic Guarantee:</strong> Given identical OHLCV windows and seed values, 
            the engine will produce the same window_hash, QuantraScore, and protocol outputs. 
            This enables perfect reproducibility for audit and verification purposes.
          </div>
        </div>
      )}
    </div>
  )
}
