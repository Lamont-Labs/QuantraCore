import { useState, useEffect } from 'react'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  component: string
  message: string
  hash?: string
}

interface ProvenanceRecord {
  hash: string
  timestamp: string
  symbol: string
  quantrascore: number
  protocols_fired: number
}

export function LogsPage() {
  const [activeTab, setActiveTab] = useState<'logs' | 'provenance'>('logs')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [provenance, setProvenance] = useState<ProvenanceRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'INFO' | 'WARN' | 'ERROR'>('all')

  useEffect(() => {
    loadData()
  }, [])

  async function loadData() {
    setIsLoading(true)
    
    const mockLogs: LogEntry[] = [
      { timestamp: new Date().toISOString(), level: 'INFO', component: 'ApexEngine', message: 'Engine initialized successfully' },
      { timestamp: new Date().toISOString(), level: 'INFO', component: 'DataManager', message: 'Connected to Polygon.io' },
      { timestamp: new Date().toISOString(), level: 'WARN', component: 'RateLimit', message: 'Approaching API rate limit (80%)' },
      { timestamp: new Date().toISOString(), level: 'INFO', component: 'ProtocolRunner', message: '80 Tier protocols loaded' },
      { timestamp: new Date().toISOString(), level: 'DEBUG', component: 'Cache', message: 'TTL cache initialized: 1000 entries, 5min TTL' },
      { timestamp: new Date().toISOString(), level: 'INFO', component: 'OmegaDirectives', message: '20 safety directives active' },
    ]
    
    const mockProvenance: ProvenanceRecord[] = [
      { hash: 'd680e6cc41aabd1c', timestamp: new Date().toISOString(), symbol: 'AAPL', quantrascore: 72.5, protocols_fired: 45 },
      { hash: 'a1b2c3d4e5f67890', timestamp: new Date().toISOString(), symbol: 'MSFT', quantrascore: 68.2, protocols_fired: 42 },
      { hash: 'f0e9d8c7b6a59483', timestamp: new Date().toISOString(), symbol: 'GOOGL', quantrascore: 55.8, protocols_fired: 38 },
    ]
    
    setLogs(mockLogs)
    setProvenance(mockProvenance)
    setIsLoading(false)
  }

  function getLevelColor(level: string) {
    switch (level) {
      case 'ERROR': return 'text-red-400'
      case 'WARN': return 'text-yellow-400'
      case 'DEBUG': return 'text-slate-500'
      default: return 'text-cyan-400'
    }
  }

  const filteredLogs = filter === 'all' ? logs : logs.filter(l => l.level === filter)

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">≡</span>
          Logs & Provenance
        </h2>
        <p className="text-slate-400 text-sm">
          System logs and cryptographic provenance records for audit trail and reproducibility verification.
        </p>
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
          System Logs
        </button>
        <button
          onClick={() => setActiveTab('provenance')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'provenance'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-cyan-300 border border-transparent'
          }`}
        >
          Provenance Records
        </button>
      </div>

      {activeTab === 'logs' && (
        <div className="flex-1 apex-card overflow-hidden flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-300">Recent Logs</h3>
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
              <div className="text-slate-500">Loading logs...</div>
            ) : filteredLogs.length === 0 ? (
              <div className="text-slate-500">No logs matching filter</div>
            ) : (
              filteredLogs.map((log, i) => (
                <div key={i} className="flex gap-4 py-1 hover:bg-[#0096ff]/5">
                  <span className="text-slate-500 w-48 flex-shrink-0">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`w-16 flex-shrink-0 ${getLevelColor(log.level)}`}>
                    [{log.level}]
                  </span>
                  <span className="text-slate-400 w-32 flex-shrink-0">
                    {log.component}
                  </span>
                  <span className="text-slate-300">{log.message}</span>
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
            Every analysis produces a cryptographic hash for reproducibility verification.
          </p>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-[#0096ff]/20">
                  <th className="text-left py-2 px-3">Hash</th>
                  <th className="text-left py-2 px-3">Timestamp</th>
                  <th className="text-left py-2 px-3">Symbol</th>
                  <th className="text-right py-2 px-3">QuantraScore</th>
                  <th className="text-right py-2 px-3">Protocols</th>
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
