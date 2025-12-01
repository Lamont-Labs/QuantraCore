import { useState, useEffect } from 'react'
import { api, LogsResponse, ProvenanceResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface LogsProvenancePanelProps {
  compact?: boolean
}

type TabType = 'logs' | 'provenance'

export function LogsProvenancePanel({ compact = false }: LogsProvenancePanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('logs')
  const [logs, setLogs] = useState<LogsResponse | null>(null)
  const [provenance, setProvenance] = useState<ProvenanceResponse | null>(null)
  const [logLevel, setLogLevel] = useState<string>('all')
  const { config } = useVelocityMode()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [logsData, provData] = await Promise.all([
          api.getSystemLogs({ level: logLevel === 'all' ? undefined : logLevel, limit: 50 }).catch(() => null),
          api.getProvenanceRecords(20).catch(() => null)
        ])
        if (logsData) setLogs(logsData)
        if (provData) setProvenance(provData)
      } catch (err) {
        console.error('Failed to load logs:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, config?.refreshIntervals?.system || 30000)
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.system, logLevel])

  const getLevelColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'error': return 'text-red-400 bg-red-500/20'
      case 'warning': case 'warn': return 'text-yellow-400 bg-yellow-500/20'
      case 'info': return 'text-cyan-400 bg-cyan-500/20'
      case 'debug': return 'text-slate-400 bg-slate-500/20'
      default: return 'text-slate-400 bg-slate-500/20'
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400'
    if (score >= 60) return 'text-yellow-400'
    if (score >= 40) return 'text-orange-400'
    return 'text-red-400'
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const logEntries = logs?.logs ?? []
  const provenanceRecords = provenance?.records ?? []

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">📋</span>
          Logs & Provenance
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('logs')}
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              activeTab === 'logs' 
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50' 
                : 'bg-slate-700/30 text-slate-400 border border-slate-600/30 hover:border-slate-500/50'
            }`}
          >
            System Logs
          </button>
          <button
            onClick={() => setActiveTab('provenance')}
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              activeTab === 'provenance' 
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50' 
                : 'bg-slate-700/30 text-slate-400 border border-slate-600/30 hover:border-slate-500/50'
            }`}
          >
            Audit Trail
          </button>
        </div>
      </div>

      {activeTab === 'logs' && (
        <>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs text-slate-500">Filter:</span>
            {['all', 'error', 'warning', 'info'].map(level => (
              <button
                key={level}
                onClick={() => setLogLevel(level)}
                className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                  logLevel === level 
                    ? 'bg-cyan-500/20 text-cyan-400' 
                    : 'bg-slate-700/30 text-slate-400 hover:text-white'
                }`}
              >
                {level.toUpperCase()}
              </button>
            ))}
            <span className="text-xs text-slate-500 ml-auto">
              {logs?.total_count ?? 0} total
            </span>
          </div>

          <div className="space-y-1 max-h-64 overflow-y-auto font-mono text-xs">
            {logEntries.length === 0 ? (
              <div className="text-center py-4 text-slate-500">No logs available</div>
            ) : (
              logEntries.map((log, i) => (
                <div key={i} className="flex items-start gap-2 bg-slate-800/30 rounded px-2 py-1.5 hover:bg-slate-800/50">
                  <span className="text-slate-500 shrink-0">{formatTime(log.timestamp)}</span>
                  <span className={`px-1 rounded text-[10px] font-bold shrink-0 ${getLevelColor(log.level)}`}>
                    {log.level.substring(0, 3).toUpperCase()}
                  </span>
                  <span className="text-cyan-400 shrink-0">[{log.component}]</span>
                  <span className="text-white truncate">{log.message}</span>
                </div>
              ))
            )}
          </div>

          {logs?.has_more && (
            <div className="text-center mt-2">
              <span className="text-xs text-slate-500">More logs available...</span>
            </div>
          )}
        </>
      )}

      {activeTab === 'provenance' && (
        <>
          <div className="text-xs text-slate-400 mb-2">
            Cryptographic audit trail • {provenance?.count ?? 0} records
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {provenanceRecords.length === 0 ? (
              <div className="text-center py-4 text-slate-500">No provenance records</div>
            ) : (
              provenanceRecords.map((record, i) => (
                <div key={i} className="bg-slate-800/30 rounded-lg p-2 border border-slate-700/30 hover:border-cyan-500/30 transition-colors">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-bold">{record.symbol}</span>
                      <span className={`font-medium ${getScoreColor(record.quantrascore)}`}>
                        QS:{record.quantrascore.toFixed(0)}
                      </span>
                    </div>
                    <span className="text-xs text-slate-500">{formatTime(record.timestamp)}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="text-slate-500">Regime: <span className="text-white">{record.regime}</span></span>
                    <span className="text-slate-500">Risk: <span className="text-yellow-400">{record.risk_tier}</span></span>
                    <span className="text-slate-500">Protocols: <span className="text-cyan-400">{record.protocols_fired}</span></span>
                  </div>
                  <div className="mt-1 text-[10px] text-slate-600 font-mono truncate" title={record.hash}>
                    #{record.hash}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="mt-3 pt-2 border-t border-slate-700/30 text-xs text-slate-500">
            {provenance?.note ?? 'Provenance ensures deterministic reproducibility'}
          </div>
        </>
      )}
    </div>
  )
}
