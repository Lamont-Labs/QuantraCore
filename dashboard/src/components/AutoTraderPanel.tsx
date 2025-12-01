import { useState, useEffect } from 'react'
import { api, AutoTraderStatusResponse, ContinuationAnalysisResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface AutoTraderPanelProps {
  compact?: boolean
}

export function AutoTraderPanel({ compact = false }: AutoTraderPanelProps) {
  const [status, setStatus] = useState<AutoTraderStatusResponse | null>(null)
  const [continuation, setContinuation] = useState<ContinuationAnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { config } = useVelocityMode()

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const statusData = await api.getAutoTraderStatus()
        setStatus(statusData)
        setError(null)
      } catch (err) {
        console.error('Failed to load autotrader status:', err)
        setError('Failed to load')
      }
      
      try {
        const contData = await api.getContinuationAnalysis()
        setContinuation(contData)
      } catch (err) {
        console.error('Failed to load continuation data:', err)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, config?.refreshIntervals?.system || 30000)
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.system])

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    return new Date(timestamp).toLocaleTimeString()
  }

  const isEnabled = status?.enabled ?? false
  const isProfitable = (status?.today_pnl ?? 0) >= 0
  const recentTrades = status?.recent_trades ?? []
  const isPaperMode = status?.mode === 'paper'

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">🤖</span>
          AutoTrader
          <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 text-[10px] font-bold rounded border border-amber-500/40">
            PAPER ONLY
          </span>
        </h3>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
            isEnabled 
              ? 'bg-green-500/20 text-green-400 border border-green-500/50' 
              : 'bg-slate-600/30 text-slate-400 border border-slate-500/30'
          }`}>
            {isEnabled ? 'ACTIVE' : 'DISABLED'}
          </span>
          <span className="text-xs text-slate-500">
            {status?.timestamp ? new Date(status.timestamp).toLocaleTimeString() : '--:--:--'}
          </span>
        </div>
      </div>
      
      {isPaperMode && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-2 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-amber-400 text-sm">⚠️</span>
            <span className="text-amber-400/90 text-xs font-medium">
              SIMULATED TRADING - No real money at risk
            </span>
          </div>
        </div>
      )}

      {error ? (
        <div className="text-center py-4">
          <span className="text-yellow-400 text-sm">AutoTrader Offline</span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-3 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-400 mb-1">Today's Trades</div>
              <div className="text-xl font-bold text-white">{status?.today_trades ?? 0}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-400 mb-1">Today's P&L</div>
              <div className={`text-xl font-bold ${isProfitable ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(status?.today_pnl ?? 0)}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-400 mb-1">Positions</div>
              <div className="text-xl font-bold text-white">{status?.active_positions ?? 0}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
              <div className="text-xs text-slate-400 mb-1">Pending</div>
              <div className="text-xl font-bold text-yellow-400">{status?.pending_orders ?? 0}</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-slate-800/30 rounded p-2">
              <span className="text-xs text-slate-500">Last Scan</span>
              <div className="text-sm text-white">{formatTime(status?.last_scan ?? null)}</div>
            </div>
            <div className="bg-slate-800/30 rounded p-2">
              <span className="text-xs text-slate-500">Last Trade</span>
              <div className="text-sm text-white">{formatTime(status?.last_trade ?? null)}</div>
            </div>
          </div>

          {status?.config && (
            <div className="grid grid-cols-4 gap-2 mb-4 text-xs">
              <div className="bg-slate-900/50 rounded p-2 text-center">
                <div className="text-slate-500">Max Daily</div>
                <div className="text-white font-medium">{status.config.max_daily_trades}</div>
              </div>
              <div className="bg-slate-900/50 rounded p-2 text-center">
                <div className="text-slate-500">Min Score</div>
                <div className="text-cyan-400 font-medium">{status.config.min_quantrascore}</div>
              </div>
              <div className="bg-slate-900/50 rounded p-2 text-center">
                <div className="text-slate-500">Max Size</div>
                <div className="text-white font-medium">{formatCurrency(status.config.max_position_size)}</div>
              </div>
              <div className="bg-slate-900/50 rounded p-2 text-center">
                <div className="text-slate-500">Risk/Trade</div>
                <div className="text-yellow-400 font-medium">{(status.config.risk_per_trade * 100).toFixed(1)}%</div>
              </div>
            </div>
          )}

          {status?.daily_limit_reached && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded p-2 mb-4">
              <span className="text-yellow-400 text-xs">Daily trade limit reached</span>
            </div>
          )}

          <div className="border-t border-slate-700/50 pt-3">
            <div className="text-xs text-slate-400 mb-2">Recent Trades</div>
            {recentTrades.length === 0 ? (
              <div className="text-center py-2 text-slate-500 text-sm">No recent trades</div>
            ) : (
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {recentTrades.slice(0, 5).map((trade, i) => (
                  <div key={i} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-bold ${trade.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                        {trade.side.toUpperCase()}
                      </span>
                      <span className="text-sm text-white font-medium">{trade.symbol}</span>
                      <span className="text-xs text-slate-400">x{trade.quantity}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-slate-400">${trade.price.toFixed(2)}</span>
                      {trade.pnl !== undefined && (
                        <span className={`text-xs font-medium ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {continuation && continuation.positions.length > 0 && (
            <div className="border-t border-slate-700/50 pt-3 mt-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-slate-400">Position Hold Analysis</div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Avg Continuation:</span>
                  <span className={`text-xs font-bold ${
                    continuation.summary.avg_continuation >= 0.6 ? 'text-green-400' : 
                    continuation.summary.avg_continuation >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {(continuation.summary.avg_continuation * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {continuation.positions.map((pos) => {
                  const contProb = pos.continuation.probability
                  const decision = pos.decision.hold_decision
                  const pnlColor = pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  const contColor = contProb >= 0.6 ? 'text-green-400' : contProb >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                  const decisionColor = decision.includes('hold_strong') ? 'bg-green-500/20 text-green-400 border-green-500/50' :
                                        decision.includes('hold_normal') ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50' :
                                        decision.includes('trail') ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50' :
                                        decision.includes('reduce') ? 'bg-orange-500/20 text-orange-400 border-orange-500/50' :
                                        'bg-red-500/20 text-red-400 border-red-500/50'
                  
                  return (
                    <div key={pos.symbol} className="bg-slate-800/30 rounded-lg p-2 border border-slate-700/30">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-white">{pos.symbol}</span>
                          <span className={`text-xs ${pnlColor}`}>
                            {pos.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl)}
                          </span>
                        </div>
                        <span className={`px-2 py-0.5 text-[10px] font-bold rounded border ${decisionColor}`}>
                          {decision.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-2 text-xs">
                        <div>
                          <span className="text-slate-500">Cont:</span>
                          <span className={`ml-1 font-medium ${contColor}`}>{(contProb * 100).toFixed(0)}%</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Mom:</span>
                          <span className={`ml-1 font-medium ${
                            pos.continuation.momentum_status === 'healthy' ? 'text-green-400' :
                            pos.continuation.momentum_status === 'weakening' ? 'text-yellow-400' :
                            pos.continuation.momentum_status === 'diverging' ? 'text-red-400' : 'text-slate-400'
                          }`}>{pos.continuation.momentum_status}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Exh:</span>
                          <span className={`ml-1 font-medium ${pos.continuation.exhaustion_level > 0.5 ? 'text-red-400' : 'text-slate-300'}`}>
                            {(pos.continuation.exhaustion_level * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">Rev:</span>
                          <span className={`ml-1 font-medium ${pos.continuation.reversal_probability > 0.5 ? 'text-red-400' : 'text-slate-300'}`}>
                            {(pos.continuation.reversal_probability * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className="mt-1 text-[10px] text-slate-500 truncate">
                        {pos.decision.suggested_action}
                      </div>
                    </div>
                  )
                })}
              </div>
              {continuation.summary.positions_at_risk > 0 && (
                <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400">
                  {continuation.summary.positions_at_risk} position(s) showing reversal risk
                </div>
              )}
            </div>
          )}

          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="text-xs text-amber-400/70 text-center">
              Paper trading simulation only - Not investment advice - Past performance does not guarantee results
            </div>
          </div>
        </>
      )}
    </div>
  )
}
