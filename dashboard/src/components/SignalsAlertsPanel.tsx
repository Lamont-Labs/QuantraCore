import { useState, useEffect } from 'react'
import { api, SignalsListResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface SMSStatusResponse {
  enabled: boolean
  phone_configured: boolean
  alerts_sent_today: number
  config: {
    min_quantrascore: number
    min_conviction: string
    alert_cooldown_minutes: number
  }
  timestamp: string
}

interface SignalsAlertsPanelProps {
  compact?: boolean
}

export function SignalsAlertsPanel({ compact = false }: SignalsAlertsPanelProps) {
  const [signals, setSignals] = useState<SignalsListResponse | null>(null)
  const [smsStatus, setSmsStatus] = useState<SMSStatusResponse | null>(null)
  const { config } = useVelocityMode()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [signalsData, smsData] = await Promise.all([
          api.getSignalsList().catch(() => null),
          api.getSmsStatus().catch(() => null)
        ])
        if (signalsData) setSignals(signalsData)
        if (smsData) setSmsStatus(smsData)
      } catch (err) {
        console.error('Failed to load signals:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, config?.refreshIntervals?.setups || 15000)
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.setups])

  const getConvictionColor = (conviction: string) => {
    switch (conviction?.toLowerCase()) {
      case 'extreme': return 'text-red-400 bg-red-500/20 border-red-500/50'
      case 'high': return 'text-orange-400 bg-orange-500/20 border-orange-500/50'
      case 'moderate': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50'
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/50'
    }
  }

  const getDirectionIcon = (direction: string) => {
    return direction?.toLowerCase() === 'long' ? '↗' : '↘'
  }

  const signalList = signals?.signals ?? []
  const highConviction = signals?.high_conviction_count ?? 0

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">📡</span>
          Signals & Alerts
        </h3>
        <div className="flex items-center gap-2">
          {smsStatus?.enabled ? (
            <span className="px-2 py-0.5 rounded text-xs font-bold bg-green-500/20 text-green-400 border border-green-500/50 flex items-center gap-1">
              <span>📱</span> SMS ON
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-xs font-bold bg-slate-600/30 text-slate-400 border border-slate-500/30">
              SMS OFF
            </span>
          )}
          <span className="text-xs text-slate-500">
            {signals?.timestamp ? new Date(signals.timestamp).toLocaleTimeString() : '--:--:--'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Active Signals</div>
          <div className="text-xl font-bold text-white">{signals?.count ?? 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">High Conviction</div>
          <div className="text-xl font-bold text-orange-400">{highConviction}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Alerts Today</div>
          <div className="text-xl font-bold text-cyan-400">{smsStatus?.alerts_sent_today ?? 0}</div>
        </div>
      </div>

      {smsStatus && (
        <div className="bg-slate-900/50 rounded-lg p-3 mb-4 border border-slate-700/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400">SMS Alert Settings</span>
            <span className={`w-2 h-2 rounded-full ${smsStatus.phone_configured ? 'bg-green-400' : 'bg-yellow-400'}`} />
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Min Score</span>
              <div className="text-cyan-400 font-medium">{smsStatus.config?.min_quantrascore ?? 75}</div>
            </div>
            <div>
              <span className="text-slate-500">Min Conviction</span>
              <div className="text-orange-400 font-medium capitalize">{smsStatus.config?.min_conviction ?? 'High'}</div>
            </div>
            <div>
              <span className="text-slate-500">Cooldown</span>
              <div className="text-white font-medium">{smsStatus.config?.alert_cooldown_minutes ?? 15}m</div>
            </div>
          </div>
        </div>
      )}

      <div className="border-t border-slate-700/50 pt-3">
        <div className="text-xs text-slate-400 mb-2">Live Signals</div>
        {signalList.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-slate-500 text-4xl mb-2">📭</div>
            <div className="text-slate-400 text-sm">No active signals</div>
            <div className="text-slate-500 text-xs mt-1">Market may be closed</div>
          </div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {signalList.map((signal, i) => (
              <div key={i} className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30 hover:border-cyan-500/30 transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`text-lg ${signal.direction === 'long' ? 'text-green-400' : 'text-red-400'}`}>
                      {getDirectionIcon(signal.direction)}
                    </span>
                    <span className="text-white font-bold">{signal.symbol}</span>
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${getConvictionColor(signal.conviction)}`}>
                      {signal.conviction}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-cyan-400 font-bold text-sm">{signal.quantrascore.toFixed(0)}</div>
                    <div className="text-xs text-slate-500">QS</div>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500">Entry</span>
                    <div className="text-white">${signal.entry_price.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Stop</span>
                    <div className="text-red-400">${signal.stop_loss.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Target</span>
                    <div className="text-green-400">${signal.target.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Timing</span>
                    <div className="text-yellow-400 capitalize">{signal.timing}</div>
                  </div>
                </div>
                {signal.predicted_top && (
                  <div className="mt-2 pt-2 border-t border-slate-700/30">
                    <span className="text-xs text-slate-500">Predicted Top: </span>
                    <span className="text-xs text-cyan-400 font-medium">${signal.predicted_top.toFixed(2)}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-slate-700/50">
        <div className="text-xs text-amber-400/70 text-center">
          Signals are structural probability analyses for research only - not trading advice
        </div>
      </div>
    </div>
  )
}
