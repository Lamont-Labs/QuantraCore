import { useState, useEffect } from 'react'
import { api, RunnerScreenerResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface ScreenerStatusResponse {
  enabled: boolean
  scanning: boolean
  last_scan?: string | null
  symbols_monitored: number
  alerts_today: number
  config: {
    min_volume_surge: number
    min_price_change: number
    max_float_shares: number
    scan_interval_seconds: number
  }
  timestamp: string
}

interface RunnerScreenerPanelProps {
  compact?: boolean
}

export function RunnerScreenerPanel({ compact = false }: RunnerScreenerPanelProps) {
  const [runners, setRunners] = useState<RunnerScreenerResponse | null>(null)
  const [screenerStatus, setScreenerStatus] = useState<ScreenerStatusResponse | null>(null)
  const { config, isTurbo } = useVelocityMode()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [runnersData, statusData] = await Promise.all([
          api.getRunnerScreener().catch(() => null),
          api.getScreenerStatus().catch(() => null)
        ])
        if (runnersData) setRunners(runnersData)
        if (statusData) setScreenerStatus(statusData)
      } catch (err) {
        console.error('Failed to load runner screener:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, isTurbo ? 5000 : (config?.refreshIntervals?.setups || 15000))
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.setups, isTurbo])

  const getRunnerStateColor = (state: string) => {
    switch (state?.toLowerCase()) {
      case 'explosive': return 'text-red-400 bg-red-500/20 border-red-500/50 animate-pulse'
      case 'primed': return 'text-orange-400 bg-orange-500/20 border-orange-500/50'
      case 'building': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50'
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/50'
    }
  }

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${(value * 100).toFixed(1)}%`
  }

  const formatVolumeSurge = (value: number) => {
    return `${value.toFixed(0)}x`
  }

  const detectedRunners = runners?.detected_runners ?? []
  const isScanning = screenerStatus?.scanning ?? false

  return (
    <div className={`apex-card ${isTurbo ? 'border-red-500/30' : ''} ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">🚀</span>
          Runner Screener
          {isTurbo && <span className="text-xs text-red-400 animate-pulse">TURBO</span>}
        </h3>
        <div className="flex items-center gap-2">
          {isScanning ? (
            <span className="px-2 py-0.5 rounded text-xs font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 animate-pulse">
              SCANNING
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-xs font-bold bg-slate-600/30 text-slate-400 border border-slate-500/30">
              IDLE
            </span>
          )}
          <span className="text-xs text-slate-500">
            {runners?.timestamp ? new Date(runners.timestamp).toLocaleTimeString() : '--:--:--'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Monitored</div>
          <div className="text-xl font-bold text-white">{screenerStatus?.symbols_monitored ?? 114}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Scanned</div>
          <div className="text-xl font-bold text-cyan-400">{runners?.scan_count ?? 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Detected</div>
          <div className="text-xl font-bold text-orange-400">{runners?.detection_count ?? 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Alerts</div>
          <div className="text-xl font-bold text-red-400">{screenerStatus?.alerts_today ?? 0}</div>
        </div>
      </div>

      {screenerStatus?.config && (
        <div className="grid grid-cols-4 gap-2 mb-4 text-xs bg-slate-900/50 rounded-lg p-2 border border-slate-700/30">
          <div className="text-center">
            <div className="text-slate-500">Vol Surge</div>
            <div className="text-white font-medium">{screenerStatus.config.min_volume_surge}x</div>
          </div>
          <div className="text-center">
            <div className="text-slate-500">Min Move</div>
            <div className="text-green-400 font-medium">{(screenerStatus.config.min_price_change * 100).toFixed(0)}%</div>
          </div>
          <div className="text-center">
            <div className="text-slate-500">Max Float</div>
            <div className="text-yellow-400 font-medium">{(screenerStatus.config.max_float_shares / 1e6).toFixed(0)}M</div>
          </div>
          <div className="text-center">
            <div className="text-slate-500">Interval</div>
            <div className="text-cyan-400 font-medium">{screenerStatus.config.scan_interval_seconds}s</div>
          </div>
        </div>
      )}

      <div className="border-t border-slate-700/50 pt-3">
        <div className="text-xs text-slate-400 mb-2">Detected Runners</div>
        {detectedRunners.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-slate-500 text-4xl mb-2">🔍</div>
            <div className="text-slate-400 text-sm">No runners detected</div>
            <div className="text-slate-500 text-xs mt-1">Scanning {screenerStatus?.symbols_monitored ?? 114} low-float stocks</div>
          </div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {detectedRunners.map((runner, i) => (
              <div key={i} className={`bg-slate-800/40 rounded-lg p-3 border transition-all hover:scale-[1.01] ${
                runner.runner_state === 'explosive' ? 'border-red-500/50 shadow-lg shadow-red-500/10' : 'border-slate-700/30'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-bold text-lg">{runner.symbol}</span>
                    <span className={`px-1.5 py-0.5 rounded text-xs font-bold border ${getRunnerStateColor(runner.runner_state)}`}>
                      {runner.runner_state?.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-cyan-400 font-bold">{runner.quantrascore.toFixed(0)}</div>
                    <div className="text-xs text-slate-500">QS</div>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500">Price</span>
                    <div className="text-white font-medium">${runner.current_price.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Change</span>
                    <div className={`font-medium ${runner.price_change_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatPercent(runner.price_change_pct)}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-500">Volume</span>
                    <div className="text-yellow-400 font-medium">{formatVolumeSurge(runner.volume_surge)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Prob</span>
                    <div className="text-orange-400 font-medium">{(runner.runner_probability * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-slate-700/50">
        <div className="text-xs text-red-400/70 text-center">
          EXTREME RISK: Low-float penny stocks carry substantial loss potential - Research only, not trading advice
        </div>
      </div>
    </div>
  )
}
