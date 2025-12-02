import { useState, useEffect } from 'react'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface HyperspeedMetrics {
  total_cycles_completed: number
  total_samples_generated: number
  total_bars_replayed: number
  total_simulations_run: number
  total_training_runs: number
  total_model_updates: number
  cumulative_real_time_equivalent_days: number
  cumulative_actual_runtime_hours: number
  average_acceleration_factor: number
  peak_acceleration_factor: number
  last_cycle_at: string | null
  system_active: boolean
  overnight_mode_active: boolean
}

interface HyperspeedStatus {
  status: string
  engine: {
    active: boolean
    mode: string
    current_cycle: string | null
    cached_samples: number
    model_attached: boolean
    scheduler: {
      state: string
      is_market_hours: boolean
      is_overnight_window: boolean
      time_until_overnight: string
      overnight_remaining: string
      registered_tasks: number
      cycles_completed: number
    }
    metrics: HyperspeedMetrics
    battle_cluster: {
      simulations_run: number
      active: boolean
      strategy_performance: Record<string, any>
    }
    replay_sessions: number
  }
  timestamp: string
}

interface CycleResult {
  status: string
  cycle_id?: string
  bars_processed?: number
  simulations?: number
  samples?: number
  training_triggered?: boolean
  model_updated?: boolean
  duration_seconds?: number
  error?: string
}

interface HyperspeedPanelProps {
  compact?: boolean
}

export function HyperspeedPanel({ compact = false }: HyperspeedPanelProps) {
  const [status, setStatus] = useState<HyperspeedStatus | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [lastCycleResult, setLastCycleResult] = useState<CycleResult | null>(null)
  const { config } = useVelocityMode()

  const fetchStatus = async () => {
    try {
      const response = await fetch('/hyperspeed/status')
      if (response.ok) {
        const data = await response.json()
        setStatus(data)
      }
    } catch (err) {
      console.error('Failed to load hyperspeed status:', err)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, config?.refreshIntervals?.models || 30000)
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.models])

  const startCycle = async (retryCount = 0) => {
    setActionLoading('cycle')
    if (retryCount === 0) setLastCycleResult(null)
    try {
      const response = await fetch('/hyperspeed/cycle/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ years: 1, symbols: ['AAPL', 'MSFT', 'GOOGL', 'TSLA'] })
      })
      if (!response.ok) {
        if (retryCount < 2) {
          setLastCycleResult({ status: 'retrying', error: `Attempt ${retryCount + 1} failed, retrying...` })
          await new Promise(r => setTimeout(r, 500))
          return startCycle(retryCount + 1)
        }
        throw new Error(`Server error: ${response.status}`)
      }
      const result = await response.json()
      setLastCycleResult(result)
      await fetchStatus()
    } catch (err) {
      if (retryCount < 2) {
        setLastCycleResult({ status: 'retrying', error: `Attempt ${retryCount + 1} failed, retrying...` })
        await new Promise(r => setTimeout(r, 500))
        return startCycle(retryCount + 1)
      }
      setLastCycleResult({ 
        status: 'error', 
        error: `${err}. Note: In multi-worker mode, cycles may need to be triggered from the same worker instance. Try again if this fails.`
      })
    } finally {
      setActionLoading(null)
    }
  }

  const startReplay = async () => {
    setActionLoading('replay')
    try {
      await fetch('/hyperspeed/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ years: 5 })
      })
      setTimeout(fetchStatus, 1000)
    } finally {
      setActionLoading(null)
    }
  }

  const startBattle = async () => {
    setActionLoading('battle')
    try {
      await fetch('/hyperspeed/battle', {
        method: 'POST'
      })
      setTimeout(fetchStatus, 1000)
    } finally {
      setActionLoading(null)
    }
  }

  const toggleOvernight = async () => {
    const isActive = status?.engine?.metrics?.overnight_mode_active
    setActionLoading('overnight')
    try {
      await fetch(`/hyperspeed/overnight/${isActive ? 'stop' : 'start'}`, {
        method: 'POST'
      })
      setTimeout(fetchStatus, 1000)
    } finally {
      setActionLoading(null)
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num?.toFixed(0) || '0'
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    return new Date(timestamp).toLocaleString()
  }

  const metrics = status?.engine?.metrics
  const scheduler = status?.engine?.scheduler
  const cluster = status?.engine?.battle_cluster
  const isActive = status?.engine?.active || false
  const overnightActive = metrics?.overnight_mode_active || false

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-purple-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">⚡</span>
          Hyperspeed Learning
        </h3>
        <div className="flex items-center gap-2">
          {isActive && (
            <span className="px-2 py-0.5 rounded text-xs font-bold border text-cyan-400 bg-cyan-500/20 border-cyan-500/50 animate-pulse">
              ACTIVE
            </span>
          )}
          {overnightActive && (
            <span className="px-2 py-0.5 rounded text-xs font-bold border text-purple-400 bg-purple-500/20 border-purple-500/50">
              OVERNIGHT
            </span>
          )}
          {!isActive && !overnightActive && (
            <span className="px-2 py-0.5 rounded text-xs font-bold border text-slate-400 bg-slate-500/20 border-slate-500/50">
              IDLE
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Acceleration</div>
          <div className="text-xl font-bold text-purple-400">{formatNumber(metrics?.average_acceleration_factor || 0)}x</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Samples</div>
          <div className="text-xl font-bold text-cyan-400">{formatNumber(metrics?.total_samples_generated || 0)}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Simulations</div>
          <div className="text-xl font-bold text-green-400">{formatNumber(metrics?.total_simulations_run || 0)}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Cycles</div>
          <div className="text-xl font-bold text-yellow-400">{metrics?.total_cycles_completed || 0}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-800/30 rounded p-2">
          <span className="text-xs text-slate-500">Days Equivalent</span>
          <div className="text-sm text-white font-medium">{formatNumber(metrics?.cumulative_real_time_equivalent_days || 0)} days</div>
        </div>
        <div className="bg-slate-800/30 rounded p-2">
          <span className="text-xs text-slate-500">Actual Runtime</span>
          <div className="text-sm text-cyan-400 font-medium">{(metrics?.cumulative_actual_runtime_hours || 0).toFixed(1)} hrs</div>
        </div>
        <div className="bg-slate-800/30 rounded p-2">
          <span className="text-xs text-slate-500">Peak Speed</span>
          <div className="text-sm text-purple-400 font-medium">{formatNumber(metrics?.peak_acceleration_factor || 0)}x</div>
        </div>
      </div>

      {scheduler && (
        <div className="bg-slate-900/50 rounded-lg p-3 mb-4 border border-slate-700/30">
          <div className="text-xs text-slate-400 mb-2">Scheduler Status</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-slate-500">State</span>
              <div className={`font-medium ${scheduler.state === 'running' ? 'text-green-400' : 'text-slate-400'}`}>
                {scheduler.state?.toUpperCase()}
              </div>
            </div>
            <div>
              <span className="text-slate-500">Market Hours</span>
              <div className={`font-medium ${scheduler.is_market_hours ? 'text-green-400' : 'text-slate-400'}`}>
                {scheduler.is_market_hours ? 'OPEN' : 'CLOSED'}
              </div>
            </div>
            <div>
              <span className="text-slate-500">Overnight Window</span>
              <div className={`font-medium ${scheduler.is_overnight_window ? 'text-purple-400' : 'text-slate-400'}`}>
                {scheduler.is_overnight_window ? 'ACTIVE' : 'WAITING'}
              </div>
            </div>
          </div>
        </div>
      )}

      {cluster && Object.keys(cluster.strategy_performance || {}).length > 0 && (
        <div className="bg-slate-900/50 rounded-lg p-3 mb-4 border border-slate-700/30">
          <div className="text-xs text-slate-400 mb-2">Strategy Performance</div>
          <div className="space-y-1">
            {Object.entries(cluster.strategy_performance).map(([strategy, perf]: [string, any]) => (
              <div key={strategy} className="flex justify-between text-xs">
                <span className="text-slate-300 capitalize">{strategy}</span>
                <span className={`font-medium ${perf.avg_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {perf.win_rate_pct}% WR | {perf.avg_return_pct >= 0 ? '+' : ''}{perf.avg_return_pct}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => startCycle()}
          disabled={actionLoading !== null || isActive}
          className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors"
        >
          {actionLoading === 'cycle' ? 'Running...' : 'Full Cycle'}
        </button>
        <button
          onClick={startReplay}
          disabled={actionLoading !== null || isActive}
          className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors"
        >
          {actionLoading === 'replay' ? 'Starting...' : 'Replay 5Y'}
        </button>
        <button
          onClick={startBattle}
          disabled={actionLoading !== null || isActive}
          className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors"
        >
          {actionLoading === 'battle' ? 'Starting...' : 'Battle Sim'}
        </button>
        <button
          onClick={toggleOvernight}
          disabled={actionLoading !== null}
          className={`px-3 py-1.5 ${overnightActive ? 'bg-orange-600 hover:bg-orange-500' : 'bg-indigo-600 hover:bg-indigo-500'} disabled:bg-slate-600 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors`}
        >
          {actionLoading === 'overnight' ? 'Updating...' : (overnightActive ? 'Stop Overnight' : 'Start Overnight')}
        </button>
      </div>

      {lastCycleResult && (
        <div className={`mt-3 p-3 rounded-lg border ${
          lastCycleResult.status === 'completed' 
            ? 'bg-green-500/10 border-green-500/30' 
            : lastCycleResult.status === 'error'
            ? 'bg-red-500/10 border-red-500/30'
            : lastCycleResult.status === 'retrying'
            ? 'bg-blue-500/10 border-blue-500/30'
            : 'bg-yellow-500/10 border-yellow-500/30'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className={`text-xs font-bold uppercase ${
              lastCycleResult.status === 'completed' ? 'text-green-400' 
              : lastCycleResult.status === 'error' ? 'text-red-400' 
              : lastCycleResult.status === 'retrying' ? 'text-blue-400 animate-pulse'
              : 'text-yellow-400'
            }`}>
              {lastCycleResult.status === 'retrying' ? 'Retrying...' : `Cycle ${lastCycleResult.status}`}
            </span>
            {lastCycleResult.cycle_id && (
              <span className="text-xs text-slate-400 font-mono">{lastCycleResult.cycle_id}</span>
            )}
          </div>
          {lastCycleResult.status === 'completed' && (
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div>
                <span className="text-slate-500">Bars</span>
                <div className="text-cyan-400 font-medium">{formatNumber(lastCycleResult.bars_processed || 0)}</div>
              </div>
              <div>
                <span className="text-slate-500">Simulations</span>
                <div className="text-purple-400 font-medium">{lastCycleResult.simulations}</div>
              </div>
              <div>
                <span className="text-slate-500">Samples</span>
                <div className="text-green-400 font-medium">{lastCycleResult.samples}</div>
              </div>
              <div>
                <span className="text-slate-500">Time</span>
                <div className="text-yellow-400 font-medium">{lastCycleResult.duration_seconds?.toFixed(2)}s</div>
              </div>
            </div>
          )}
          {lastCycleResult.training_triggered && (
            <div className="mt-2 text-xs text-green-400">
              Training triggered - model {lastCycleResult.model_updated ? 'updated' : 'unchanged'}
            </div>
          )}
          {lastCycleResult.error && (
            <div className="mt-1 text-xs text-red-400 font-mono">{lastCycleResult.error}</div>
          )}
        </div>
      )}

      <div className="mt-3 text-xs text-slate-500">
        Last cycle: {formatTime(metrics?.last_cycle_at || null)}
      </div>
    </div>
  )
}
