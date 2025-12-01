import { useState, useEffect } from 'react'
import { api, ContinuousLearningStatusResponse, IncrementalLearningStatusResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface ContinuousLearningPanelProps {
  compact?: boolean
}

export function ContinuousLearningPanel({ compact = false }: ContinuousLearningPanelProps) {
  const [continuous, setContinuous] = useState<ContinuousLearningStatusResponse | null>(null)
  const [incremental, setIncremental] = useState<IncrementalLearningStatusResponse | null>(null)
  const { config } = useVelocityMode()

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [contData, incrData] = await Promise.all([
          api.getContinuousLearningStatus().catch(() => null),
          api.getIncrementalLearningStatus().catch(() => null)
        ])
        if (contData) setContinuous(contData)
        if (incrData) setIncremental(incrData)
      } catch (err) {
        console.error('Failed to load learning status:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, config?.refreshIntervals?.models || 60000)
    return () => clearInterval(interval)
  }, [config?.refreshIntervals?.models])

  const getStateColor = (state: string) => {
    switch (state?.toLowerCase()) {
      case 'training': return 'text-cyan-400 bg-cyan-500/20 border-cyan-500/50 animate-pulse'
      case 'running': return 'text-green-400 bg-green-500/20 border-green-500/50'
      case 'idle': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50'
      case 'stopped': return 'text-slate-400 bg-slate-500/20 border-slate-500/50'
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/50'
    }
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const state = String(continuous?.state ?? 'unknown')

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className="text-lg">🧠</span>
          ApexLab Learning
        </h3>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-bold border ${getStateColor(state)}`}>
            {state?.toUpperCase() ?? 'UNKNOWN'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Total Cycles</div>
          <div className="text-xl font-bold text-white">{continuous?.total_cycles ?? 0}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Samples</div>
          <div className="text-xl font-bold text-cyan-400">{formatNumber(continuous?.total_samples_processed ?? 0)}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">Cache</div>
          <div className="text-xl font-bold text-green-400">{formatNumber(continuous?.cache_size ?? 0)}</div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
          <div className="text-xs text-slate-400 mb-1">No Improve</div>
          <div className="text-xl font-bold text-yellow-400">{continuous?.cycles_without_improvement ?? 0}</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-slate-800/30 rounded p-2">
          <span className="text-xs text-slate-500">Last Training</span>
          <div className="text-sm text-white truncate">{formatTime(continuous?.last_training ?? null)}</div>
        </div>
        <div className="bg-slate-800/30 rounded p-2">
          <span className="text-xs text-slate-500">Current Cycle</span>
          <div className="text-sm text-cyan-400">{continuous?.current_cycle ?? 'None'}</div>
        </div>
      </div>

      {continuous?.config && (
        <div className="bg-slate-900/50 rounded-lg p-3 mb-4 border border-slate-700/30">
          <div className="text-xs text-slate-400 mb-2">Configuration</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Interval</span>
              <div className="text-white font-medium">{continuous.config.learning_interval_minutes}m</div>
            </div>
            <div>
              <span className="text-slate-500">Min Samples</span>
              <div className="text-cyan-400 font-medium">{continuous.config.min_new_samples_for_training}</div>
            </div>
            <div>
              <span className="text-slate-500">Max Cache</span>
              <div className="text-yellow-400 font-medium">{formatNumber(continuous.config.max_samples_cache)}</div>
            </div>
            <div>
              <span className="text-slate-500">Drift Thresh</span>
              <div className="text-orange-400 font-medium">{(continuous.config.feature_drift_threshold * 100).toFixed(0)}%</div>
            </div>
            <div>
              <span className="text-slate-500">Warm Start</span>
              <div className={`font-medium ${continuous.config.warm_start_enabled ? 'text-green-400' : 'text-slate-400'}`}>
                {continuous.config.warm_start_enabled ? 'ON' : 'OFF'}
              </div>
            </div>
            <div>
              <span className="text-slate-500">Multi-Pass</span>
              <div className="text-white font-medium">{continuous.config.multi_pass_epochs}</div>
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-700/30">
            <span className="text-xs text-slate-500">Universe: </span>
            <span className="text-xs text-cyan-400">{continuous.config.symbols?.length ?? 251} symbols</span>
            <span className="text-xs text-slate-500 ml-2">Lookback: </span>
            <span className="text-xs text-white">{continuous.config.lookback_days} days</span>
          </div>
        </div>
      )}

      {incremental && (
        <div className="border-t border-slate-700/50 pt-3">
          <div className="text-xs text-slate-400 mb-2">Incremental Learning ({incremental.learning_mode})</div>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800/40 rounded p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Total Samples</span>
                <span className="text-sm text-white font-medium">{formatNumber(incremental.model?.buffer?.total_samples ?? 0)}</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-slate-500">Anchor Buffer</span>
                <span className="text-sm text-cyan-400 font-medium">{formatNumber(incremental.model?.buffer?.anchor_samples ?? 0)}</span>
              </div>
            </div>
            <div className="bg-slate-800/40 rounded p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Recency Buffer</span>
                <span className="text-sm text-green-400 font-medium">{formatNumber(incremental.model?.buffer?.recency_samples ?? 0)}</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-slate-500">Rare Patterns</span>
                <span className="text-sm text-yellow-400 font-medium">
                  {formatNumber(incremental.model?.buffer?.rare_patterns ?? 0)}
                </span>
              </div>
            </div>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Model: <span className="text-white">v{incremental.model?.version ?? 'N/A'}</span>
            <span className="ml-2">Size: <span className="text-cyan-400">{incremental.model?.model_size ?? 'N/A'}</span></span>
            {incremental.model?.is_fitted && (
              <span className="ml-2 text-green-400">Fitted</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
