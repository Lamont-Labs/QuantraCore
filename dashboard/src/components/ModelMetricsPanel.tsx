import { useState, useEffect, useCallback } from 'react'
import { api, type PredictiveStatusResponse, type ContinuousLearningStatusResponse } from '../lib/api'

export function ModelMetricsPanel() {
  const [predictive, setPredictive] = useState<PredictiveStatusResponse | null>(null)
  const [learning, setLearning] = useState<ContinuousLearningStatusResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  const loadData = useCallback(async () => {
    try {
      const [predictiveData, learningData] = await Promise.all([
        api.getPredictiveStatus().catch(() => null),
        api.getContinuousLearningStatus().catch(() => null),
      ])
      setPredictive(predictiveData)
      setLearning(learningData)
      setLastUpdate(new Date())
    } catch (err) {
      console.error('Failed to load model metrics:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [loadData])

  async function handleReloadModels() {
    try {
      await api.reloadModels()
      loadData()
    } catch (err) {
      console.error('Failed to reload models:', err)
    }
  }

  if (isLoading) {
    return (
      <div className="apex-card animate-pulse">
        <div className="h-6 w-32 bg-slate-700 rounded mb-4" />
        <div className="grid grid-cols-3 gap-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-16 bg-slate-700 rounded-lg" />
          ))}
        </div>
      </div>
    )
  }

  const metrics = predictive?.metrics

  return (
    <div className="apex-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          ApexCore V3 Metrics
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={handleReloadModels}
            className="px-2 py-1 text-xs rounded bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 border border-slate-600/50 transition-colors"
          >
            Reload
          </button>
          <span className="text-xs text-slate-500">{lastUpdate.toLocaleTimeString()}</span>
        </div>
      </div>

      <div className="mb-4 p-3 rounded-lg bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-500/30">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-slate-400">Model Status</div>
            <div className={`text-lg font-bold ${predictive?.model_loaded ? 'text-emerald-400' : 'text-red-400'}`}>
              {predictive?.status ?? 'UNKNOWN'}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-slate-400">Training Samples</div>
            <div className="text-lg font-bold text-cyan-400">
              {predictive?.training_samples?.toLocaleString() ?? 'N/A'}
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-slate-400">Heads</div>
            <div className="text-lg font-bold text-purple-400">
              {predictive?.heads?.total ?? 7}
            </div>
          </div>
        </div>
      </div>

      {metrics && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <MetricCard
            label="Runner Accuracy"
            value={`${(metrics.runner_accuracy * 100).toFixed(1)}%`}
            isGood={metrics.runner_accuracy >= 0.85}
          />
          <MetricCard
            label="Quality Accuracy"
            value={`${(metrics.quality_accuracy * 100).toFixed(1)}%`}
            isGood={metrics.quality_accuracy >= 0.75}
          />
          <MetricCard
            label="Avoid Accuracy"
            value={`${(metrics.avoid_accuracy * 100).toFixed(1)}%`}
            isGood={metrics.avoid_accuracy >= 0.90}
          />
          <MetricCard
            label="Regime Accuracy"
            value={`${(metrics.regime_accuracy * 100).toFixed(1)}%`}
            isGood={metrics.regime_accuracy >= 0.80}
          />
          <MetricCard
            label="Timing Accuracy"
            value={`${(metrics.timing_accuracy * 100).toFixed(1)}%`}
            isGood={metrics.timing_accuracy >= 0.80}
          />
          <MetricCard
            label="QuantraScore RMSE"
            value={metrics.quantrascore_rmse.toFixed(4)}
            isGood={metrics.quantrascore_rmse <= 0.05}
          />
        </div>
      )}

      {learning && (
        <div className="pt-4 border-t border-slate-700/50">
          <div className="text-xs text-slate-400 uppercase tracking-wider mb-3">Continuous Learning</div>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-2 rounded bg-slate-800/50">
              <div className="text-xs text-slate-500">State</div>
              <div className={`font-semibold ${learning.running ? 'text-emerald-400' : 'text-slate-400'}`}>
                {learning.state.toUpperCase()}
              </div>
            </div>
            <div className="p-2 rounded bg-slate-800/50">
              <div className="text-xs text-slate-500">Total Cycles</div>
              <div className="font-semibold text-white">{learning.total_cycles}</div>
            </div>
            <div className="p-2 rounded bg-slate-800/50">
              <div className="text-xs text-slate-500">Samples Processed</div>
              <div className="font-semibold text-white">{learning.total_samples_processed.toLocaleString()}</div>
            </div>
            <div className="p-2 rounded bg-slate-800/50">
              <div className="text-xs text-slate-500">Cache Size</div>
              <div className="font-semibold text-white">{learning.cache_size.toLocaleString()}</div>
            </div>
          </div>
          {learning.last_training && (
            <div className="mt-2 text-xs text-slate-500">
              Last training: {new Date(learning.last_training).toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string
  isGood: boolean
}

function MetricCard({ label, value, isGood }: MetricCardProps) {
  return (
    <div className={`p-2 rounded-lg border ${
      isGood 
        ? 'bg-emerald-500/10 border-emerald-500/30' 
        : 'bg-slate-800/50 border-slate-700/30'
    }`}>
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className={`font-bold ${isGood ? 'text-emerald-400' : 'text-white'}`}>
        {value}
      </div>
    </div>
  )
}
