import { useState, useEffect, useRef } from 'react'
import { api, type ApexLabStatusResponse } from '../lib/api'

export function ApexLabPage() {
  const [status, setStatus] = useState<ApexLabStatusResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    loadStatus()
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  async function loadStatus() {
    try {
      const data = await api.getApexLabStatus()
      setStatus(data)
      setIsLoading(false)

      if (data.is_training && !pollIntervalRef.current) {
        pollIntervalRef.current = setInterval(async () => {
          const updated = await api.getApexLabStatus()
          setStatus(updated)
          if (!updated.is_training && pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
        }, 2000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load status')
      setIsLoading(false)
    }
  }

  async function handleStartTraining() {
    setError(null)
    try {
      await api.startApexLabTraining({
        lookback_days: 365,
        timeframe: '1d'
      })
      
      pollIntervalRef.current = setInterval(async () => {
        const updated = await api.getApexLabStatus()
        setStatus(updated)
        if (!updated.is_training && pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      }, 2000)
      
      await loadStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training')
    }
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-slate-400">Loading ApexLab status...</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">⬡</span>
          ApexLab Training Environment
        </h2>
        <p className="text-slate-400 text-sm mb-6">
          Real machine learning training pipeline using live market data. Builds training datasets and trains GradientBoosting ensembles.
        </p>

        {error && (
          <div className="mb-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Version</div>
            <div className="text-xl font-bold text-cyan-400">{status?.version || 'v2.0'}</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Schema Fields</div>
            <div className="text-xl font-bold text-white">{status?.schema_fields || 40}+</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Training Samples</div>
            <div className="text-xl font-bold text-white">{status?.training_samples?.toLocaleString() || '0'}</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Manifests Available</div>
            <div className="text-xl font-bold text-cyan-400">{status?.manifests_available || 0}</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleStartTraining}
            disabled={status?.is_training}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              status?.is_training
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/25'
            }`}
          >
            {status?.is_training ? 'Training in Progress...' : 'Start Training Pipeline'}
          </button>
          
          {status?.last_training && (
            <div className="text-sm text-slate-400">
              Last training: {new Date(status.last_training).toLocaleString()}
            </div>
          )}
        </div>

        {status?.is_training && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-slate-400 mb-1">
              <span>{status.current_step}</span>
              <span>{status.progress.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-[#0a0f1a] rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                style={{ width: `${status.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 apex-card overflow-hidden">
        <h3 className="text-lg font-semibold text-slate-300 mb-4">Training Logs</h3>
        <div className="h-64 bg-[#030508] rounded-lg p-4 font-mono text-sm overflow-y-auto">
          {!status?.logs || status.logs.length === 0 ? (
            <div className="text-slate-500">No training logs yet. Click "Start Training Pipeline" to begin real training.</div>
          ) : (
            status.logs.map((log, i) => (
              <div 
                key={i} 
                className={`${
                  log.includes('[SUCCESS]') ? 'text-green-400' :
                  log.includes('[ERROR]') ? 'text-red-400' :
                  log.includes('[WARN]') ? 'text-yellow-400' :
                  'text-slate-300'
                }`}
              >
                {log}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="apex-card">
        <h3 className="text-lg font-semibold text-slate-300 mb-4">Training Pipeline Details</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-slate-400 mb-1">Data Source</div>
            <div className="text-white">Polygon.io (Real-time)</div>
          </div>
          <div>
            <div className="text-slate-400 mb-1">Model Type</div>
            <div className="text-white">GradientBoosting Ensemble</div>
          </div>
          <div>
            <div className="text-slate-400 mb-1">Validation</div>
            <div className="text-white">Walk-forward, 80/20 split</div>
          </div>
          <div>
            <div className="text-slate-400 mb-1">Prediction Heads</div>
            <div className="text-white">Runner Prob, Quality Tier, Avoid Trade</div>
          </div>
          <div>
            <div className="text-slate-400 mb-1">Feature Count</div>
            <div className="text-white">40+ fields per sample</div>
          </div>
          <div>
            <div className="text-slate-400 mb-1">Output Format</div>
            <div className="text-white">Joblib + JSON Manifest</div>
          </div>
        </div>
      </div>
    </div>
  )
}
