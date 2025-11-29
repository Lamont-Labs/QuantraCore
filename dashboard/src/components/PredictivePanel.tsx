import { useState, useEffect, useCallback } from 'react'
import { api, type PredictiveStatusResponse, type PredictiveAdvisoryResponse, type ModelInfoResponse } from '../lib/api'

interface PredictivePanelProps {
  symbol: string | null
  onClose?: () => void
}

export function PredictivePanel({ symbol, onClose }: PredictivePanelProps) {
  const [status, setStatus] = useState<PredictiveStatusResponse | null>(null)
  const [advisory, setAdvisory] = useState<PredictiveAdvisoryResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)

  const loadStatus = useCallback(async () => {
    try {
      const data = await api.getPredictiveStatus()
      setStatus(data)
      setStatusError(null)
    } catch (err) {
      console.error('Failed to load predictive status:', err)
      setStatusError('Connection error')
    }
  }, [])

  const loadModelInfo = useCallback(async () => {
    try {
      const data = await api.getModelInfo()
      setModelInfo(data)
    } catch (err) {
      console.error('Failed to load model info:', err)
    }
  }, [])

  useEffect(() => {
    loadStatus()
    loadModelInfo()
  }, [loadStatus, loadModelInfo])

  useEffect(() => {
    if (statusError && retryCount < 3) {
      const timer = setTimeout(() => {
        setRetryCount(c => c + 1)
        loadStatus()
        loadModelInfo()
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [statusError, retryCount, loadStatus, loadModelInfo])

  const isLayerEnabled = Boolean(status?.enabled && status?.status !== 'DISABLED' && status?.status !== 'ERROR')
  const statusLoaded = status !== null
  
  useEffect(() => {
    if (!statusLoaded) return
    
    if (symbol && isLayerEnabled) {
      loadAdvisory(symbol)
    } else {
      setAdvisory(null)
      setError(null)
    }
  }, [symbol, isLayerEnabled, statusLoaded])

  async function loadAdvisory(sym: string, retries = 0) {
    const MAX_RETRIES = 2
    const RETRY_DELAY = 1000
    
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.getPredictiveAdvise({ symbol: sym })
      setAdvisory(data)
    } catch (err) {
      if (retries < MAX_RETRIES) {
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (retries + 1)))
        return loadAdvisory(sym, retries + 1)
      }
      setError(err instanceof Error ? err.message : 'Advisory unavailable')
      setAdvisory(null)
    } finally {
      setIsLoading(false)
    }
  }

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'UPRANK': return 'text-green-400 bg-green-900/30'
      case 'DOWNRANK': return 'text-orange-400 bg-orange-900/30'
      case 'AVOID': return 'text-red-400 bg-red-900/30'
      case 'NEUTRAL': return 'text-gray-400 bg-gray-700/30'
      case 'DISABLED': return 'text-gray-500 bg-gray-800/30'
      default: return 'text-gray-400 bg-gray-700/30'
    }
  }

  const getQualityTierColor = (tier: string) => {
    switch (tier) {
      case 'A_PLUS': return 'text-emerald-400'
      case 'A': return 'text-green-400'
      case 'B': return 'text-blue-400'
      case 'C': return 'text-yellow-400'
      case 'D': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          Predictive Layer V2
        </h3>
        {onClose && (
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Status</div>
          <div className={`text-sm font-medium ${status?.enabled ? 'text-green-400' : 'text-gray-500'}`}>
            {status?.status || 'Loading...'}
          </div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Model</div>
          <div className="text-sm font-medium text-gray-300">
            {modelInfo?.model_variant || status?.model_variant || 'N/A'}
          </div>
        </div>
      </div>

      {symbol && status && !status.enabled && (
        <div className="border-t border-gray-700 pt-4">
          <div className="bg-gray-900/50 rounded-lg p-4 text-center">
            <svg className="w-8 h-8 mx-auto mb-2 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <div className="text-gray-500 text-sm">Predictive Layer Disabled</div>
            <div className="text-gray-600 text-xs mt-1">Status: {status.status}</div>
            <div className="text-gray-600 text-xs">Models not trained or unavailable</div>
          </div>
        </div>
      )}

      {symbol && isLayerEnabled && (
        <>
          <div className="border-t border-gray-700 pt-4 mb-4">
            <div className="text-sm text-gray-400 mb-2">
              Advisory for <span className="text-white font-medium">{symbol}</span>
            </div>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400"></div>
            </div>
          ) : error ? (
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
              {error}
            </div>
          ) : advisory ? (
            <div className="space-y-4">
              <div className={`rounded-lg p-4 ${getRecommendationColor(advisory.recommendation)}`}>
                <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Recommendation</div>
                <div className="text-xl font-bold">{advisory.recommendation}</div>
                {advisory.reasons.length > 0 && (
                  <div className="mt-2 text-sm opacity-80">
                    {advisory.reasons.map((reason, i) => (
                      <div key={i}>• {reason}</div>
                    ))}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Runner Probability</div>
                  <div className="text-lg font-semibold text-white">
                    {(advisory.runner_prob * 100).toFixed(1)}%
                  </div>
                  <div className="mt-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                      style={{ width: `${advisory.runner_prob * 100}%` }}
                    />
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Quality Tier</div>
                  <div className={`text-lg font-semibold ${getQualityTierColor(advisory.quality_tier)}`}>
                    {advisory.quality_tier}
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Avoid Trade</div>
                  <div className={`text-lg font-semibold ${advisory.avoid_trade_prob > 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                    {(advisory.avoid_trade_prob * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 uppercase mb-1">Confidence</div>
                  <div className="text-lg font-semibold text-white">
                    {(advisory.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-500 uppercase mb-2">Score Comparison</div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500">Engine Score</div>
                    <div className="text-lg font-semibold text-blue-400">
                      {advisory.engine_quantra_score.toFixed(1)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Model Score</div>
                    <div className="text-lg font-semibold text-purple-400">
                      {advisory.model_quantra_score.toFixed(1)}
                    </div>
                  </div>
                </div>
              </div>

              <div className="text-xs text-gray-500 italic">
                {advisory.compliance_note}
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-sm text-center py-4">
              Select a symbol to see predictive advisory
            </div>
          )}
        </>
      )}

      {!symbol && (
        <div className="text-gray-500 text-sm text-center py-8">
          <svg className="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Select a symbol from the universe to get predictive advisory
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-xs text-amber-400/70 flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Predictions are advisory only - engine has final authority
        </div>
      </div>
    </div>
  )
}
