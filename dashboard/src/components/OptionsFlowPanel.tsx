import { useState, useEffect, useCallback } from 'react'
import { useVelocityMode } from '../hooks/useVelocityMode'
import { throttledFetch } from '../lib/requestQueue'

const DEFAULT_REFRESH = 60000

interface OptionsFlow {
  symbol: string
  timestamp: string
  option_type: string
  strike: number
  expiry: string
  premium: number
  size: number
  is_unusual: boolean
  is_sweep: boolean
  implied_volatility: number
  sentiment: string
}

interface FlowSummary {
  total_premium: number
  call_premium: number
  put_premium: number
  put_call_ratio: number
  unusual_count: number
  sweep_count: number
  bullish_flow_pct: number
  top_symbols: { symbol: string; premium: number; direction: string }[]
}

export function OptionsFlowPanel() {
  const [flows, setFlows] = useState<OptionsFlow[]>([])
  const [summary, setSummary] = useState<FlowSummary | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'unusual' | 'sweeps'>('unusual')
  const { config } = useVelocityMode()
  const refreshInterval = config?.refreshIntervals?.setups || DEFAULT_REFRESH

  const fetchFlows = useCallback(async () => {
    setIsLoading(true)
    try {
      const [flowsRes, summaryRes] = await Promise.all([
        throttledFetch(async () => {
          const res = await fetch('/api/data/options-flow?min_premium=10000')
          return res.ok ? res.json() : null
        }, 1),
        throttledFetch(async () => {
          const res = await fetch('/api/data/options-flow/summary')
          return res.ok ? res.json() : null
        }, 1)
      ])
      
      if (flowsRes) {
        setFlows(flowsRes.flows || [])
      }
      
      if (summaryRes) {
        setSummary(summaryRes)
      }
    } catch (err) {
      console.error('Failed to fetch options flow:', err)
      setSummary(null)
      setFlows([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchFlows()
    const interval = setInterval(fetchFlows, refreshInterval)
    return () => clearInterval(interval)
  }, [fetchFlows, refreshInterval])

  const formatPremium = (value: number) => {
    if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`
    if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}K`
    return `$${value}`
  }

  const filteredFlows = flows.filter(flow => {
    if (filter === 'unusual') return flow.is_unusual
    if (filter === 'sweeps') return flow.is_sweep
    return true
  })

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Options Flow</h3>
            <p className="text-xs text-slate-400">Smart Money Tracking</p>
          </div>
        </div>
        
        <div className="flex gap-1">
          {(['unusual', 'sweeps', 'all'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                filter === f
                  ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                  : 'bg-slate-800/50 text-slate-400 hover:text-slate-300'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {summary && (
        <div className="grid grid-cols-4 gap-2 mb-4">
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-xs text-slate-400">Total Premium</div>
            <div className="text-lg font-bold text-white">{formatPremium(summary.total_premium)}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-xs text-slate-400">P/C Ratio</div>
            <div className={`text-lg font-bold ${
              summary.put_call_ratio < 0.7 ? 'text-emerald-400' : 
              summary.put_call_ratio > 1.3 ? 'text-red-400' : 'text-amber-400'
            }`}>
              {summary.put_call_ratio.toFixed(2)}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-xs text-slate-400">Bullish Flow</div>
            <div className={`text-lg font-bold ${
              summary.bullish_flow_pct > 60 ? 'text-emerald-400' : 
              summary.bullish_flow_pct < 40 ? 'text-red-400' : 'text-amber-400'
            }`}>
              {summary.bullish_flow_pct.toFixed(1)}%
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-xs text-slate-400">Unusual</div>
            <div className="text-lg font-bold text-purple-400">{summary.unusual_count}</div>
          </div>
        </div>
      )}

      {summary && (
        <div className="mb-4">
          <div className="text-xs text-slate-400 mb-2">Top Flow by Premium</div>
          <div className="space-y-1">
            {summary.top_symbols.slice(0, 5).map((item, i) => (
              <div key={i} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1">
                <span className="text-sm font-medium text-white">{item.symbol}</span>
                <span className="text-xs text-slate-400">{formatPremium(item.premium)}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  item.direction === 'BULLISH' ? 'bg-emerald-500/20 text-emerald-400' :
                  item.direction === 'BEARISH' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'
                }`}>
                  {item.direction}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-1 max-h-48 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full"></div>
          </div>
        ) : !summary && filteredFlows.length === 0 ? (
          <div className="text-center py-6">
            <div className="text-slate-400 text-sm mb-2">No options flow providers connected</div>
            <div className="text-xs text-slate-500">Add API keys to enable:</div>
            <div className="text-xs text-purple-400 mt-1">UNUSUAL_WHALES_API_KEY, FLOWALGO_API_KEY</div>
          </div>
        ) : filteredFlows.length === 0 ? (
          <div className="text-center py-4 text-slate-400 text-sm">
            No matching flow data
          </div>
        ) : (
          filteredFlows.slice(0, 10).map((flow, i) => (
            <div key={i} className="flex items-center justify-between bg-slate-800/30 hover:bg-slate-800/50 rounded px-2 py-1.5 transition-colors">
              <div className="flex items-center gap-2">
                <span className="font-medium text-white">{flow.symbol}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  flow.option_type === 'CALL' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {flow.option_type}
                </span>
                {flow.is_sweep && (
                  <span className="text-xs px-1 py-0.5 bg-amber-500/20 text-amber-400 rounded">SWEEP</span>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-slate-400">${flow.strike}</span>
                <span className="text-slate-400">{flow.expiry.split('T')[0]}</span>
                <span className="font-medium text-purple-400">{formatPremium(flow.premium)}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
