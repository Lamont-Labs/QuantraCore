import { useState, useEffect, useCallback } from 'react'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface DarkPoolPrint {
  symbol: string
  timestamp: string
  price: number
  size: number
  value: number
  is_above_ask: boolean
  is_below_bid: boolean
}

interface ShortInterest {
  symbol: string
  short_percent_float: number
  days_to_cover: number
  cost_to_borrow: number
}

interface DarkPoolSummary {
  total_volume: number
  total_value: number
  buy_ratio: number
  net_flow: string
  block_count: number
  top_prints: DarkPoolPrint[]
  high_short_interest: ShortInterest[]
  accumulation_signals: { symbol: string; signal: string; confidence: number }[]
}

export function DarkPoolPanel() {
  const [data, setData] = useState<DarkPoolSummary | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [view, setView] = useState<'flow' | 'shorts'>('flow')
  const { refreshInterval } = useVelocityMode()

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/data/dark-pool/summary')
      if (res.ok) {
        const json = await res.json()
        setData(json)
      } else {
        setData(null)
      }
    } catch (err) {
      console.error('Failed to fetch dark pool data:', err)
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, refreshInterval)
    return () => clearInterval(interval)
  }, [fetchData, refreshInterval])

  const formatValue = (value: number) => {
    if (value >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(2)}B`
    if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`
    if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}K`
    return `$${value}`
  }

  const formatVolume = (value: number) => {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`
    return value.toString()
  }

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Dark Pool Flow</h3>
            <p className="text-xs text-slate-400">Institutional Activity</p>
          </div>
        </div>
        
        <div className="flex gap-1">
          {(['flow', 'shorts'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                view === v
                  ? 'bg-indigo-500/30 text-indigo-300 border border-indigo-500/50'
                  : 'bg-slate-800/50 text-slate-400 hover:text-slate-300'
              }`}
            >
              {v === 'flow' ? 'Flow' : 'Short Interest'}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full"></div>
        </div>
      ) : !data ? (
        <div className="text-center py-8">
          <div className="text-slate-400 text-sm mb-2">No dark pool data available</div>
          <div className="text-xs text-slate-500">Add provider API keys to enable:</div>
          <div className="text-xs text-indigo-400 mt-1">FINRA_ADF_API_KEY</div>
        </div>
      ) : data && view === 'flow' ? (
        <>
          <div className="grid grid-cols-4 gap-2 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Total Value</div>
              <div className="text-lg font-bold text-white">{formatValue(data.total_value)}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Buy Ratio</div>
              <div className={`text-lg font-bold ${
                data.buy_ratio > 0.55 ? 'text-emerald-400' : 
                data.buy_ratio < 0.45 ? 'text-red-400' : 'text-amber-400'
              }`}>
                {(data.buy_ratio * 100).toFixed(0)}%
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Net Flow</div>
              <div className={`text-sm font-bold ${
                data.net_flow === 'ACCUMULATION' ? 'text-emerald-400' : 
                data.net_flow === 'DISTRIBUTION' ? 'text-red-400' : 'text-amber-400'
              }`}>
                {data.net_flow}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Blocks</div>
              <div className="text-lg font-bold text-indigo-400">{data.block_count.toLocaleString()}</div>
            </div>
          </div>

          <div className="mb-4">
            <div className="text-xs text-slate-400 mb-2">Accumulation Signals</div>
            <div className="flex flex-wrap gap-2">
              {data.accumulation_signals.map((sig, i) => (
                <div 
                  key={i}
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    sig.signal === 'ACCUMULATION' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' :
                    sig.signal === 'DISTRIBUTION' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                    'bg-slate-500/20 text-slate-400 border border-slate-500/40'
                  }`}
                >
                  {sig.symbol}: {sig.signal} ({(sig.confidence * 100).toFixed(0)}%)
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-400 mb-2">Large Block Trades</div>
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {data.top_prints.map((print, i) => (
                <div key={i} className="flex items-center justify-between bg-slate-800/30 hover:bg-slate-800/50 rounded px-2 py-1.5 transition-colors">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white">{print.symbol}</span>
                    {print.is_above_ask && (
                      <span className="text-xs px-1 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">BUY</span>
                    )}
                    {print.is_below_bid && (
                      <span className="text-xs px-1 py-0.5 bg-red-500/20 text-red-400 rounded">SELL</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="text-slate-400">{formatVolume(print.size)}</span>
                    <span className="text-slate-400">${print.price.toFixed(2)}</span>
                    <span className="font-medium text-indigo-400">{formatValue(print.value)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : data && view === 'shorts' ? (
        <div className="space-y-1">
          <div className="grid grid-cols-4 gap-2 text-xs text-slate-400 px-2 pb-2 border-b border-slate-700/50">
            <span>Symbol</span>
            <span className="text-right">Short %</span>
            <span className="text-right">Days Cover</span>
            <span className="text-right">CTB %</span>
          </div>
          {data.high_short_interest.map((item, i) => (
            <div key={i} className="grid grid-cols-4 gap-2 items-center bg-slate-800/30 hover:bg-slate-800/50 rounded px-2 py-2 transition-colors">
              <span className="font-medium text-white">{item.symbol}</span>
              <span className={`text-right text-sm ${
                item.short_percent_float > 30 ? 'text-red-400 font-bold' :
                item.short_percent_float > 20 ? 'text-amber-400' : 'text-slate-300'
              }`}>
                {item.short_percent_float.toFixed(1)}%
              </span>
              <span className={`text-right text-sm ${
                item.days_to_cover > 5 ? 'text-amber-400' : 'text-slate-300'
              }`}>
                {item.days_to_cover.toFixed(1)}
              </span>
              <span className={`text-right text-sm ${
                item.cost_to_borrow > 20 ? 'text-red-400' :
                item.cost_to_borrow > 10 ? 'text-amber-400' : 'text-slate-300'
              }`}>
                {item.cost_to_borrow.toFixed(1)}%
              </span>
            </div>
          ))}
          <div className="mt-3 p-2 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-300">
            High short interest stocks may experience volatile price movements. Exercise caution.
          </div>
        </div>
      ) : null}
    </div>
  )
}
