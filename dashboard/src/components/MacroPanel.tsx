import { useState, useEffect, useCallback } from 'react'
import { useVelocityMode } from '../hooks/useVelocityMode'
import { throttledFetch } from '../lib/requestQueue'

const DEFAULT_REFRESH = 120000

interface EconomicIndicator {
  name: string
  value: number
  previous: number
  change: number
  unit: string
  trend: string
}

interface EconomicEvent {
  name: string
  datetime: string
  importance: string
  forecast: number | null
  previous: number | null
  actual: number | null
}

interface MacroData {
  regime: string
  risk_appetite: string
  yield_curve: string
  inflation_trend: string
  growth_trend: string
  fed_stance: string
  confidence: number
  key_indicators: EconomicIndicator[]
  upcoming_events: EconomicEvent[]
  yield_curve_data: { maturity: string; yield: number }[]
}

export function MacroPanel() {
  const [data, setData] = useState<MacroData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [view, setView] = useState<'overview' | 'calendar' | 'yields'>('overview')
  const { config } = useVelocityMode()
  const refreshInterval = config?.refreshIntervals?.setups || DEFAULT_REFRESH

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await throttledFetch(async () => {
        const res = await fetch('/api/data/macro/summary')
        return res.ok ? res.json() : null
      }, 1)
      setData(result)
    } catch (err) {
      console.error('Failed to fetch macro data:', err)
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, refreshInterval * 2)
    return () => clearInterval(interval)
  }, [fetchData, refreshInterval])

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'RISK_ON': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/40'
      case 'RISK_OFF': return 'text-red-400 bg-red-500/20 border-red-500/40'
      default: return 'text-amber-400 bg-amber-500/20 border-amber-500/40'
    }
  }

  const getTrendIcon = (trend: string) => {
    if (trend === 'UP') return '↑'
    if (trend === 'DOWN') return '↓'
    return '→'
  }

  const getTrendColor = (trend: string, inverse: boolean = false) => {
    if (trend === 'UP') return inverse ? 'text-red-400' : 'text-emerald-400'
    if (trend === 'DOWN') return inverse ? 'text-emerald-400' : 'text-red-400'
    return 'text-slate-400'
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Macro Environment</h3>
            <p className="text-xs text-slate-400">Economic Regime Analysis</p>
          </div>
        </div>
        
        <div className="flex gap-1">
          {(['overview', 'calendar', 'yields'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                view === v
                  ? 'bg-amber-500/30 text-amber-300 border border-amber-500/50'
                  : 'bg-slate-800/50 text-slate-400 hover:text-slate-300'
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full"></div>
        </div>
      ) : !data ? (
        <div className="text-center py-8">
          <div className="text-slate-400 text-sm mb-2">No macro data available</div>
          <div className="text-xs text-slate-500">Add provider API keys to enable:</div>
          <div className="text-xs text-amber-400 mt-1">FRED_API_KEY</div>
        </div>
      ) : data && view === 'overview' ? (
        <>
          <div className="grid grid-cols-3 gap-2 mb-4">
            <div className={`rounded-lg p-3 border ${getRegimeColor(data.regime)}`}>
              <div className="text-xs opacity-70">Market Regime</div>
              <div className="text-lg font-bold">{data.regime.replace('_', ' ')}</div>
              <div className="text-xs opacity-70">{(data.confidence * 100).toFixed(0)}% confidence</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400">Yield Curve</div>
              <div className={`text-lg font-bold ${
                data.yield_curve === 'INVERTED' ? 'text-red-400' :
                data.yield_curve === 'FLAT' ? 'text-amber-400' : 'text-emerald-400'
              }`}>
                {data.yield_curve}
              </div>
              <div className="text-xs text-slate-400">2Y-10Y Spread</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400">Fed Stance</div>
              <div className={`text-lg font-bold ${
                data.fed_stance === 'RESTRICTIVE' ? 'text-red-400' :
                data.fed_stance === 'ACCOMMODATIVE' ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {data.fed_stance}
              </div>
              <div className="text-xs text-slate-400">Policy Direction</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Inflation</div>
              <div className={`text-sm font-medium ${
                data.inflation_trend === 'DECLINING' ? 'text-emerald-400' :
                data.inflation_trend === 'RISING' ? 'text-red-400' : 'text-amber-400'
              }`}>
                {data.inflation_trend}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Growth</div>
              <div className={`text-sm font-medium ${
                data.growth_trend === 'STRONG' ? 'text-emerald-400' :
                data.growth_trend === 'WEAK' || data.growth_trend === 'CONTRACTION' ? 'text-red-400' : 'text-amber-400'
              }`}>
                {data.growth_trend}
              </div>
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-400 mb-2">Key Indicators</div>
            <div className="grid grid-cols-2 gap-1">
              {data.key_indicators.map((ind, i) => (
                <div key={i} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1.5">
                  <span className="text-xs text-slate-300">{ind.name}</span>
                  <div className="flex items-center gap-1">
                    <span className="text-sm font-medium text-white">{ind.value}{ind.unit}</span>
                    <span className={`text-xs ${getTrendColor(ind.trend, ind.name.includes('Unemployment') || ind.name.includes('VIX'))}`}>
                      {getTrendIcon(ind.trend)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : data && view === 'calendar' ? (
        <div className="space-y-2 max-h-72 overflow-y-auto">
          {data.upcoming_events.map((event, i) => (
            <div key={i} className={`bg-slate-800/30 rounded-lg p-3 border-l-2 ${
              event.importance === 'HIGH' ? 'border-l-red-500' :
              event.importance === 'MEDIUM' ? 'border-l-amber-500' : 'border-l-slate-500'
            }`}>
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-sm font-medium text-white">{event.name}</div>
                  <div className="text-xs text-slate-400">{formatDate(event.datetime)}</div>
                </div>
                <span className={`text-xs px-1.5 py-0.5 rounded ${
                  event.importance === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                  event.importance === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-500/20 text-slate-400'
                }`}>
                  {event.importance}
                </span>
              </div>
              <div className="flex gap-4 mt-2 text-xs">
                {event.forecast !== null && (
                  <span className="text-slate-400">Forecast: <span className="text-cyan-400">{event.forecast}</span></span>
                )}
                {event.previous !== null && (
                  <span className="text-slate-400">Previous: <span className="text-slate-300">{event.previous}</span></span>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : data && view === 'yields' ? (
        <div>
          <div className="mb-4 p-3 bg-slate-800/50 rounded-lg">
            <div className="text-xs text-slate-400 mb-1">Yield Curve Status</div>
            <div className={`text-lg font-bold ${
              data.yield_curve === 'INVERTED' ? 'text-red-400' :
              data.yield_curve === 'FLAT' ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {data.yield_curve}
              {data.yield_curve === 'INVERTED' && (
                <span className="text-xs font-normal ml-2 text-red-300">Recession Warning</span>
              )}
            </div>
          </div>
          
          <div className="relative h-32 mb-4">
            <svg className="w-full h-full">
              <polyline
                points={data.yield_curve_data.map((d, i) => 
                  `${(i / (data.yield_curve_data.length - 1)) * 100}%,${100 - ((d.yield - 4) / 2) * 100}%`
                ).join(' ')}
                fill="none"
                stroke="#0096ff"
                strokeWidth="2"
              />
              {data.yield_curve_data.map((d, i) => (
                <circle
                  key={i}
                  cx={`${(i / (data.yield_curve_data.length - 1)) * 100}%`}
                  cy={`${100 - ((d.yield - 4) / 2) * 100}%`}
                  r="4"
                  fill="#0096ff"
                />
              ))}
            </svg>
          </div>
          
          <div className="grid grid-cols-4 gap-1">
            {data.yield_curve_data.map((d, i) => (
              <div key={i} className="text-center bg-slate-800/30 rounded p-1.5">
                <div className="text-xs text-slate-400">{d.maturity}</div>
                <div className="text-sm font-medium text-white">{d.yield.toFixed(2)}%</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}
