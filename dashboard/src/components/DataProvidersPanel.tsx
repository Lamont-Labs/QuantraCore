import { useState, useEffect, useCallback } from 'react'

interface ProviderStatus {
  name: string
  available: boolean
  connected: boolean
  subscription_tier: string
  data_types: string[]
  last_error: string | null
  cost_per_month: number
}

interface ProviderSummary {
  active_count: number
  total_count: number
  active_cost: number
  potential_cost: number
  providers: Record<string, ProviderStatus>
}

const PROVIDER_ICONS: Record<string, string> = {
  polygon: '📊',
  alpaca_data: '🦙',
  unusual_whales: '🐋',
  flowalgo: '🌊',
  finnhub: '📰',
  stocktwits: '💬',
  binance: '₿',
  fred: '🏛️',
  nasdaq_totalview: '📈',
  finra_adf: '🏢',
}

export function DataProvidersPanel() {
  const [data, setData] = useState<ProviderSummary | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showGuide, setShowGuide] = useState(false)

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/data/providers/status')
      if (res.ok) {
        const json = await res.json()
        setData(json)
      } else {
        setData(null)
      }
    } catch (err) {
      console.error('Failed to fetch provider status:', err)
      setData(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [fetchData])

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Data Providers</h3>
            <p className="text-xs text-slate-400">Multi-Source Integration</p>
          </div>
        </div>
        
        <button
          onClick={() => setShowGuide(!showGuide)}
          className="px-2 py-1 text-xs bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors"
        >
          {showGuide ? 'Hide Guide' : 'Setup Guide'}
        </button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
        </div>
      ) : !data ? (
        <div className="text-center py-8">
          <div className="text-slate-400 text-sm mb-2">Unable to load provider status</div>
          <div className="text-xs text-slate-500">Backend connection issue</div>
          <div className="text-xs text-cyan-400 mt-1">Check server logs for details</div>
        </div>
      ) : data && !showGuide ? (
        <>
          <div className="grid grid-cols-4 gap-2 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Active</div>
              <div className="text-lg font-bold text-emerald-400">{data.active_count}/{data.total_count}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Monthly Cost</div>
              <div className="text-lg font-bold text-white">${data.active_cost}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Full Suite</div>
              <div className="text-lg font-bold text-slate-400">${data.potential_cost}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400">Status</div>
              <div className="text-sm font-bold text-cyan-400">OPERATIONAL</div>
            </div>
          </div>

          <div className="space-y-1 max-h-60 overflow-y-auto">
            {Object.entries(data.providers).map(([key, provider]) => (
              <div 
                key={key} 
                className={`flex items-center justify-between rounded px-2 py-2 ${
                  provider.connected ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-slate-800/30 border border-slate-700/30'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">{PROVIDER_ICONS[key] || '📡'}</span>
                  <div>
                    <div className="text-sm font-medium text-white">{provider.name}</div>
                    <div className="text-xs text-slate-400">
                      {Array.isArray(provider.data_types) ? provider.data_types.join(', ') : provider.subscription_tier || 'N/A'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {provider.cost_per_month > 0 && (
                    <span className="text-xs text-slate-400">${provider.cost_per_month}/mo</span>
                  )}
                  <span className={`w-2 h-2 rounded-full ${
                    provider.connected ? 'bg-emerald-400' : 'bg-slate-500'
                  }`}></span>
                  <span className={`text-xs ${provider.connected ? 'text-emerald-400' : 'text-slate-500'}`}>
                    {provider.connected ? 'Connected' : 'Not Set'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="space-y-3 max-h-80 overflow-y-auto text-xs">
          <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
            <div className="font-semibold text-cyan-400 mb-1">Plug-and-Play Setup</div>
            <p className="text-slate-300">
              Simply add API keys as environment variables. Providers are auto-discovered and activated.
            </p>
          </div>
          
          <div className="space-y-2">
            <div className="text-slate-400 font-medium">Market Data</div>
            <div className="pl-2 space-y-1 text-slate-300">
              <div><code className="text-cyan-400">POLYGON_API_KEY</code> - Real-time ticks, OHLCV ($249/mo)</div>
              <div><code className="text-cyan-400">ALPACA_PAPER_API_KEY</code> - IEX data (Free)</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-slate-400 font-medium">Options Flow</div>
            <div className="pl-2 space-y-1 text-slate-300">
              <div><code className="text-purple-400">UNUSUAL_WHALES_API_KEY</code> - Options flow ($35/mo)</div>
              <div><code className="text-purple-400">FLOWALGO_API_KEY</code> - Institutional sweeps ($175/mo)</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-slate-400 font-medium">Sentiment & News</div>
            <div className="pl-2 space-y-1 text-slate-300">
              <div><code className="text-blue-400">FINNHUB_API_KEY</code> - News, sentiment (Free tier)</div>
              <div>StockTwits - Social sentiment (No key needed)</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-slate-400 font-medium">Dark Pool & Level 2</div>
            <div className="pl-2 space-y-1 text-slate-300">
              <div><code className="text-indigo-400">FINRA_ADF_API_KEY</code> - Dark pool prints ($50/mo)</div>
              <div><code className="text-indigo-400">NASDAQ_TOTALVIEW_API_KEY</code> - Order book ($100/mo)</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="text-slate-400 font-medium">Economic Data</div>
            <div className="pl-2 space-y-1 text-slate-300">
              <div><code className="text-amber-400">FRED_API_KEY</code> - Fed economic data (Free)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
