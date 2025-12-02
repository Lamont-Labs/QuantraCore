import { useState, useEffect, useCallback } from 'react'
import { useVelocityMode } from '../hooks/useVelocityMode'
import { throttledFetch } from '../lib/requestQueue'

const DEFAULT_REFRESH = 60000

interface NewsItem {
  title: string
  source: string
  timestamp: string
  sentiment: number
  symbols: string[]
  url?: string
}

interface SentimentData {
  overall_score: number
  trend: string
  news_count: number
  social_volume: number
  bullish_pct: number
  bearish_pct: number
  neutral_pct: number
  top_mentions: { symbol: string; count: number; sentiment: number }[]
  recent_news: NewsItem[]
}

export function SentimentPanel() {
  const [data, setData] = useState<SentimentData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [view, setView] = useState<'overview' | 'news'>('overview')
  const { config } = useVelocityMode()
  const refreshInterval = config?.refreshIntervals?.setups || DEFAULT_REFRESH

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await throttledFetch(async () => {
        const res = await fetch('/api/data/sentiment/summary')
        return res.ok ? res.json() : null
      }, 1)
      setData(result)
    } catch (err) {
      console.error('Failed to fetch sentiment:', err)
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

  const getSentimentColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-400'
    if (score >= 0.55) return 'text-cyan-400'
    if (score >= 0.45) return 'text-amber-400'
    if (score >= 0.3) return 'text-orange-400'
    return 'text-red-400'
  }

  const getSentimentBg = (score: number) => {
    if (score >= 0.7) return 'bg-emerald-500/20 border-emerald-500/40'
    if (score >= 0.55) return 'bg-cyan-500/20 border-cyan-500/40'
    if (score >= 0.45) return 'bg-amber-500/20 border-amber-500/40'
    if (score >= 0.3) return 'bg-orange-500/20 border-orange-500/40'
    return 'bg-red-500/20 border-red-500/40'
  }

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Market Sentiment</h3>
            <p className="text-xs text-slate-400">News & Social Analysis</p>
          </div>
        </div>
        
        <div className="flex gap-1">
          {(['overview', 'news'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                view === v
                  ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-500/50'
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
          <div className="animate-spin w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
        </div>
      ) : !data ? (
        <div className="text-center py-8">
          <div className="text-slate-400 text-sm mb-2">No sentiment data available</div>
          <div className="text-xs text-slate-500">Add provider API keys to enable:</div>
          <div className="text-xs text-cyan-400 mt-1">FINNHUB_API_KEY</div>
        </div>
      ) : data && view === 'overview' ? (
        <>
          <div className="grid grid-cols-3 gap-2 mb-4">
            <div className={`rounded-lg p-3 border ${getSentimentBg(data.overall_score)}`}>
              <div className="text-xs text-slate-400">Overall Sentiment</div>
              <div className={`text-2xl font-bold ${getSentimentColor(data.overall_score)}`}>
                {(data.overall_score * 100).toFixed(0)}
              </div>
              <div className={`text-xs ${getSentimentColor(data.overall_score)}`}>
                {data.trend}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400">News Today</div>
              <div className="text-xl font-bold text-white">{data.news_count.toLocaleString()}</div>
              <div className="text-xs text-slate-400">articles</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400">Social Volume</div>
              <div className="text-xl font-bold text-white">{(data.social_volume / 1000).toFixed(1)}K</div>
              <div className="text-xs text-slate-400">mentions</div>
            </div>
          </div>

          <div className="mb-4">
            <div className="text-xs text-slate-400 mb-2">Sentiment Distribution</div>
            <div className="flex gap-0.5 h-4 rounded overflow-hidden">
              <div 
                className="bg-emerald-500 transition-all"
                style={{ width: `${data.bullish_pct}%` }}
                title={`Bullish: ${data.bullish_pct.toFixed(1)}%`}
              ></div>
              <div 
                className="bg-slate-500 transition-all"
                style={{ width: `${data.neutral_pct}%` }}
                title={`Neutral: ${data.neutral_pct.toFixed(1)}%`}
              ></div>
              <div 
                className="bg-red-500 transition-all"
                style={{ width: `${data.bearish_pct}%` }}
                title={`Bearish: ${data.bearish_pct.toFixed(1)}%`}
              ></div>
            </div>
            <div className="flex justify-between text-xs mt-1">
              <span className="text-emerald-400">{data.bullish_pct.toFixed(1)}% Bull</span>
              <span className="text-slate-400">{data.neutral_pct.toFixed(1)}% Neutral</span>
              <span className="text-red-400">{data.bearish_pct.toFixed(1)}% Bear</span>
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-400 mb-2">Top Mentioned Symbols</div>
            <div className="space-y-1">
              {data.top_mentions.map((item, i) => (
                <div key={i} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 w-4">{i + 1}</span>
                    <span className="font-medium text-white">{item.symbol}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate-400">{item.count.toLocaleString()}</span>
                    <div className={`w-16 text-right text-xs font-medium ${getSentimentColor(item.sentiment)}`}>
                      {(item.sentiment * 100).toFixed(0)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : data && view === 'news' ? (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {data.recent_news.map((news, i) => (
            <div key={i} className="bg-slate-800/30 hover:bg-slate-800/50 rounded-lg p-3 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="text-sm text-white font-medium line-clamp-2">{news.title}</div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-slate-400">{news.source}</span>
                    <span className="text-xs text-slate-500">|</span>
                    <span className="text-xs text-slate-400">
                      {new Date(news.timestamp).toLocaleTimeString()}
                    </span>
                    <div className="flex gap-1">
                      {news.symbols.map((sym, j) => (
                        <span key={j} className="text-xs px-1.5 py-0.5 bg-cyan-500/20 text-cyan-400 rounded">
                          {sym}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${getSentimentBg(news.sentiment)} ${getSentimentColor(news.sentiment)}`}>
                  {(news.sentiment * 100).toFixed(0)}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
