import { useState } from 'react'
import { api, type BacktestResult } from '../lib/api'

export function ResearchPage() {
  const [symbol, setSymbol] = useState('')
  const [lookbackDays, setLookbackDays] = useState(365)
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<BacktestResult[]>([])
  const [error, setError] = useState<string | null>(null)

  async function handleRunBacktest() {
    if (!symbol) return
    setIsRunning(true)
    setError(null)
    
    try {
      const result = await api.runBacktest({
        symbol: symbol.toUpperCase(),
        lookback_days: lookbackDays,
        timeframe: '1d'
      })
      setResults(prev => [result, ...prev])
      setSymbol('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">◎</span>
          Research & Backtests
        </h2>
        <p className="text-slate-400 text-sm mb-6">
          Run real historical analysis using the Apex engine with live market data from Polygon.io.
        </p>
        
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm text-slate-400 mb-2">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="Enter symbol (e.g., AAPL)"
              className="w-full px-4 py-2 bg-[#0a0f1a] border border-[#0096ff]/30 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
            />
          </div>
          <div className="w-48">
            <label className="block text-sm text-slate-400 mb-2">Lookback (days)</label>
            <select
              value={lookbackDays}
              onChange={(e) => setLookbackDays(Number(e.target.value))}
              className="w-full px-4 py-2 bg-[#0a0f1a] border border-[#0096ff]/30 rounded-lg text-white focus:outline-none focus:border-cyan-400"
            >
              <option value={90}>90 days</option>
              <option value={180}>180 days</option>
              <option value={365}>1 year</option>
              <option value={730}>2 years</option>
            </select>
          </div>
          <button
            onClick={handleRunBacktest}
            disabled={!symbol || isRunning}
            className={`px-6 py-2 rounded-lg font-semibold transition-all ${
              !symbol || isRunning
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/25'
            }`}
          >
            {isRunning ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing...
              </span>
            ) : (
              'Run Backtest'
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200 text-sm">
            {error}
          </div>
        )}
      </div>

      <div className="flex-1 apex-card overflow-hidden">
        <h3 className="text-lg font-semibold text-slate-300 mb-4">Backtest Results</h3>
        
        {results.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500">
            <div className="text-center">
              <div className="text-4xl mb-3 opacity-30">📊</div>
              <div>No backtests run yet</div>
              <div className="text-sm mt-1">Enter a symbol above to run real historical analysis</div>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-[#0096ff]/20">
                  <th className="text-left py-2 px-3">Symbol</th>
                  <th className="text-left py-2 px-3">Period</th>
                  <th className="text-right py-2 px-3">Trades</th>
                  <th className="text-right py-2 px-3">Win Rate</th>
                  <th className="text-right py-2 px-3">Return</th>
                  <th className="text-right py-2 px-3">Sharpe</th>
                  <th className="text-right py-2 px-3">Max DD</th>
                  <th className="text-right py-2 px-3">Avg QS</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-[#0096ff]/10 hover:bg-[#0096ff]/5">
                    <td className="py-3 px-3 font-mono text-cyan-400">{r.symbol}</td>
                    <td className="py-3 px-3 text-slate-400">{r.start_date} → {r.end_date}</td>
                    <td className="py-3 px-3 text-right">{r.trades}</td>
                    <td className="py-3 px-3 text-right">{r.win_rate.toFixed(1)}%</td>
                    <td className={`py-3 px-3 text-right ${r.total_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {r.total_return >= 0 ? '+' : ''}{r.total_return.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right">{r.sharpe_ratio.toFixed(2)}</td>
                    <td className="py-3 px-3 text-right text-red-400">{r.max_drawdown.toFixed(1)}%</td>
                    <td className="py-3 px-3 text-right text-cyan-400">{r.avg_quantrascore.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {results.length > 0 && results[0] && (
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
              <h4 className="text-sm text-slate-400 mb-3">Regime Distribution (Latest)</h4>
              <div className="space-y-2">
                {Object.entries(results[0].regime_distribution).map(([regime, count]) => (
                  <div key={regime} className="flex justify-between text-sm">
                    <span className="text-slate-300">{regime.replace(/_/g, ' ')}</span>
                    <span className="text-cyan-400">{count}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
              <h4 className="text-sm text-slate-400 mb-3">Top Protocols (Latest)</h4>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {Object.entries(results[0].protocol_frequency).slice(0, 8).map(([pid, count]) => (
                  <div key={pid} className="flex justify-between text-sm">
                    <span className="text-slate-300 font-mono">{pid}</span>
                    <span className="text-cyan-400">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
