import { useState } from 'react'

interface BacktestResult {
  symbol: string
  startDate: string
  endDate: string
  trades: number
  winRate: number
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
}

export function ResearchPage() {
  const [symbol, setSymbol] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<BacktestResult[]>([])

  async function handleRunBacktest() {
    if (!symbol) return
    setIsRunning(true)
    
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    const mockResult: BacktestResult = {
      symbol: symbol.toUpperCase(),
      startDate: '2024-01-01',
      endDate: '2024-12-31',
      trades: Math.floor(Math.random() * 50) + 10,
      winRate: Math.random() * 30 + 40,
      totalReturn: (Math.random() * 40) - 10,
      sharpeRatio: Math.random() * 2,
      maxDrawdown: Math.random() * 20 + 5,
    }
    
    setResults(prev => [mockResult, ...prev])
    setIsRunning(false)
    setSymbol('')
  }

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">◎</span>
          Research & Backtests
        </h2>
        <p className="text-slate-400 text-sm mb-6">
          Run historical analysis and backtests on symbols using the Apex engine protocols.
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
                Running...
              </span>
            ) : (
              'Run Backtest'
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 apex-card overflow-hidden">
        <h3 className="text-lg font-semibold text-slate-300 mb-4">Backtest History</h3>
        
        {results.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500">
            <div className="text-center">
              <div className="text-4xl mb-3 opacity-30">📊</div>
              <div>No backtests run yet</div>
              <div className="text-sm mt-1">Enter a symbol above to start</div>
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
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-[#0096ff]/10 hover:bg-[#0096ff]/5">
                    <td className="py-3 px-3 font-mono text-cyan-400">{r.symbol}</td>
                    <td className="py-3 px-3 text-slate-400">{r.startDate} → {r.endDate}</td>
                    <td className="py-3 px-3 text-right">{r.trades}</td>
                    <td className="py-3 px-3 text-right">{r.winRate.toFixed(1)}%</td>
                    <td className={`py-3 px-3 text-right ${r.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {r.totalReturn >= 0 ? '+' : ''}{r.totalReturn.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right">{r.sharpeRatio.toFixed(2)}</td>
                    <td className="py-3 px-3 text-right text-red-400">-{r.maxDrawdown.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
