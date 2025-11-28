import { type ScanResult, type UniverseResult } from '../lib/api'

interface UniverseTableProps {
  data: UniverseResult | null
  selectedSymbol?: string
  onSymbolClick: (result: ScanResult) => void
  isLoading: boolean
}

function getScoreClass(score: number): string {
  if (score >= 80) return 'score-excellent'
  if (score >= 60) return 'score-strong'
  if (score >= 40) return 'score-moderate'
  if (score >= 20) return 'score-weak'
  return 'score-poor'
}

function getRegimeColor(regime: string): string {
  const colors: Record<string, string> = {
    'trending_up': 'text-emerald-400',
    'trending_down': 'text-red-400',
    'ranging': 'text-amber-400',
    'volatile': 'text-orange-400',
    'compressed': 'text-violet-400',
    'breakout': 'text-cyan-400',
  }
  return colors[regime] || 'text-slate-400'
}

function getRiskBadge(tier: string): { bg: string; text: string } {
  const badges: Record<string, { bg: string; text: string }> = {
    'low': { bg: 'bg-emerald-500/20 border-emerald-500/30', text: 'text-emerald-400' },
    'medium': { bg: 'bg-amber-500/20 border-amber-500/30', text: 'text-amber-400' },
    'high': { bg: 'bg-orange-500/20 border-orange-500/30', text: 'text-orange-400' },
    'extreme': { bg: 'bg-red-500/20 border-red-500/30', text: 'text-red-400' },
  }
  return badges[tier] || badges['medium']
}

function LoadingRows() {
  return (
    <>
      {[...Array(8)].map((_, i) => (
        <tr key={i} className="animate-pulse">
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-4 w-16 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-4 w-12 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-4 w-20 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-5 w-14 bg-slate-700 rounded-full shimmer" />
          </td>
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-4 w-16 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-slate-700/50">
            <div className="h-4 w-8 bg-slate-700 rounded shimmer" />
          </td>
        </tr>
      ))}
    </>
  )
}

export function UniverseTable({ data, selectedSymbol, onSymbolClick, isLoading }: UniverseTableProps) {
  return (
    <div className="apex-card flex-1 overflow-hidden flex flex-col relative">
      <div
        className="absolute inset-0 opacity-5 pointer-events-none flex items-center justify-center"
        style={{ zIndex: 0 }}
      >
        <img
          src="/assets/quantra-q-icon.png"
          alt=""
          className="w-64 h-64 opacity-30"
          onError={(e) => e.currentTarget.style.display = 'none'}
        />
      </div>

      <div className="flex items-center justify-between mb-4 relative z-10">
        <div className="flex items-center gap-2">
          <h2 className="apex-heading">Universe Scanner</h2>
          {data && (
            <span className="text-xs text-slate-500">
              {data.success_count} of {data.scan_count} symbols
            </span>
          )}
        </div>
        {data?.timestamp && (
          <span className="text-xs text-slate-500 font-mono">
            {new Date(data.timestamp).toLocaleTimeString()}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto relative z-10">
        <table className="apex-table">
          <thead className="sticky top-0 bg-slate-900/95">
            <tr>
              <th>Symbol</th>
              <th>
                <span className="flex items-center gap-1">
                  <img
                    src="/assets/quantra-q-icon.png"
                    alt="Q"
                    className="w-3 h-3"
                    onError={(e) => e.currentTarget.style.display = 'none'}
                  />
                  Score
                </span>
              </th>
              <th>Regime</th>
              <th>Risk Tier</th>
              <th>Entropy</th>
              <th>Protocols</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <LoadingRows />
            ) : data?.results.length ? (
              data.results.map((result) => {
                const riskBadge = getRiskBadge(result.risk_tier)
                return (
                  <tr
                    key={result.symbol}
                    onClick={() => onSymbolClick(result)}
                    className={`cursor-pointer transition-colors
                      ${selectedSymbol === result.symbol
                        ? 'bg-cyan-500/10'
                        : 'hover:bg-slate-800/30'
                      }`}
                  >
                    <td className="font-medium text-slate-100">{result.symbol}</td>
                    <td className={`font-mono font-semibold ${getScoreClass(result.quantrascore)}`}>
                      {result.quantrascore.toFixed(1)}
                    </td>
                    <td className={`text-sm ${getRegimeColor(result.regime)}`}>
                      {result.regime.replace('_', ' ')}
                    </td>
                    <td>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${riskBadge.bg} ${riskBadge.text}`}>
                        {result.risk_tier}
                      </span>
                    </td>
                    <td className="text-sm text-slate-400">{result.entropy_state}</td>
                    <td className="text-sm text-slate-500 font-mono">{result.protocol_fired_count}</td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td colSpan={6} className="py-12 text-center text-slate-500">
                  Click "Run Scan" to analyze the universe
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {data?.errors && data.errors.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-700/50 relative z-10">
          <div className="text-xs text-amber-500">
            {data.errors.length} symbol(s) failed to scan
          </div>
        </div>
      )}
    </div>
  )
}
