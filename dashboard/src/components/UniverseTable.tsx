import { type ScanResult, type UniverseResult } from '../lib/api'

interface UniverseTableProps {
  data: UniverseResult | null
  selectedSymbol?: string
  onSymbolClick: (result: ScanResult) => void
  isLoading: boolean
  scanMode?: string
}

function getScoreClass(score: number): string {
  if (score >= 80) return 'score-excellent'
  if (score >= 60) return 'score-strong'
  if (score >= 40) return 'score-moderate'
  if (score >= 20) return 'score-weak'
  return 'score-poor'
}

function getFuseScoreClass(score: number): string {
  if (score >= 70) return 'fuse-score-high'
  if (score >= 40) return 'fuse-score-medium'
  return 'fuse-score-low'
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

function getCapBadgeClass(bucket: string): string {
  const badges: Record<string, string> = {
    'mega': 'badge-mega',
    'large': 'badge-large',
    'mid': 'badge-mid',
    'small': 'badge-small',
    'micro': 'badge-micro',
    'nano': 'badge-nano',
    'penny': 'badge-penny',
  }
  return badges[bucket] || 'bg-slate-700/50 text-slate-400 border-slate-600'
}

function LoadingRows() {
  return (
    <>
      {[...Array(8)].map((_, i) => (
        <tr key={i} className="animate-pulse">
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-16 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-5 w-14 bg-slate-700 rounded-full shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-12 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-10 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-20 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-5 w-14 bg-slate-700 rounded-full shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-16 bg-slate-700 rounded shimmer" />
          </td>
          <td className="py-3 border-t border-[#0096ff]/10">
            <div className="h-4 w-8 bg-slate-700 rounded shimmer" />
          </td>
        </tr>
      ))}
    </>
  )
}

const SYMBOL_CAP_MAP: Record<string, string> = {
  'AAPL': 'mega', 'MSFT': 'mega', 'GOOGL': 'mega', 'AMZN': 'mega', 'NVDA': 'mega',
  'META': 'mega', 'TSLA': 'mega', 'BRK.B': 'mega', 'JPM': 'mega', 'V': 'mega',
  'UNH': 'mega', 'XOM': 'mega', 'JNJ': 'mega', 'WMT': 'mega', 'MA': 'mega',
  'BAC': 'large', 'GS': 'large', 'MS': 'large', 'WFC': 'large', 'PYPL': 'large',
  'INTC': 'large', 'NFLX': 'large', 'AMD': 'large', 'CRM': 'large', 'ORCL': 'large',
  'PLTR': 'mid', 'SNAP': 'mid', 'ROKU': 'mid', 'COIN': 'mid', 'HOOD': 'mid',
  'RBLX': 'mid', 'DDOG': 'mid', 'NET': 'mid', 'CRWD': 'mid', 'ZS': 'mid',
  'SNOW': 'mid', 'TTD': 'mid', 'SQ': 'mid', 'AFRM': 'mid', 'UPST': 'mid',
  'CLOV': 'small', 'SOFI': 'small', 'MARA': 'small', 'RIOT': 'small', 'AMC': 'small',
  'GME': 'small', 'PLUG': 'small', 'FCEL': 'small', 'LAZR': 'small', 'QS': 'small',
  'CHPT': 'small', 'BLNK': 'small', 'TLRY': 'small', 'CGC': 'small', 'ACB': 'small',
  'IMPP': 'micro', 'INDO': 'micro', 'HUSA': 'micro', 'CEI': 'micro', 'PROG': 'micro',
  'PHUN': 'micro', 'DWAC': 'micro', 'BKKT': 'micro', 'ASTS': 'micro', 'IONQ': 'micro',
  'SAVA': 'micro', 'OCGN': 'micro', 'BNGO': 'micro',
  'MULN': 'nano', 'FFIE': 'nano',
}

function getSymbolCap(symbol: string): string {
  return SYMBOL_CAP_MAP[symbol] || 'unknown'
}

function generateMockFuseScore(symbol: string): number {
  const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const cap = getSymbolCap(symbol)
  let base = (hash % 40)
  if (cap === 'nano' || cap === 'penny') base += 35
  else if (cap === 'micro') base += 25
  else if (cap === 'small') base += 15
  return Math.min(base, 95)
}

export function UniverseTable({ data, selectedSymbol, onSymbolClick, isLoading, scanMode }: UniverseTableProps) {
  const isSmallCapMode = scanMode && ['high_vol_small_caps', 'low_float_runners', 'momentum_runners'].includes(scanMode)
  
  return (
    <div className="apex-card flex-1 overflow-hidden flex flex-col relative">
      <div
        className="absolute inset-0 opacity-5 pointer-events-none flex items-center justify-center"
        style={{ zIndex: 0 }}
      >
        <svg viewBox="0 0 200 200" className="w-64 h-64 opacity-20">
          <circle cx="100" cy="100" r="90" stroke="#0096ff" strokeWidth="1" fill="none" opacity="0.3"/>
          <circle cx="100" cy="100" r="60" stroke="#00d4ff" strokeWidth="1" fill="none" opacity="0.2"/>
          <circle cx="100" cy="100" r="30" stroke="#0096ff" strokeWidth="2" fill="rgba(0, 150, 255, 0.05)"/>
          <path d="M100 10 L100 30 M100 170 L100 190 M10 100 L30 100 M170 100 L190 100" stroke="#00d4ff" strokeWidth="1" opacity="0.4"/>
        </svg>
      </div>

      <div className="flex items-center justify-between mb-4 relative z-10">
        <div className="flex items-center gap-3">
          <h2 className="apex-heading neon-text">Universe Scanner</h2>
          {data && (
            <span className="text-xs text-slate-500">
              {data.success_count} of {data.scan_count} symbols
            </span>
          )}
          {scanMode && (
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-[#0096ff]/10 text-[#00d4ff] border border-[#0096ff]/30">
              {scanMode.replace(/_/g, ' ').toUpperCase()}
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
          <thead className="sticky top-0 bg-[#030508]/95 backdrop-blur-sm">
            <tr>
              <th>Symbol</th>
              <th>Cap</th>
              <th>
                <span className="flex items-center gap-1 text-[#00d4ff]">
                  Q-Score
                </span>
              </th>
              <th>MR Fuse</th>
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
                const capBucket = getSymbolCap(result.symbol)
                const fuseScore = generateMockFuseScore(result.symbol)
                const isSmallCap = ['small', 'micro', 'nano', 'penny'].includes(capBucket)
                
                return (
                  <tr
                    key={result.symbol}
                    onClick={() => onSymbolClick(result)}
                    className={`cursor-pointer transition-colors
                      ${selectedSymbol === result.symbol
                        ? 'bg-[#0096ff]/15'
                        : 'hover:bg-[#0096ff]/05'
                      }`}
                  >
                    <td className="font-medium text-slate-100">
                      <div className="flex items-center gap-2">
                        {result.symbol}
                        {isSmallCap && isSmallCapMode && (
                          <span className="badge-smallcap">SMALLCAP</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getCapBadgeClass(capBucket)}`}>
                        {capBucket}
                      </span>
                    </td>
                    <td className={`font-mono font-semibold ${getScoreClass(result.quantrascore)}`}>
                      {result.quantrascore.toFixed(1)}
                    </td>
                    <td className={`font-mono text-sm ${getFuseScoreClass(fuseScore)}`}>
                      {fuseScore.toFixed(0)}
                      {fuseScore >= 70 && (
                        <span className="ml-1 text-red-400 animate-pulse">!</span>
                      )}
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
                <td colSpan={8} className="py-12 text-center text-slate-500">
                  <div className="flex flex-col items-center gap-2">
                    <svg className="w-12 h-12 text-[#0096ff]/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span>Click "Run Scan" to analyze the universe</span>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {data?.errors && data.errors.length > 0 && (
        <div className="mt-3 pt-3 border-t border-[#0096ff]/20 relative z-10">
          <div className="text-xs text-amber-500">
            {data.errors.length} symbol(s) failed to scan
          </div>
        </div>
      )}
    </div>
  )
}
