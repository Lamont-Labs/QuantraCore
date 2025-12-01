import { useState, useEffect } from 'react'
import { type ScanResult } from '../lib/api'

interface DataProviderStatus {
  name: string
  available: boolean
  rate_limit?: number
}

const SWING_MODES = [
  { id: 'momentum_runners', label: 'Momentum Runners', risk: 'high', description: 'High momentum + volume surge' },
  { id: 'mid_cap_focus', label: 'Mid-Cap Focus', risk: 'medium', description: 'Growth potential, moderate risk' },
  { id: 'mega_large_focus', label: 'Blue Chips', risk: 'low', description: 'Stable, liquid names' },
  { id: 'high_vol_small_caps', label: 'High Vol Small Caps', risk: 'high', description: 'Small caps with volatility' },
]

export function SwingTradePage() {
  const [selectedMode, setSelectedMode] = useState('momentum_runners')
  const [isScanning, setIsScanning] = useState(false)
  const [scanResults, setScanResults] = useState<ScanResult[]>([])
  const [providers, setProviders] = useState<DataProviderStatus[]>([])
  const [error, setError] = useState<string | null>(null)
  const [lastScanTime, setLastScanTime] = useState<string | null>(null)
  const [selectedSetup, setSelectedSetup] = useState<ScanResult | null>(null)

  useEffect(() => {
    loadProviderStatus()
  }, [])

  async function loadProviderStatus() {
    try {
      const response = await fetch('http://localhost:8000/data_providers')
      if (response.ok) {
        const data = await response.json()
        setProviders(data.providers || [])
      }
    } catch (err) {
      console.error('Failed to load provider status:', err)
    }
  }

  async function runSwingScan() {
    setIsScanning(true)
    setError(null)
    setScanResults([])

    const hasDataProvider = providers.some(p => p.available && p.name !== 'Synthetic')
    if (!hasDataProvider) {
      setError('No real data providers available. Please check API credentials.')
      setIsScanning(false)
      return
    }

    try {
      const response = await fetch('http://localhost:8000/scan_universe_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: selectedMode,
          max_results: 50,
          include_mr_fuse: true
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Scan failed with status ${response.status}`)
      }

      const result = await response.json()
      const results = result.results || []

      const sorted = [...results].sort((a: ScanResult, b: ScanResult) => b.quantrascore - a.quantrascore)
      setScanResults(sorted)
      setLastScanTime(new Date().toLocaleTimeString())

      if (sorted.length > 0) {
        setSelectedSetup(sorted[0])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed')
    } finally {
      setIsScanning(false)
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 75) return 'text-emerald-400'
    if (score >= 60) return 'text-cyan-400'
    if (score >= 45) return 'text-amber-400'
    return 'text-slate-400'
  }

  function getRiskColor(tier: string): string {
    const t = tier.toLowerCase()
    if (t.includes('low') || t === 'a+' || t === 'a') return 'text-emerald-400'
    if (t.includes('medium') || t === 'b') return 'text-amber-400'
    return 'text-red-400'
  }

  function getVerdictBadge(action: string): { bg: string; text: string } {
    const a = action.toUpperCase()
    if (a.includes('BUY') || a.includes('LONG') || a === 'ENTER') {
      return { bg: 'bg-emerald-500/20 border-emerald-500/50', text: 'text-emerald-400' }
    }
    if (a.includes('SELL') || a.includes('SHORT') || a === 'EXIT') {
      return { bg: 'bg-red-500/20 border-red-500/50', text: 'text-red-400' }
    }
    if (a.includes('HOLD') || a.includes('WAIT')) {
      return { bg: 'bg-amber-500/20 border-amber-500/50', text: 'text-amber-400' }
    }
    return { bg: 'bg-slate-500/20 border-slate-500/50', text: 'text-slate-400' }
  }

  const modeConfig = SWING_MODES.find(m => m.id === selectedMode)
  const actionableSetups = scanResults.filter(r => 
    r.quantrascore >= 60 && 
    (r.verdict_action.toUpperCase().includes('BUY') || 
     r.verdict_action.toUpperCase().includes('LONG') ||
     r.verdict_action.toUpperCase() === 'ENTER')
  )

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Swing Trading Scanner</h1>
          <p className="text-sm text-slate-400 mt-1">
            End-of-day analysis for multi-day position opportunities
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            {providers.map((p) => (
              <div
                key={p.name}
                className={`px-2 py-1 rounded text-xs font-mono ${
                  p.available
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-slate-500/20 text-slate-500 border border-slate-500/30'
                }`}
              >
                {p.name}
              </div>
            ))}
          </div>

          {lastScanTime && (
            <span className="text-xs text-slate-500">
              Last scan: {lastScanTime}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {SWING_MODES.map((mode) => (
          <button
            key={mode.id}
            onClick={() => setSelectedMode(mode.id)}
            className={`p-4 rounded-lg border transition-all ${
              selectedMode === mode.id
                ? 'bg-cyan-500/10 border-cyan-500/50 ring-1 ring-cyan-500/30'
                : 'bg-[#0a1020] border-slate-700/50 hover:border-cyan-500/30'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className={`text-sm font-semibold ${
                selectedMode === mode.id ? 'text-cyan-400' : 'text-slate-300'
              }`}>
                {mode.label}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded ${
                mode.risk === 'low' ? 'bg-emerald-500/20 text-emerald-400' :
                mode.risk === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                'bg-red-500/20 text-red-400'
              }`}>
                {mode.risk.toUpperCase()}
              </span>
            </div>
            <p className="text-xs text-slate-500">{mode.description}</p>
            <p className="text-xs text-slate-600 mt-1">
              Uses configured universe
            </p>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={runSwingScan}
          disabled={isScanning}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            isScanning
              ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-400 hover:to-blue-400 shadow-lg shadow-cyan-500/20'
          }`}
        >
          {isScanning ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Scanning {modeConfig?.label}...
            </span>
          ) : (
            `Run ${modeConfig?.label} Scan`
          )}
        </button>

        {actionableSetups.length > 0 && (
          <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
            <span className="text-emerald-400 font-bold">{actionableSetups.length}</span>
            <span className="text-emerald-300 text-sm">actionable setups found</span>
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200">
          {error}
        </div>
      )}

      <div className="flex-1 flex gap-6 min-h-0">
        <div className="flex-1 overflow-auto">
          {scanResults.length > 0 ? (
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[#0a1020]">
                <tr className="text-left text-slate-400 border-b border-slate-700/50">
                  <th className="py-3 px-4">Symbol</th>
                  <th className="py-3 px-4 text-center">QuantraScore</th>
                  <th className="py-3 px-4">Regime</th>
                  <th className="py-3 px-4">Risk</th>
                  <th className="py-3 px-4">Verdict</th>
                  <th className="py-3 px-4 text-center">MR Fuse</th>
                </tr>
              </thead>
              <tbody>
                {scanResults.map((result) => {
                  const verdict = getVerdictBadge(result.verdict_action)
                  return (
                    <tr
                      key={result.symbol}
                      onClick={() => setSelectedSetup(result)}
                      className={`border-b border-slate-800/50 cursor-pointer transition-colors ${
                        selectedSetup?.symbol === result.symbol
                          ? 'bg-cyan-500/10'
                          : 'hover:bg-slate-800/30'
                      }`}
                    >
                      <td className="py-3 px-4 font-mono font-bold text-white">
                        {result.symbol}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`text-lg font-bold ${getScoreColor(result.quantrascore)}`}>
                          {result.quantrascore.toFixed(0)}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-300 capitalize">
                        {result.regime.replace(/_/g, ' ')}
                      </td>
                      <td className="py-3 px-4">
                        <span className={getRiskColor(result.risk_tier)}>
                          {result.risk_tier}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded border text-xs font-semibold ${verdict.bg} ${verdict.text}`}>
                          {result.verdict_action}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        {(result.monster_score ?? 0) > 0 ? (
                          <span className="text-amber-400 font-bold">
                            {((result.monster_score ?? 0) * 100).toFixed(0)}%
                          </span>
                        ) : (
                          <span className="text-slate-600">-</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-500">
              <div className="text-center">
                <div className="text-4xl mb-4">◎</div>
                <p>Select a mode and run a scan to find swing setups</p>
              </div>
            </div>
          )}
        </div>

        {selectedSetup && (
          <div className="w-80 bg-[#0a1020] border border-slate-700/50 rounded-lg p-4 overflow-auto">
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-white">{selectedSetup.symbol}</h3>
                <span className={`text-2xl font-bold ${getScoreColor(selectedSetup.quantrascore)}`}>
                  {selectedSetup.quantrascore.toFixed(0)}
                </span>
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {selectedSetup.regime.replace(/_/g, ' ')}
              </div>
            </div>

            <div className="space-y-3">
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Verdict</div>
                <div className={`text-lg font-bold ${getVerdictBadge(selectedSetup.verdict_action).text}`}>
                  {selectedSetup.verdict_action}
                </div>
              </div>

              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Risk Tier</div>
                <div className={`font-bold ${getRiskColor(selectedSetup.risk_tier)}`}>
                  {selectedSetup.risk_tier}
                </div>
              </div>

              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Entropy State</div>
                <div className="text-slate-300">{selectedSetup.entropy_state}</div>
              </div>

              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-400 mb-1">Suppression</div>
                <div className="text-slate-300">{selectedSetup.suppression_state}</div>
              </div>

              {(selectedSetup.monster_score ?? 0) > 0 && (
                <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                  <div className="text-xs text-amber-400 mb-1">MonsterRunner Fuse</div>
                  <div className="text-amber-400 font-bold">
                    {((selectedSetup.monster_score ?? 0) * 100).toFixed(0)}% probability
                  </div>
                  <div className="text-xs text-amber-400/70 mt-1">
                    {selectedSetup.monster_runner_fired?.join(', ') || 'No protocols fired'}
                  </div>
                </div>
              )}

              {selectedSetup.omega_alerts && selectedSetup.omega_alerts.length > 0 && (
                <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <div className="text-xs text-red-400 mb-1">Omega Alerts</div>
                  <div className="text-red-400 text-sm">
                    {selectedSetup.omega_alerts.join(', ')}
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-slate-700/50">
                <div className="text-xs text-slate-500 mb-2">Swing Trade Notes</div>
                <ul className="text-xs text-slate-400 space-y-1">
                  <li>• Daily timeframe analysis</li>
                  <li>• Hold period: 2-10 days typical</li>
                  <li>• Paper trade to validate</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-amber-900/20 border border-amber-500/30 rounded-lg">
        <div className="flex items-start gap-3">
          <span className="text-amber-400 text-xl">⚠</span>
          <div>
            <div className="text-amber-400 font-semibold text-sm">Research Mode Only</div>
            <p className="text-amber-200/70 text-xs mt-1">
              All outputs are structural probability analyses for research purposes only. 
              Not financial advice. Paper trade to validate before any real capital deployment.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
