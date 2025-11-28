import { useState, useEffect } from 'react'
import { LeftRail } from './components/LeftRail'
import { Header } from './components/Header'
import { UniverseTable } from './components/UniverseTable'
import { DetailPanel } from './components/DetailPanel'
import { api, type ScanResult, type HealthResponse, type UniverseResult } from './lib/api'

export type NavItem = 'dashboard' | 'research' | 'apexlab' | 'models' | 'logs'

const DEFAULT_UNIVERSE = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'NFLX',
  'JPM', 'BAC', 'GS', 'MS', 'WFC', 'V', 'MA', 'PYPL', 'SQ', 'COIN'
]

const MODE_UNIVERSES: Record<string, string[]> = {
  demo: DEFAULT_UNIVERSE,
  mega_large_focus: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JPM', 'V', 'UNH', 'XOM', 'JNJ', 'WMT', 'MA'],
  mid_cap_focus: ['PLTR', 'SNAP', 'ROKU', 'COIN', 'HOOD', 'RBLX', 'DDOG', 'NET', 'CRWD', 'ZS', 'SNOW', 'TTD', 'SQ', 'AFRM', 'UPST'],
  high_vol_small_caps: ['CLOV', 'SOFI', 'MARA', 'RIOT', 'AMC', 'GME', 'PLUG', 'FCEL', 'LAZR', 'QS', 'CHPT', 'BLNK', 'TLRY', 'CGC', 'ACB'],
  low_float_runners: ['MULN', 'FFIE', 'IMPP', 'INDO', 'HUSA', 'CEI', 'PHUN', 'DWAC', 'BKKT', 'PROG', 'ASTS', 'IONQ', 'SAVA', 'OCGN', 'BNGO'],
  momentum_runners: ['NVDA', 'AMD', 'TSLA', 'MARA', 'RIOT', 'COIN', 'HOOD', 'SOFI', 'PLTR', 'RBLX', 'DDOG', 'NET', 'CRWD', 'SNOW', 'TTD'],
  full_us_equities: [...DEFAULT_UNIVERSE, 'PLTR', 'SNAP', 'ROKU', 'COIN', 'HOOD', 'RBLX', 'CLOV', 'SOFI', 'MARA', 'RIOT'],
  ci_test: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
}

export default function App() {
  const [activeNav, setActiveNav] = useState<NavItem>('dashboard')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [universeData, setUniverseData] = useState<UniverseResult | null>(null)
  const [selectedSymbol, setSelectedSymbol] = useState<ScanResult | null>(null)
  const [isScanning, setIsScanning] = useState(false)
  const [isScanningSymbol, setIsScanningSymbol] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scanMode, setScanMode] = useState('demo')

  useEffect(() => {
    loadHealth()
  }, [])

  async function loadHealth() {
    try {
      const data = await api.getHealth()
      setHealth(data)
    } catch (err) {
      console.error('Failed to load health:', err)
    }
  }

  async function handleRunScan() {
    setIsScanning(true)
    setError(null)
    try {
      const symbols = MODE_UNIVERSES[scanMode] || DEFAULT_UNIVERSE
      const result = await api.scanUniverse({
        symbols,
        timeframe: '1d',
        lookback_days: 150
      })
      setUniverseData(result)
      if (result.results.length > 0) {
        setSelectedSymbol(result.results[0])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed')
    } finally {
      setIsScanning(false)
    }
  }

  async function handleSymbolClick(result: ScanResult) {
    setSelectedSymbol(result)
    setIsScanningSymbol(true)
    try {
      const detailed = await api.scanSymbol({
        symbol: result.symbol,
        timeframe: '1d',
        lookback_days: 150
      })
      setSelectedSymbol(detailed)
    } catch (err) {
      console.error('Failed to get symbol details:', err)
    } finally {
      setIsScanningSymbol(false)
    }
  }

  function handleModeChange(mode: string) {
    setScanMode(mode)
    setUniverseData(null)
    setSelectedSymbol(null)
  }

  const isSmallCapMode = ['high_vol_small_caps', 'low_float_runners'].includes(scanMode)
  const isExtremeRiskMode = scanMode === 'low_float_runners'

  return (
    <div className="flex h-screen">
      <LeftRail activeNav={activeNav} onNavChange={setActiveNav} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header
          health={health}
          onRunScan={handleRunScan}
          isScanning={isScanning}
          scanMode={scanMode}
          onModeChange={handleModeChange}
          availableModes={Object.keys(MODE_UNIVERSES)}
        />

        <main className="flex-1 overflow-hidden p-6">
          {(isSmallCapMode || isExtremeRiskMode) && (
            <div className={`mb-4 p-4 rounded-lg border ${
              isExtremeRiskMode 
                ? 'bg-red-900/20 border-red-500/50 text-red-200' 
                : 'bg-amber-900/20 border-amber-500/50 text-amber-200'
            }`}>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span className="font-semibold">
                  {isExtremeRiskMode ? 'EXTREME RISK MODE' : 'SMALL-CAP MODE'}
                </span>
              </div>
              <p className="mt-1 text-sm opacity-80">
                {isExtremeRiskMode 
                  ? 'Low-float and penny stocks carry extreme volatility risk. These are structural analysis outputs only, not trading recommendations.'
                  : 'Small-cap and micro-cap stocks carry elevated risk. MonsterRunner fuse scores may indicate potential volatility events.'
                }
              </p>
            </div>
          )}

          <div className="h-full flex gap-6">
            <div className="flex-1 flex flex-col min-w-0">
              {error && (
                <div className="mb-4 p-4 bg-red-900/30 border border-red-500/50 rounded-lg text-red-200">
                  {error}
                </div>
              )}

              <UniverseTable
                data={universeData}
                selectedSymbol={selectedSymbol?.symbol}
                onSymbolClick={handleSymbolClick}
                isLoading={isScanning}
                scanMode={scanMode}
              />
            </div>

            <DetailPanel
              symbol={selectedSymbol}
              isLoading={isScanningSymbol}
            />
          </div>
        </main>

        <footer className="h-8 bg-[#030508] border-t border-[#0096ff]/20 px-6 flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-4">
            <span>QUANTRACORE APEX v9.0-A</span>
            <span className="text-[#0096ff]/50">|</span>
            <span>LAMONT LABS</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Research Mode Only</span>
            <span className="text-[#0096ff]/50">|</span>
            <span>Desktop Only</span>
            <span className="text-[#0096ff]/50">|</span>
            <span>Not Financial Advice</span>
          </div>
        </footer>
      </div>
    </div>
  )
}
