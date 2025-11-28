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

export default function App() {
  const [activeNav, setActiveNav] = useState<NavItem>('dashboard')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [universeData, setUniverseData] = useState<UniverseResult | null>(null)
  const [selectedSymbol, setSelectedSymbol] = useState<ScanResult | null>(null)
  const [isScanning, setIsScanning] = useState(false)
  const [isScanningSymbol, setIsScanningSymbol] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
      const result = await api.scanUniverse({
        symbols: DEFAULT_UNIVERSE,
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

  return (
    <div className="flex h-screen">
      <LeftRail activeNav={activeNav} onNavChange={setActiveNav} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header
          health={health}
          onRunScan={handleRunScan}
          isScanning={isScanning}
        />

        <main className="flex-1 overflow-hidden p-6">
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
              />
            </div>

            <DetailPanel
              symbol={selectedSymbol}
              isLoading={isScanningSymbol}
            />
          </div>
        </main>
      </div>
    </div>
  )
}
