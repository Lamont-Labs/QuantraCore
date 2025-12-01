import { useState, useEffect, useRef } from 'react'
import { api, type TradingSetupsResponse, type TradingSetup } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface TradingSetupsPanelProps {
  onSymbolSelect?: (symbol: string) => void
}

export function TradingSetupsPanel({ onSymbolSelect }: TradingSetupsPanelProps) {
  const [setups, setSetups] = useState<TradingSetupsResponse | null>(null)
  const [selectedSetup, setSelectedSetup] = useState<TradingSetup | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [newSymbols, setNewSymbols] = useState<Set<string>>(new Set())
  const { config, isHighVelocity, isTurbo } = useVelocityMode()
  const prevSetupsRef = useRef<Map<string, number>>(new Map())
  const selectedSetupRef = useRef<TradingSetup | null>(null)

  useEffect(() => {
    selectedSetupRef.current = selectedSetup
  }, [selectedSetup])

  useEffect(() => {
    let mounted = true
    const refreshInterval = config?.refreshIntervals?.setups || 15000
    
    async function loadData() {
      try {
        const data = await api.getTradingSetups(10, 50).catch(() => null)
        if (!mounted) return
        
        if (data && data.setups) {
          const currentScores = new Map(data.setups.map(s => [s.symbol, s.quantrascore]))
          const changedSymbols = new Set<string>()
          
          data.setups.forEach(s => {
            const prevScore = prevSetupsRef.current.get(s.symbol)
            if (prevScore !== undefined && Math.abs(s.quantrascore - prevScore) > 2) {
              changedSymbols.add(s.symbol)
            }
          })
          
          if (changedSymbols.size > 0) {
            setNewSymbols(changedSymbols)
            setTimeout(() => mounted && setNewSymbols(new Set()), 1000)
          }
          
          prevSetupsRef.current = currentScores
          setSetups(data)
          if (data.setups.length > 0 && !selectedSetupRef.current) {
            setSelectedSetup(data.setups[0])
          }
        }
        setLastUpdate(new Date())
      } catch (err) {
        console.warn('TradingSetupsPanel load error:', err)
      }
    }
    
    loadData()
    const interval = setInterval(loadData, refreshInterval)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [config?.refreshIntervals?.setups])

  function handleSetupClick(setup: TradingSetup) {
    setSelectedSetup(setup)
    onSymbolSelect?.(setup.symbol)
  }

  function getScoreColor(score: number): string {
    if (score >= 80) return 'text-emerald-400'
    if (score >= 60) return 'text-cyan-400'
    if (score >= 40) return 'text-amber-400'
    return 'text-slate-400'
  }

  function getConvictionBadge(conviction: string): { bg: string; text: string } {
    switch (conviction.toLowerCase()) {
      case 'high':
        return { bg: 'bg-emerald-500/20 border-emerald-500/50', text: 'text-emerald-400' }
      case 'medium':
        return { bg: 'bg-cyan-500/20 border-cyan-500/50', text: 'text-cyan-400' }
      default:
        return { bg: 'bg-slate-500/20 border-slate-500/50', text: 'text-slate-400' }
    }
  }


  return (
    <div className={`apex-card ${isTurbo ? 'border-red-500/30' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Top Setups
          {setups && <span className="text-xs text-slate-500 ml-2">({setups.count})</span>}
          {isHighVelocity && (
            <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded ${isTurbo ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
              {config.refreshIntervals.setups / 1000}s
            </span>
          )}
        </h3>
        <span className="text-xs text-slate-500">{lastUpdate.toLocaleTimeString()}</span>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 space-y-2 max-h-80 overflow-auto">
          {setups?.setups.map((setup, index) => {
            const convictionBadge = getConvictionBadge(setup.conviction)
            const isSelected = selectedSetup?.symbol === setup.symbol
            const hasChanged = newSymbols.has(setup.symbol)

            return (
              <div
                key={setup.symbol}
                onClick={() => handleSetupClick(setup)}
                className={`p-3 rounded-lg cursor-pointer transition-all border ${
                  hasChanged
                    ? 'bg-amber-500/20 border-amber-500/50 animate-pulse'
                    : isSelected
                    ? 'bg-cyan-500/10 border-cyan-500/50'
                    : 'bg-slate-800/50 border-slate-700/30 hover:border-cyan-500/30'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 w-4">#{index + 1}</span>
                    <span className={`font-bold ${hasChanged ? 'text-amber-400' : 'text-white'}`}>{setup.symbol}</span>
                    {hasChanged && <span className="text-xs text-amber-400">UPDATED</span>}
                    <span className={`px-1.5 py-0.5 rounded text-xs border ${convictionBadge.bg} ${convictionBadge.text}`}>
                      {setup.conviction}
                    </span>
                  </div>
                  <span className={`text-lg font-bold ${getScoreColor(setup.quantrascore)}`}>
                    {setup.quantrascore.toFixed(1)}
                  </span>
                </div>

                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500">Entry</span>
                    <div className="text-white font-mono">${setup.entry.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Stop</span>
                    <div className="text-red-400 font-mono">${setup.stop.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">Target</span>
                    <div className="text-emerald-400 font-mono">${setup.target.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-slate-500">R:R</span>
                    <div className="text-cyan-400 font-mono">{setup.risk_reward.toFixed(1)}</div>
                  </div>
                </div>
              </div>
            )
          })}

          {(!setups || setups.setups.length === 0) && (
            <div className="text-center py-8 text-slate-500">
              <div className="text-2xl mb-2">◎</div>
              <div className="text-sm">No actionable setups found</div>
              <div className="text-xs mt-1">Market may be closed or no high-conviction opportunities</div>
            </div>
          )}
        </div>

        {selectedSetup && (
          <div className="w-56 bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
            <div className="text-center mb-3">
              <div className="text-lg font-bold text-white">{selectedSetup.symbol}</div>
              <div className={`text-2xl font-bold ${getScoreColor(selectedSetup.quantrascore)}`}>
                {selectedSetup.quantrascore.toFixed(1)}
              </div>
              <div className="text-xs text-slate-400 capitalize">{selectedSetup.regime.replace(/_/g, ' ')}</div>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Current</span>
                <span className="text-white font-mono">${selectedSetup.current_price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Position Size</span>
                <span className="text-white font-mono">{selectedSetup.shares} shares</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Position Value</span>
                <span className="text-white font-mono">${selectedSetup.position_value.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Risk Amount</span>
                <span className="text-red-400 font-mono">${selectedSetup.risk_amount.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Reward Amount</span>
                <span className="text-emerald-400 font-mono">${selectedSetup.reward_amount.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Timing</span>
                <span className="text-cyan-400">{selectedSetup.timing || 'none'}</span>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <div className="text-xs text-amber-400/70 text-center">
                Research analysis only - not trading advice
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
