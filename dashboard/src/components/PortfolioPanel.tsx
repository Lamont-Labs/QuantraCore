import { useState, useEffect, useCallback } from 'react'
import { api, type PortfolioSnapshot, type BrokerStatusResponse } from '../lib/api'

interface PortfolioPanelProps {
  compact?: boolean
}

export function PortfolioPanel({ compact = false }: PortfolioPanelProps) {
  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null)
  const [broker, setBroker] = useState<BrokerStatusResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  const loadData = useCallback(async () => {
    try {
      const [portfolioData, brokerData] = await Promise.all([
        api.getPortfolioSnapshot().catch(() => null),
        api.getBrokerStatus().catch(() => null),
      ])
      setPortfolio(portfolioData)
      setBroker(brokerData)
      setLastUpdate(new Date())
    } catch (err) {
      console.error('Failed to load portfolio:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 15000)
    return () => clearInterval(interval)
  }, [loadData])

  if (isLoading) {
    return (
      <div className="apex-card animate-pulse">
        <div className="h-6 w-32 bg-slate-700 rounded mb-4" />
        <div className="h-24 bg-slate-700 rounded-lg mb-4" />
        <div className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-12 bg-slate-700 rounded" />
          ))}
        </div>
      </div>
    )
  }

  const equity = broker?.equity ?? portfolio?.total_equity ?? 0
  const pnl = portfolio?.total_pnl ?? 0
  const pnlPct = equity > 0 ? (pnl / equity) * 100 : 0
  const isProfitable = pnl >= 0

  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Portfolio
        </h3>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
            broker?.is_paper 
              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30' 
              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
          }`}>
            {broker?.mode ?? 'PAPER'}
          </span>
          <span className="text-xs text-slate-500">{lastUpdate.toLocaleTimeString()}</span>
        </div>
      </div>

      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-4 mb-4 border border-slate-700/50">
        <div className="flex items-end justify-between">
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Total Equity</div>
            <div className="text-3xl font-bold text-white font-mono">
              ${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div className={`text-right ${isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
            <div className="text-lg font-bold font-mono">
              {isProfitable ? '+' : ''}{pnl >= 0 ? '$' : '-$'}{Math.abs(pnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className="text-sm opacity-80">
              {isProfitable ? '+' : ''}{pnlPct.toFixed(2)}%
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-slate-500 text-xs">Cash</div>
            <div className="text-white font-mono">${(portfolio?.cash ?? 0).toLocaleString()}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs">Positions</div>
            <div className="text-white font-mono">{broker?.position_count ?? portfolio?.positions?.length ?? 0}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs">Orders</div>
            <div className="text-white font-mono">{broker?.open_order_count ?? portfolio?.open_orders ?? 0}</div>
          </div>
        </div>
      </div>

      {!compact && portfolio?.positions && portfolio.positions.length > 0 && (
        <div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">Open Positions</div>
          <div className="space-y-2 max-h-48 overflow-auto">
            {portfolio.positions.slice(0, 5).map((pos) => {
              const positionPnl = pos.unrealized_pnl ?? 0
              const positionPnlPct = pos.avg_price > 0 ? ((pos.current_price - pos.avg_price) / pos.avg_price) * 100 : 0
              const isPositionProfitable = positionPnl >= 0

              return (
                <div
                  key={pos.symbol}
                  className="flex items-center justify-between p-2 rounded-lg bg-slate-800/50 border border-slate-700/30"
                >
                  <div>
                    <div className="font-semibold text-white">{pos.symbol}</div>
                    <div className="text-xs text-slate-400">
                      {pos.quantity} @ ${pos.avg_price.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-mono text-sm ${isPositionProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
                      {isPositionProfitable ? '+' : ''}{positionPnl >= 0 ? '$' : '-$'}{Math.abs(positionPnl).toFixed(2)}
                    </div>
                    <div className={`text-xs ${isPositionProfitable ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                      {isPositionProfitable ? '+' : ''}{positionPnlPct.toFixed(2)}%
                    </div>
                  </div>
                </div>
              )
            })}
            {portfolio.positions.length > 5 && (
              <div className="text-xs text-slate-500 text-center py-1">
                +{portfolio.positions.length - 5} more positions
              </div>
            )}
          </div>
        </div>
      )}

      {!compact && (!portfolio?.positions || portfolio.positions.length === 0) && (
        <div className="text-center py-6 text-slate-500">
          <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
          <div className="text-sm">No open positions</div>
        </div>
      )}
    </div>
  )
}
