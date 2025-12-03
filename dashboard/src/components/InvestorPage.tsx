import { useState, useEffect } from 'react'
import { api, type InvestorMetricsResponse } from '../lib/api'

export function InvestorPage() {
  const [metrics, setMetrics] = useState<InvestorMetricsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  async function fetchMetrics() {
    try {
      const data = await api.getInvestorMetrics()
      setMetrics(data)
      setLastUpdated(new Date())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-[#030508] to-[#050810] flex items-center justify-center">
        <div className="text-cyan-400 text-xl animate-pulse">Loading Investor Metrics...</div>
      </div>
    )
  }

  if (error || !metrics) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-[#030508] to-[#050810] flex items-center justify-center">
        <div className="text-red-400 text-xl">Error: {error || 'No data available'}</div>
      </div>
    )
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#030508] to-[#050810] text-white p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        <header className="text-center space-y-4 pb-8 border-b border-cyan-500/20">
          <div className="flex items-center justify-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                {metrics.system.name}
              </h1>
              <p className="text-slate-400">{metrics.system.version} | {metrics.system.company}</p>
            </div>
          </div>
          <p className="text-xl text-slate-300">{metrics.system.tagline}</p>
          <div className="flex items-center justify-center gap-2">
            <span className={`w-2 h-2 rounded-full ${metrics.system.status === 'operational' ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`}></span>
            <span className="text-sm text-slate-400 uppercase tracking-wider">
              System {metrics.system.status}
            </span>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4">API Infrastructure</h3>
            <div className="text-4xl font-bold text-white mb-2">{metrics.capabilities.api_endpoints}+</div>
            <div className="text-slate-400">REST API Endpoints</div>
          </div>
          
          <div className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4">ML Models</h3>
            <div className="text-4xl font-bold text-white mb-2">{metrics.capabilities.ml_models_loaded}</div>
            <div className="text-slate-400">Ensemble Models Loaded</div>
          </div>
          
          <div className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
            <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4">Data Providers</h3>
            <div className="text-4xl font-bold text-white mb-2">{metrics.capabilities.data_provider_count}</div>
            <div className="text-slate-400">Real-Time Data Sources</div>
          </div>
        </section>

        {metrics.portfolio.status === 'connected' && (
          <section className="bg-[#0a0f14] rounded-xl border border-emerald-500/30 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                Live Portfolio (Paper Trading)
              </h3>
              <span className="text-xs text-slate-500">Real-time via Alpaca</span>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <div className="text-sm text-slate-400 mb-1">Total Equity</div>
                <div className="text-2xl font-bold text-white">
                  {formatCurrency(metrics.portfolio.total_equity || 0)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-slate-400 mb-1">Unrealized P&L</div>
                <div className={`text-2xl font-bold ${(metrics.portfolio.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatCurrency(metrics.portfolio.total_pnl || 0)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-slate-400 mb-1">Return</div>
                <div className={`text-2xl font-bold ${(metrics.portfolio.total_pnl_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatPercent(metrics.portfolio.total_pnl_pct || 0)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-slate-400 mb-1">Win Rate</div>
                <div className="text-2xl font-bold text-cyan-400">
                  {(metrics.portfolio.win_rate || 0).toFixed(0)}%
                </div>
                <div className="text-xs text-slate-500">
                  {metrics.portfolio.winners_count}W / {metrics.portfolio.losers_count}L
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="bg-[#0a0f14] rounded-xl border border-purple-500/30 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-purple-400 uppercase tracking-wider">
              Forward Validation (Bias-Free Proof)
            </h3>
            <span className={`text-xs px-2 py-1 rounded uppercase ${
              metrics.forward_validation.status === 'operational' 
                ? 'bg-emerald-500/20 text-emerald-400' 
                : 'bg-amber-500/20 text-amber-400'
            }`}>
              {metrics.forward_validation.status}
            </span>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <div className="text-sm text-slate-400 mb-1">Predictions Recorded</div>
              <div className="text-2xl font-bold text-white">
                {metrics.forward_validation.total_predictions}
              </div>
            </div>
            
            <div>
              <div className="text-sm text-slate-400 mb-1">Pending Outcomes</div>
              <div className="text-2xl font-bold text-amber-400">
                {metrics.forward_validation.pending_outcomes}
              </div>
            </div>
            
            <div>
              <div className="text-sm text-slate-400 mb-1">Validated</div>
              <div className="text-2xl font-bold text-emerald-400">
                {metrics.forward_validation.outcomes_checked}
              </div>
            </div>
            
            <div>
              <div className="text-sm text-slate-400 mb-1">True Precision</div>
              <div className="text-2xl font-bold text-cyan-400">
                {metrics.forward_validation.true_precision != null 
                  ? `${(metrics.forward_validation.true_precision * 100).toFixed(1)}%`
                  : 'Calculating...'}
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
            <p className="text-sm text-purple-200">
              {metrics.forward_validation.message || 
                'Forward validation records predictions BEFORE outcomes are known, proving accuracy without backtesting bias. Requires 30+ days of data for statistically significant results.'}
            </p>
          </div>
        </section>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <section className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
            <h3 className="text-lg font-semibold text-cyan-400 uppercase tracking-wider mb-4">
              Autonomous Features
            </h3>
            <div className="space-y-3">
              {metrics.capabilities.autonomous_features.map((feature, i) => (
                <div key={i} className="flex items-center gap-3 p-3 bg-cyan-500/5 rounded-lg border border-cyan-500/10">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                  <span className="text-slate-300">{feature}</span>
                  <span className="ml-auto text-xs text-emerald-400 uppercase">Active</span>
                </div>
              ))}
            </div>
          </section>

          <section className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
            <h3 className="text-lg font-semibold text-cyan-400 uppercase tracking-wider mb-4">
              Technology Stack
            </h3>
            <div className="space-y-3">
              {Object.entries(metrics.technology_stack).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-slate-500/5 rounded-lg border border-slate-500/10">
                  <span className="text-slate-400 capitalize">{key.replace('_', ' ')}</span>
                  <span className="text-slate-300 font-mono text-sm">{value}</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        <section className="bg-[#0a0f14] rounded-xl border border-amber-500/30 p-6">
          <h3 className="text-lg font-semibold text-amber-400 uppercase tracking-wider mb-6">
            Development Roadmap
          </h3>
          <div className="relative">
            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gradient-to-b from-emerald-500 via-amber-500 to-slate-600"></div>
            <div className="space-y-6">
              {metrics.roadmap.map((item, i) => (
                <div key={i} className="flex items-start gap-4 pl-8 relative">
                  <div className={`absolute left-2.5 w-3 h-3 rounded-full border-2 ${
                    item.status === 'active' ? 'bg-emerald-400 border-emerald-400' :
                    item.status === 'in_progress' ? 'bg-amber-400 border-amber-400' :
                    'bg-slate-600 border-slate-500'
                  }`}></div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-semibold text-slate-400">{item.phase}</span>
                      <span className={`text-xs px-2 py-0.5 rounded uppercase ${
                        item.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                        item.status === 'in_progress' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-slate-500/20 text-slate-400'
                      }`}>
                        {item.status.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="text-white mt-1">{item.milestone}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
          <h3 className="text-lg font-semibold text-cyan-400 uppercase tracking-wider mb-4">
            Data Providers
          </h3>
          <div className="flex flex-wrap gap-3">
            {metrics.capabilities.data_providers.map((provider, i) => (
              <div key={i} className="px-4 py-2 bg-slate-500/10 rounded-lg border border-slate-500/20">
                <span className="text-slate-300">{provider}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="bg-[#0a0f14] rounded-xl border border-cyan-500/20 p-6">
          <h3 className="text-lg font-semibold text-cyan-400 uppercase tracking-wider mb-4">
            ML Model Architecture
          </h3>
          <div className="flex flex-wrap gap-3">
            {metrics.capabilities.ml_model_names.map((model, i) => (
              <div key={i} className="px-4 py-2 bg-purple-500/10 rounded-lg border border-purple-500/20">
                <span className="text-purple-300 font-mono text-sm">{model}</span>
              </div>
            ))}
          </div>
        </section>

        <footer className="text-center py-8 border-t border-slate-700">
          <p className="text-slate-500 text-sm mb-2">
            Last updated: {lastUpdated?.toLocaleTimeString()}
          </p>
          <p className="text-slate-600 text-xs max-w-2xl mx-auto">
            This is a research and development platform. Not financial advice. 
            Paper trading only. Past performance does not guarantee future results.
            All autonomous trading features operate in simulation mode.
          </p>
        </footer>
      </div>
    </div>
  )
}
