import { useState } from 'react'
import { type ScanResult, api, type TraceResult, type MonsterRunnerResult, type SignalResult } from '../lib/api'

interface DetailPanelProps {
  symbol: ScanResult | null
  isLoading: boolean
}

function getScoreGradient(score: number): string {
  if (score >= 80) return 'from-emerald-500 to-emerald-600'
  if (score >= 60) return 'from-cyan-500 to-cyan-600'
  if (score >= 40) return 'from-amber-500 to-amber-600'
  if (score >= 20) return 'from-orange-500 to-orange-600'
  return 'from-red-500 to-red-600'
}

function MetricCard({ label, value, subvalue }: { label: string; value: string; subvalue?: string }) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
      {subvalue && <div className="text-xs text-slate-500 mt-0.5">{subvalue}</div>}
    </div>
  )
}

export function DetailPanel({ symbol, isLoading }: DetailPanelProps) {
  const [trace, setTrace] = useState<TraceResult | null>(null)
  const [monsterRunner, setMonsterRunner] = useState<MonsterRunnerResult | null>(null)
  const [signal, setSignal] = useState<SignalResult | null>(null)
  const [loadingTrace, setLoadingTrace] = useState(false)
  const [loadingMR, setLoadingMR] = useState(false)
  const [loadingSignal, setLoadingSignal] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'trace' | 'monster' | 'signal'>('overview')

  async function handleViewTrace() {
    if (!symbol?.window_hash) return
    setLoadingTrace(true)
    setActiveTab('trace')
    try {
      const data = await api.getTrace(symbol.window_hash)
      setTrace(data)
    } catch (err) {
      console.error('Failed to load trace:', err)
    } finally {
      setLoadingTrace(false)
    }
  }

  async function handleViewMonsterRunner() {
    if (!symbol?.symbol) return
    setLoadingMR(true)
    setActiveTab('monster')
    try {
      const data = await api.getMonsterRunner(symbol.symbol)
      setMonsterRunner(data)
    } catch (err) {
      console.error('Failed to load monster runner:', err)
    } finally {
      setLoadingMR(false)
    }
  }

  async function handleGenerateSignal() {
    if (!symbol?.symbol) return
    setLoadingSignal(true)
    setActiveTab('signal')
    try {
      const data = await api.generateSignal(symbol.symbol)
      setSignal(data)
    } catch (err) {
      console.error('Failed to generate signal:', err)
    } finally {
      setLoadingSignal(false)
    }
  }

  if (isLoading) {
    return (
      <div className="w-96 apex-card">
        <div className="animate-pulse space-y-4">
          <div className="h-6 w-24 bg-slate-700 rounded shimmer" />
          <div className="h-24 bg-slate-700 rounded-lg shimmer" />
          <div className="grid grid-cols-2 gap-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 bg-slate-700 rounded-lg shimmer" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!symbol) {
    return (
      <div className="w-96 apex-card flex items-center justify-center">
        <div className="text-center text-slate-500">
          <div className="text-4xl mb-3 opacity-30">◎</div>
          <div>Select a symbol to view details</div>
        </div>
      </div>
    )
  }

  const scoreGradient = getScoreGradient(symbol.quantrascore)

  return (
    <div className="w-96 apex-card flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h2 className="apex-heading">Symbol Details</h2>
        </div>
        <div className="text-lg font-bold text-slate-100">{symbol.symbol}</div>
      </div>

      <div className={`rounded-xl p-4 mb-4 bg-gradient-to-r ${scoreGradient} relative overflow-hidden`}>
        <div className="absolute top-2 right-2 opacity-20">
          <img
            src="/assets/quantra-q-icon.png"
            alt=""
            className="w-12 h-12"
            onError={(e) => e.currentTarget.style.display = 'none'}
          />
        </div>
        <div className="text-white/80 text-xs uppercase tracking-wider mb-1 flex items-center gap-1">
          <img
            src="/assets/quantra-q-icon.png"
            alt="Q"
            className="w-3 h-3"
            onError={(e) => e.currentTarget.style.display = 'none'}
          />
          QuantraScore
        </div>
        <div className="text-4xl font-bold text-white">{symbol.quantrascore.toFixed(1)}</div>
        <div className="text-white/70 text-sm mt-1 capitalize">{symbol.score_bucket}</div>
      </div>

      <div className="flex gap-1 mb-4 bg-slate-800/50 rounded-lg p-1">
        {(['overview', 'trace', 'monster', 'signal'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => {
              setActiveTab(tab)
              if (tab === 'trace' && !trace) handleViewTrace()
              if (tab === 'monster' && !monsterRunner) handleViewMonsterRunner()
              if (tab === 'signal' && !signal) handleGenerateSignal()
            }}
            className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors
              ${activeTab === tab
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-slate-400 hover:text-slate-200'
              }`}
          >
            {tab === 'overview' ? 'Overview' :
             tab === 'trace' ? 'Trace' :
             tab === 'monster' ? 'Monster' : 'Signal'}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto">
        {activeTab === 'overview' && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <MetricCard
                label="Regime"
                value={symbol.regime.replace('_', ' ')}
              />
              <MetricCard
                label="Risk Tier"
                value={symbol.risk_tier}
              />
              <MetricCard
                label="Entropy"
                value={symbol.entropy_state}
              />
              <MetricCard
                label="Suppression"
                value={symbol.suppression_state}
              />
              <MetricCard
                label="Drift"
                value={symbol.drift_state}
              />
              <MetricCard
                label="Protocols"
                value={symbol.protocol_fired_count.toString()}
                subvalue="fired"
              />
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Verdict</div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold text-slate-100 capitalize">
                  {symbol.verdict_action}
                </span>
                <span className="text-sm text-slate-400">
                  {(symbol.verdict_confidence * 100).toFixed(0)}% conf
                </span>
              </div>
            </div>

            {symbol.omega_alerts.length > 0 && (
              <div className="bg-red-900/20 rounded-lg p-3 border border-red-500/30">
                <div className="text-xs text-red-400 uppercase tracking-wider mb-2">Omega Alerts</div>
                <div className="flex flex-wrap gap-2">
                  {symbol.omega_alerts.map((alert) => (
                    <span
                      key={alert}
                      className="px-2 py-0.5 rounded text-xs font-medium bg-red-500/20 text-red-300"
                    >
                      {alert}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="space-y-3">
            {loadingTrace ? (
              <div className="animate-pulse space-y-3">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-16 bg-slate-700 rounded-lg shimmer" />
                ))}
              </div>
            ) : trace ? (
              <>
                <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Microtraits</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries(trace.microtraits).slice(0, 6).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-slate-400">{key.replace(/_/g, ' ')}</span>
                        <span className="text-slate-200 font-mono">{typeof value === 'number' ? value.toFixed(3) : String(value)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                    Protocols Fired ({trace.protocol_results.filter(p => p.fired).length})
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {trace.protocol_results
                      .filter(p => p.fired)
                      .slice(0, 12)
                      .map((p) => (
                        <span
                          key={p.protocol_id}
                          className="px-1.5 py-0.5 rounded text-xs font-mono bg-cyan-500/20 text-cyan-300"
                        >
                          {p.protocol_id}
                        </span>
                      ))}
                    {trace.protocol_results.filter(p => p.fired).length > 12 && (
                      <span className="text-xs text-slate-500">
                        +{trace.protocol_results.filter(p => p.fired).length - 12} more
                      </span>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center text-slate-500 py-8">
                Failed to load trace data
              </div>
            )}
          </div>
        )}

        {activeTab === 'monster' && (
          <div className="space-y-3">
            {loadingMR ? (
              <div className="animate-pulse space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-16 bg-slate-700 rounded-lg shimmer" />
                ))}
              </div>
            ) : monsterRunner ? (
              <>
                <div className={`rounded-lg p-3 border ${
                  monsterRunner.runner_state === 'primed'
                    ? 'bg-violet-900/20 border-violet-500/30'
                    : monsterRunner.runner_state === 'forming'
                    ? 'bg-amber-900/20 border-amber-500/30'
                    : 'bg-slate-800/50 border-slate-700/30'
                }`}>
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Runner State</div>
                  <div className="flex items-center justify-between">
                    <span className={`text-lg font-semibold capitalize ${
                      monsterRunner.runner_state === 'primed' ? 'text-violet-400' :
                      monsterRunner.runner_state === 'forming' ? 'text-amber-400' : 'text-slate-400'
                    }`}>
                      {monsterRunner.runner_state}
                    </span>
                    <span className="text-2xl font-bold text-slate-100">
                      {(monsterRunner.runner_probability * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <MetricCard
                    label="Event Class"
                    value={monsterRunner.rare_event_class}
                  />
                  <MetricCard
                    label="Primed Conf"
                    value={`${(monsterRunner.metrics.primed_confidence * 100).toFixed(0)}%`}
                  />
                  <MetricCard
                    label="Compression"
                    value={monsterRunner.metrics.compression_trace.toFixed(2)}
                  />
                  <MetricCard
                    label="Volume Pulse"
                    value={monsterRunner.metrics.volume_pulse.toFixed(2)}
                  />
                </div>
              </>
            ) : (
              <div className="text-center text-slate-500 py-8">
                Failed to load MonsterRunner data
              </div>
            )}
          </div>
        )}

        {activeTab === 'signal' && (
          <div className="space-y-3">
            {loadingSignal ? (
              <div className="animate-pulse space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-16 bg-slate-700 rounded-lg shimmer" />
                ))}
              </div>
            ) : signal ? (
              <>
                <div className={`rounded-lg p-3 border ${
                  signal.signal.direction === 'long'
                    ? 'bg-emerald-900/20 border-emerald-500/30'
                    : signal.signal.direction === 'short'
                    ? 'bg-red-900/20 border-red-500/30'
                    : 'bg-slate-800/50 border-slate-700/30'
                }`}>
                  <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Signal</div>
                  <div className="flex items-center justify-between">
                    <span className={`text-lg font-semibold capitalize ${
                      signal.signal.direction === 'long' ? 'text-emerald-400' :
                      signal.signal.direction === 'short' ? 'text-red-400' : 'text-slate-400'
                    }`}>
                      {signal.signal.direction} / {signal.signal.strength}
                    </span>
                    <span className="text-sm text-slate-400">
                      {(signal.signal.confidence * 100).toFixed(0)}% conf
                    </span>
                  </div>
                </div>

                {signal.signal.entry_price && (
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard
                      label="Entry"
                      value={`$${signal.signal.entry_price.toFixed(2)}`}
                    />
                    <MetricCard
                      label="Stop Loss"
                      value={signal.signal.stop_loss ? `$${signal.signal.stop_loss.toFixed(2)}` : 'N/A'}
                    />
                    <MetricCard
                      label="Target 1"
                      value={signal.signal.target_1 ? `$${signal.signal.target_1.toFixed(2)}` : 'N/A'}
                    />
                    <MetricCard
                      label="R:R"
                      value={signal.signal.risk_reward ? signal.signal.risk_reward.toFixed(2) : 'N/A'}
                    />
                  </div>
                )}

                {signal.signal.notes && (
                  <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Notes</div>
                    <div className="text-sm text-slate-300">{signal.signal.notes}</div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center text-slate-500 py-8">
                Failed to generate signal
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-700/30">
        <div className="text-xs text-slate-500 text-center">
          Hash: <span className="font-mono">{symbol.window_hash.slice(0, 16)}...</span>
        </div>
      </div>
    </div>
  )
}
