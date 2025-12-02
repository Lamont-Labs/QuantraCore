import { useState, useEffect, useRef } from 'react'
import { api, type BrokerStatusResponse, type ComplianceScoreResponse, type DataProvidersResponse, type PredictiveStatusResponse, type MarketHoursResponse } from '../lib/api'
import { useVelocityMode } from '../hooks/useVelocityMode'

interface SystemStatusPanelProps {
  compact?: boolean
}

export function SystemStatusPanel({ compact = false }: SystemStatusPanelProps) {
  const [broker, setBroker] = useState<BrokerStatusResponse | null>(null)
  const [compliance, setCompliance] = useState<ComplianceScoreResponse | null>(null)
  const [providers, setProviders] = useState<DataProvidersResponse | null>(null)
  const [predictive, setPredictive] = useState<PredictiveStatusResponse | null>(null)
  const [marketHours, setMarketHours] = useState<MarketHoursResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [connectionState, setConnectionState] = useState<'connected' | 'connecting' | 'degraded'>('connecting')
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const { config } = useVelocityMode()
  const prevBrokerRef = useRef<BrokerStatusResponse | null>(null)
  const prevComplianceRef = useRef<ComplianceScoreResponse | null>(null)
  const prevProvidersRef = useRef<DataProvidersResponse | null>(null)
  const prevPredictiveRef = useRef<PredictiveStatusResponse | null>(null)
  const prevMarketHoursRef = useRef<MarketHoursResponse | null>(null)

  useEffect(() => {
    let mounted = true
    const refreshInterval = config?.refreshIntervals?.system || 30000
    let retryCount = 0
    const maxRetries = 3
    
    async function loadData() {
      try {
        const results = await Promise.allSettled([
          api.getBrokerStatus(),
          api.getComplianceScore(),
          api.getDataProviders(),
          api.getPredictiveStatus(),
          api.getMarketHours(),
        ])
        
        if (!mounted) return
        
        const [brokerResult, complianceResult, providersResult, predictiveResult, marketHoursResult] = results
        
        let successCount = 0
        
        if (brokerResult.status === 'fulfilled' && brokerResult.value) {
          prevBrokerRef.current = brokerResult.value
          setBroker(brokerResult.value)
          successCount++
          retryCount = 0
        } else if (prevBrokerRef.current) {
          setBroker(prevBrokerRef.current)
        }
        
        if (complianceResult.status === 'fulfilled' && complianceResult.value) {
          prevComplianceRef.current = complianceResult.value
          setCompliance(complianceResult.value)
          successCount++
        } else if (prevComplianceRef.current) {
          setCompliance(prevComplianceRef.current)
        }
        
        if (providersResult.status === 'fulfilled' && providersResult.value) {
          prevProvidersRef.current = providersResult.value
          setProviders(providersResult.value)
          successCount++
        } else if (prevProvidersRef.current) {
          setProviders(prevProvidersRef.current)
        }
        
        if (predictiveResult.status === 'fulfilled' && predictiveResult.value) {
          prevPredictiveRef.current = predictiveResult.value
          setPredictive(predictiveResult.value)
          successCount++
        } else if (prevPredictiveRef.current) {
          setPredictive(prevPredictiveRef.current)
        }
        
        if (marketHoursResult.status === 'fulfilled' && marketHoursResult.value) {
          prevMarketHoursRef.current = marketHoursResult.value
          setMarketHours(marketHoursResult.value)
          successCount++
        } else if (prevMarketHoursRef.current) {
          setMarketHours(prevMarketHoursRef.current)
        }
        
        if (successCount === 5) {
          setConnectionState('connected')
        } else if (successCount > 0) {
          setConnectionState('degraded')
        } else {
          setConnectionState('connecting')
        }
        
        setLastUpdate(new Date())
      } catch (err) {
        console.warn('SystemStatusPanel load error:', err)
        retryCount++
        if (retryCount < maxRetries) {
          setTimeout(loadData, 2000 * retryCount)
        }
      } finally {
        if (mounted) setIsLoading(false)
      }
    }
    
    loadData()
    const interval = setInterval(loadData, refreshInterval)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [config?.refreshIntervals?.system])


  const activeProviders = providers?.providers.filter(p => p.available).length ?? 0
  
  const getSessionStatus = (): 'operational' | 'warning' | 'exceptional' | 'error' => {
    if (!marketHours) return 'error'
    if (marketHours.current_session === 'regular') return 'exceptional'
    if (marketHours.trading_allowed) return 'operational'
    return 'warning'
  }
  
  const connectionColors = {
    connected: 'bg-emerald-400',
    connecting: 'bg-amber-400 animate-pulse',
    degraded: 'bg-amber-400'
  }
  
  const connectionLabels = {
    connected: 'ONLINE',
    connecting: 'CONNECTING...',
    degraded: 'DEGRADED'
  }
  
  return (
    <div className={`apex-card ${compact ? 'p-3' : ''}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connectionColors[connectionState]}`} />
          System Status
          {connectionState !== 'connected' && (
            <span className={`text-xs px-2 py-0.5 rounded ${connectionState === 'degraded' ? 'bg-amber-500/20 text-amber-400' : 'bg-blue-500/20 text-blue-400'}`}>
              {connectionLabels[connectionState]}
            </span>
          )}
        </h3>
        <span className="text-xs text-slate-500">
          {lastUpdate.toLocaleTimeString()}
        </span>
      </div>

      <div className={`grid ${compact ? 'grid-cols-2 gap-2' : 'grid-cols-5 gap-3'}`}>
        <StatusCard
          label="Market"
          value={marketHours?.session_display ?? 'Unknown'}
          status={getSessionStatus()}
          detail={marketHours?.trading_allowed ? 'Trading Active' : 'Trading Paused'}
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
      
        <StatusCard
          label="Broker"
          value={broker?.mode ?? 'OFFLINE'}
          status={broker ? 'operational' : 'error'}
          detail={broker ? `${broker.position_count} positions` : 'Not connected'}
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
        />

        <StatusCard
          label="Compliance"
          value={compliance ? `${compliance.overall_score.toFixed(1)}%` : 'N/A'}
          status={compliance ? (compliance.overall_score >= 95 ? 'exceptional' : compliance.overall_score >= 80 ? 'operational' : 'warning') : 'error'}
          detail={compliance?.excellence_level ?? 'Unknown'}
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          }
        />

        <StatusCard
          label="Data Feeds"
          value={`${activeProviders}/${providers?.providers.length ?? 0}`}
          status={activeProviders >= 2 ? 'operational' : activeProviders === 1 ? 'warning' : 'error'}
          detail={`${activeProviders} active feeds`}
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
            </svg>
          }
        />

        <StatusCard
          label="ApexCore V3"
          value={predictive?.model_loaded ? 'LOADED' : 'OFFLINE'}
          status={predictive?.model_loaded ? 'operational' : 'error'}
          detail={predictive?.metrics ? `${(predictive.metrics.runner_accuracy * 100).toFixed(1)}% runner acc` : 'Model not loaded'}
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          }
        />
      </div>

      {!compact && providers && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex flex-wrap gap-2">
            {providers.providers.map((provider) => (
              <span
                key={provider.name}
                className={`px-2 py-1 rounded text-xs font-mono ${
                  provider.available
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : 'bg-slate-700/50 text-slate-500 border border-slate-600/30'
                }`}
              >
                {provider.name}
                {provider.available && provider.rate_limit && (
                  <span className="ml-1 opacity-60">({provider.rate_limit}/s)</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {!compact && broker && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-slate-500 text-xs mb-1">Equity</div>
              <div className="text-white font-mono">${broker.equity.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs mb-1">Positions</div>
              <div className="text-white font-mono">{broker.position_count}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs mb-1">Open Orders</div>
              <div className="text-white font-mono">{broker.open_order_count}</div>
            </div>
          </div>
        </div>
      )}
      
      {!compact && marketHours && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400 uppercase tracking-wider">Extended Hours Trading</span>
            <span className={`text-xs px-2 py-0.5 rounded ${marketHours.extended_hours_enabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700 text-slate-400'}`}>
              {marketHours.extended_hours_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-4 text-xs">
            <div className={`p-2 rounded ${marketHours.current_session === 'pre_market' ? 'bg-amber-500/10 border border-amber-500/30' : 'bg-slate-800/50'}`}>
              <div className="text-amber-400 font-medium mb-1">Pre-Market</div>
              <div className="text-slate-400 font-mono">{marketHours.session_schedule.pre_market}</div>
            </div>
            <div className={`p-2 rounded ${marketHours.current_session === 'regular' ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-slate-800/50'}`}>
              <div className="text-emerald-400 font-medium mb-1">Regular</div>
              <div className="text-slate-400 font-mono">{marketHours.session_schedule.regular}</div>
            </div>
            <div className={`p-2 rounded ${marketHours.current_session === 'after_hours' ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-slate-800/50'}`}>
              <div className="text-purple-400 font-medium mb-1">After-Hours</div>
              <div className="text-slate-400 font-mono">{marketHours.session_schedule.after_hours}</div>
            </div>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            {marketHours.current_time_et} | {marketHours.is_extended_hours ? 'Extended Hours Active' : marketHours.current_session === 'regular' ? 'Regular Trading Session' : 'Market Closed'}
          </div>
        </div>
      )}
    </div>
  )
}

interface StatusCardProps {
  label: string
  value: string
  status: 'operational' | 'exceptional' | 'warning' | 'error'
  detail: string
  icon: React.ReactNode
}

function StatusCard({ label, value, status, detail, icon }: StatusCardProps) {
  const statusColors = {
    operational: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    exceptional: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
    warning: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
    error: 'text-red-400 bg-red-500/10 border-red-500/30',
  }

  const dotColors = {
    operational: 'bg-emerald-400',
    exceptional: 'bg-cyan-400',
    warning: 'bg-amber-400',
    error: 'bg-red-400',
  }

  return (
    <div className={`rounded-lg p-3 border ${statusColors[status]}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
        <span className={dotColors[status] + ' w-2 h-2 rounded-full'}></span>
      </div>
      <div className="flex items-center gap-2 mb-1">
        <span className="opacity-60">{icon}</span>
        <span className="font-bold text-sm">{value}</span>
      </div>
      <div className="text-xs opacity-70">{detail}</div>
    </div>
  )
}
