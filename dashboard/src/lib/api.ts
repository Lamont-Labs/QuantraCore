const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export interface HealthResponse {
  status: string
  timestamp: string
  engine: string
  data_layer: string
}

export interface ScanRequest {
  symbol: string
  timeframe?: string
  lookback_days?: number
  seed?: number
}

export interface UniverseScanRequest {
  symbols: string[]
  timeframe?: string
  lookback_days?: number
}

export interface ScanResult {
  symbol: string
  quantrascore: number
  score_bucket: string
  regime: string
  risk_tier: string
  entropy_state: string
  suppression_state: string
  drift_state: string
  verdict_action: string
  verdict_confidence: number
  omega_alerts: string[]
  protocol_fired_count: number
  window_hash: string
  timestamp: string
  monster_score?: number
  monster_runner_fired?: string[]
}

export interface UniverseResult {
  scan_count: number
  success_count: number
  error_count: number
  results: ScanResult[]
  errors: { symbol: string; error: string }[]
  timestamp: string
}

export interface TraceResult {
  window_hash: string
  symbol: string
  microtraits: Record<string, number>
  entropy_metrics: Record<string, unknown>
  suppression_metrics: Record<string, unknown>
  drift_metrics: Record<string, unknown>
  continuation_metrics: Record<string, unknown>
  volume_metrics: Record<string, unknown>
  protocol_results: Array<{
    protocol_id: string
    fired: boolean
    confidence: number
    signal_type?: string
    details: Record<string, unknown>
  }>
  verdict: {
    action: string
    confidence: number
  }
  omega_overrides: Record<string, unknown>
}

export interface PortfolioSnapshot {
  snapshot: {
    total_equity: number
    cash: number
    total_pnl: number
    total_realized_pnl: number
    total_unrealized_pnl: number
    position_count: number
    sector_exposure: Record<string, number>
    timestamp: string
  }
  positions: Array<{
    symbol: string
    quantity: number
    avg_price: number
    current_price: number
    unrealized_pnl: number
    market_value?: number
    side?: string
  }>
  open_orders: number
  cash: number
  total_equity: number
  total_pnl: number
  timestamp: string
}

export interface MonsterRunnerResult {
  symbol: string
  runner_probability: number
  runner_state: string
  rare_event_class: string
  metrics: {
    compression_trace: number
    entropy_floor: number
    volume_pulse: number
    range_contraction: number
    primed_confidence: number
  }
  compliance_note: string
}

export interface RiskAssessment {
  symbol: string
  risk_assessment: {
    symbol: string
    risk_tier: string
    composite_score: number
    permission: string
    denial_reasons: string[]
    factors: Record<string, number>
    omega_locks: string[]
    timestamp: string
  }
  underlying_analysis: {
    quantrascore: number
    regime: string
    entropy_state: string
  }
  timestamp: string
}

export interface SignalResult {
  symbol: string
  signal: {
    symbol: string
    direction: string
    strength: string
    entry_price: number | null
    stop_loss: number | null
    target_1: number | null
    target_2: number | null
    risk_reward: number | null
    confidence: number
    notes: string
    generated_at: string
  }
  risk_tier: string
  timestamp: string
}

export interface PredictiveStatusResponse {
  version: string
  status: string
  model_loaded: boolean
  enabled: boolean
  model_variant: string
  model_dir: string
  training_samples?: number
  trained_at?: string
  metrics?: {
    quantrascore_rmse: number
    runner_accuracy: number
    quality_accuracy: number
    avoid_accuracy: number
    regime_accuracy: number
    timing_accuracy: number
    runup_rmse: number
  }
  heads?: {
    core: number
    optional: number
    total: number
  }
  runner_threshold: number
  avoid_threshold: number
  max_disagreement: number
  compliance_note: string
  timestamp: string
}

export interface PredictiveAdvisoryRequest {
  symbol: string
  timeframe?: string
  lookback_days?: number
}

export interface PredictiveAdvisoryResponse {
  symbol: string
  base_quantra_score: number
  model_quantra_score: number
  runner_prob: number
  quality_tier: string
  avoid_trade_prob: number
  ensemble_disagreement: number
  recommendation: string
  confidence: number
  reasons: string[]
  engine_quantra_score: number
  engine_regime: string
  engine_risk_tier: string
  predictive_status: string
  compliance_note: string
  timestamp: string
}

export interface ModelInfoResponse {
  status: string
  manifest_count?: number
  latest_manifest?: Record<string, unknown>
  available_manifests?: string[]
  model_variant?: string
  created_at?: string
  message?: string
  compliance_note: string
  timestamp: string
}

export interface BatchAdvisoryRequest {
  symbols: string[]
  timeframe?: string
  lookback_days?: number
  max_results?: number
}

export interface BatchAdvisoryResponse {
  total_requested: number
  total_processed: number
  total_errors: number
  uprank_count: number
  avoid_count: number
  predictive_status: string
  results: PredictiveAdvisoryResponse[]
  errors: { symbol: string; error: string }[]
  compliance_note: string
  timestamp: string
}

export interface BrokerStatusResponse {
  mode: string
  adapter: string
  is_paper: boolean
  equity: number
  position_count: number
  open_order_count: number
  daily_turnover: number
  config: {
    execution_mode: string
    default_account: string
    alpaca_paper_configured: boolean
    alpaca_live_enabled: boolean
    risk: {
      max_notional_exposure_usd: number
      max_positions: number
      block_short_selling: boolean
    }
  }
  safety_note: string
  timestamp: string
}

export interface ComplianceScoreResponse {
  overall_score: number
  excellence_level: string
  timestamp: string
  metrics: {
    determinism_iterations: number
    stress_test_multiplier: number
    latency_margin_ms: number
    audit_completeness: number
    proof_integrity: number
    omega_directive_adherence: number
  }
  standards_met: string[]
  standards_exceeded: string[]
  areas_of_excellence: string[]
  compliance_mode: string
}

export interface DataProviderStatus {
  name: string
  available: boolean
  rate_limit: number | null
}

export interface DataProvidersResponse {
  providers: DataProviderStatus[]
  active_count: number
  timestamp: string
}

export interface TradingSetup {
  symbol: string
  quantrascore: number
  current_price: number
  entry: number
  stop: number
  target: number
  shares: number
  position_value: number
  risk_amount: number
  reward_amount: number
  risk_reward: number
  conviction: string
  regime: string
  timing: string
}

export interface TradingSetupsResponse {
  setups: TradingSetup[]
  count: number
  timestamp: string
}

export interface RunnerScreenerResponse {
  detected_runners: Array<{
    symbol: string
    runner_probability: number
    quantrascore: number
    volume_surge: number
    price_change_pct: number
    runner_state: string
    current_price: number
    detected_at: string
  }>
  scan_count: number
  detection_count: number
  last_scan: string
  timestamp: string
}

export interface AutoTraderStatusResponse {
  enabled: boolean
  mode: string
  last_scan: string | null
  last_trade: string | null
  today_trades: number
  today_pnl: number
  active_positions: number
  pending_orders: number
  daily_limit_reached: boolean
  config: {
    max_daily_trades: number
    min_quantrascore: number
    max_position_size: number
    risk_per_trade: number
  }
  recent_trades: Array<{
    symbol: string
    side: string
    quantity: number
    price: number
    timestamp: string
    pnl?: number
  }>
  timestamp: string
}

export interface PositionContinuationData {
  symbol: string
  entry_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  continuation: {
    probability: number
    reversal_probability: number
    trend_strength: number
    momentum_status: string
    exhaustion_level: number
    confidence: number
  }
  decision: {
    hold_decision: string
    hold_reason: string
    suggested_action: string
    adjusted_stop: number | null
    adjusted_target: number | null
    hold_extension_bars: number
  }
  last_update: string
  compliance_note: string
}

export interface ContinuationAnalysisResponse {
  positions: PositionContinuationData[]
  summary: {
    total_positions: number
    decisions?: Record<string, number>
    avg_continuation: number
    positions_at_risk: number
  }
  config?: {
    strong_hold_threshold: number
    normal_hold_threshold: number
    reduce_threshold: number
    exit_threshold: number
  }
  compliance_note: string
  timestamp: string
  error?: string
}

export interface SignalsListResponse {
  signals: Array<{
    symbol: string
    direction: string
    strength: string
    entry_price: number
    stop_loss: number
    target: number
    quantrascore: number
    conviction: string
    timing: string
    predicted_top: number | null
    generated_at: string
  }>
  count: number
  high_conviction_count: number
  timestamp: string
}

export interface ContinuousLearningStatusResponse {
  state: string
  running: boolean
  total_cycles: number
  total_samples_processed: number
  cycles_without_improvement: number
  last_training: string
  cache_size: number
  current_cycle: string | null
  config: {
    learning_interval_minutes: number
    min_new_samples_for_training: number
    max_samples_cache: number
    feature_drift_threshold: number
    label_drift_threshold: number
    performance_drop_threshold: number
    validation_holdout_ratio: number
    min_accuracy_improvement: number
    max_cycles_without_improvement: number
    warm_start_enabled: boolean
    multi_pass_epochs: number
    sliding_window_overlap: number
    symbols: string[]
    lookback_days: number
  }
}

export interface IncrementalLearningStatusResponse {
  status: string
  learning_mode: string
  features: string[]
  model: {
    version: string
    model_size: string
    is_fitted: boolean
    decay_halflife_days: number
    buffer: {
      anchor_samples: number
      recency_samples: number
      total_samples: number
      rare_patterns: number
      runner_ratio: number
    }
    heads: {
      quantrascore_trees: number
      runner_trees: number
      quality_trees: number
      avoid_trees: number
      regime_trees: number
      timing_trees: number
      runup_trees: number
    }
    manifest: Record<string, unknown> | null
  }
  timestamp: string
}

async function request<T>(endpoint: string, options?: RequestInit, retries = 2): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'apex-dev-key',
          ...options?.headers,
        },
      })

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(error.detail || `HTTP ${response.status}`)
      }

      return response.json()
    } catch (err) {
      const isLastAttempt = attempt === retries
      
      if (err instanceof TypeError) {
        if (!isLastAttempt) {
          await new Promise(r => setTimeout(r, 500 * (attempt + 1)))
          continue
        }
        throw new Error(`Network error: ${err.message}`)
      }
      
      if (!isLastAttempt && err instanceof Error && err.message.includes('Network')) {
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)))
        continue
      }
      
      throw err
    }
  }
  
  throw new Error(`Failed after ${retries + 1} attempts`)
}

export const api = {
  getHealth(): Promise<HealthResponse> {
    return request('/health')
  },

  scanSymbol(params: ScanRequest): Promise<ScanResult> {
    return request('/scan_symbol', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  scanUniverse(params: UniverseScanRequest): Promise<UniverseResult> {
    return request('/scan_universe', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getTrace(windowHash: string): Promise<TraceResult> {
    return request(`/trace/${windowHash}`)
  },

  getPortfolioSnapshot(): Promise<PortfolioSnapshot> {
    return request('/portfolio/status')
  },

  getMonsterRunner(symbol: string): Promise<MonsterRunnerResult> {
    return request(`/monster_runner/${symbol}`, { method: 'POST' })
  },

  getRiskAssessment(symbol: string): Promise<RiskAssessment> {
    return request(`/risk/assess/${symbol}`, { method: 'POST' })
  },

  generateSignal(symbol: string): Promise<SignalResult> {
    return request(`/signal/generate/${symbol}`, { method: 'POST' })
  },

  getPredictiveStatus(): Promise<PredictiveStatusResponse> {
    return request('/predictive/status')
  },

  getPredictiveAdvise(params: PredictiveAdvisoryRequest): Promise<PredictiveAdvisoryResponse> {
    return request('/predictive/advise', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getModelInfo(): Promise<ModelInfoResponse> {
    return request('/predictive/model_info')
  },

  getBatchAdvise(params: BatchAdvisoryRequest): Promise<BatchAdvisoryResponse> {
    return request('/predictive/batch_advise', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getBrokerStatus(): Promise<BrokerStatusResponse> {
    return request('/broker/status')
  },

  getComplianceScore(): Promise<ComplianceScoreResponse> {
    return request('/compliance/score')
  },

  getDataProviders(): Promise<DataProvidersResponse> {
    return request('/data_providers')
  },

  getTradingSetups(topN: number = 10, minScore: number = 50): Promise<TradingSetupsResponse> {
    return request(`/trading/setups?top_n=${topN}&min_score=${minScore}`)
  },

  getRunnerScreener(): Promise<RunnerScreenerResponse> {
    return request('/screener/alerts')
  },

  getAutoTraderStatus(): Promise<AutoTraderStatusResponse> {
    return request('/autotrader/status')
  },

  getContinuationAnalysis(): Promise<ContinuationAnalysisResponse> {
    return request('/positions/continuation')
  },

  getSignalsList(): Promise<SignalsListResponse> {
    return request('/signals/live')
  },

  getSmsStatus(): Promise<{ enabled: boolean; phone_configured: boolean; alerts_sent_today: number; config: { min_quantrascore: number; min_conviction: string; alert_cooldown_minutes: number }; timestamp: string }> {
    return request('/sms/status')
  },

  getScreenerStatus(): Promise<{ enabled: boolean; scanning: boolean; symbols_monitored: number; alerts_today: number; config: { min_volume_surge: number; min_price_change: number; max_float_shares: number; scan_interval_seconds: number }; timestamp: string }> {
    return request('/screener/status')
  },

  getContinuousLearningStatus(): Promise<ContinuousLearningStatusResponse> {
    return request('/apexlab/continuous/status')
  },

  getIncrementalLearningStatus(): Promise<IncrementalLearningStatusResponse> {
    return request('/apexlab/incremental/status')
  },

  reloadModels(): Promise<{ status: string; message: string; timestamp: string }> {
    return request('/model/reload', { method: 'POST' })
  },

  runBacktest(params: BacktestRequest): Promise<BacktestResult> {
    return request('/backtest', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getApexLabStatus(): Promise<ApexLabStatusResponse> {
    return request('/apexlab/status')
  },

  startApexLabTraining(params: ApexLabTrainRequest): Promise<{ status: string; message: string; timestamp: string }> {
    return request('/apexlab/train', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getSystemLogs(params?: LogsQueryParams): Promise<LogsResponse> {
    const queryParams = new URLSearchParams()
    if (params?.level) queryParams.append('level', params.level)
    if (params?.limit) queryParams.append('limit', params.limit.toString())
    if (params?.offset) queryParams.append('offset', params.offset.toString())
    const query = queryParams.toString() ? `?${queryParams.toString()}` : ''
    return request(`/logs/system${query}`)
  },

  getProvenanceRecords(limit?: number): Promise<ProvenanceResponse> {
    const query = limit ? `?limit=${limit}` : ''
    return request(`/logs/provenance${query}`)
  },
}

export interface BacktestRequest {
  symbol: string
  start_date?: string
  end_date?: string
  lookback_days?: number
  timeframe?: string
}

export interface BacktestResult {
  symbol: string
  start_date: string
  end_date: string
  trades: number
  win_count: number
  loss_count: number
  win_rate: number
  total_return: number
  avg_return: number
  sharpe_ratio: number
  max_drawdown: number
  avg_quantrascore: number
  regime_distribution: Record<string, number>
  protocol_frequency: Record<string, number>
  timestamp: string
}

export interface ApexLabStatusResponse {
  version: string
  schema_fields: number
  training_samples: number
  last_training: string | null
  is_training: boolean
  progress: number
  current_step: string
  logs: string[]
  manifests_available: number
}

export interface ApexLabTrainRequest {
  symbols?: string[]
  lookback_days?: number
  timeframe?: string
}

export interface LogEntry {
  timestamp: string
  level: string
  component: string
  message: string
  file?: string
}

export interface LogsResponse {
  logs: LogEntry[]
  total_count: number
  has_more: boolean
}

export interface LogsQueryParams {
  level?: string
  component?: string
  limit?: number
  offset?: number
}

export interface ProvenanceRecord {
  hash: string
  timestamp: string
  symbol: string
  quantrascore: number
  protocols_fired: number
  regime: string
  risk_tier: string
}

export interface ProvenanceResponse {
  records: ProvenanceRecord[]
  count: number
  note: string
  timestamp: string
}
