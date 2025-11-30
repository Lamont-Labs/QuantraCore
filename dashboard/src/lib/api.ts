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
  enabled: boolean
  model_variant: string
  model_dir: string
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

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
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

  // Backtest endpoints
  runBacktest(params: BacktestRequest): Promise<BacktestResult> {
    return request('/backtest', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  // ApexLab endpoints
  getApexLabStatus(): Promise<ApexLabStatusResponse> {
    return request('/apexlab/status')
  },

  startApexLabTraining(params: ApexLabTrainRequest): Promise<{ status: string; message: string; timestamp: string }> {
    return request('/apexlab/train', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  // Logs endpoints
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

// Backtest types
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

// ApexLab types
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

// Logs types
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
