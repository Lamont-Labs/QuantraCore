import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./lib/api', () => ({
  api: {
    getHealth: vi.fn().mockResolvedValue({
      status: 'healthy',
      timestamp: '2025-11-28T00:00:00Z',
      engine: 'operational',
      data_layer: 'operational',
    }),
    scanUniverse: vi.fn().mockResolvedValue({
      scan_count: 3,
      success_count: 3,
      error_count: 0,
      results: [
        {
          symbol: 'AAPL',
          quantrascore: 78.5,
          score_bucket: 'strong',
          regime: 'trending_up',
          risk_tier: 'medium',
          entropy_state: 'normal',
          suppression_state: 'none',
          drift_state: 'stable',
          verdict_action: 'hold',
          verdict_confidence: 0.75,
          omega_alerts: [],
          protocol_fired_count: 12,
          window_hash: 'abc123',
          timestamp: '2025-11-28T00:00:00Z',
        },
      ],
      errors: [],
      timestamp: '2025-11-28T00:00:00Z',
    }),
    scanSymbol: vi.fn().mockResolvedValue({
      symbol: 'AAPL',
      quantrascore: 78.5,
      score_bucket: 'strong',
      regime: 'trending_up',
      risk_tier: 'medium',
      entropy_state: 'normal',
      suppression_state: 'none',
      drift_state: 'stable',
      verdict_action: 'hold',
      verdict_confidence: 0.75,
      omega_alerts: [],
      protocol_fired_count: 12,
      window_hash: 'abc123',
      timestamp: '2025-11-28T00:00:00Z',
    }),
  },
}))

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the Lamont Labs branding', () => {
    render(<App />)
    expect(screen.getAllByText('LAMONT LABS').length).toBeGreaterThan(0)
  })

  it('renders the QuantraCore branding', () => {
    render(<App />)
    expect(screen.getAllByText('QUANTRACORE').length).toBeGreaterThan(0)
  })

  it('renders the navigation items', () => {
    render(<App />)
    expect(screen.getAllByText('Command Center').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Research / Backtests').length).toBeGreaterThan(0)
    expect(screen.getAllByText('ApexLab').length).toBeGreaterThan(0)
    expect(screen.getAllByText('ApexCore Models').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Logs & Provenance').length).toBeGreaterThan(0)
  })

  it('renders the Run Scan button', () => {
    render(<App />)
    expect(screen.getByText('Run Scan')).toBeInTheDocument()
  })

  it('shows placeholder message when no data', () => {
    render(<App />)
    expect(screen.getByText(/Click "Run Scan" to analyze the universe/)).toBeInTheDocument()
  })
})
