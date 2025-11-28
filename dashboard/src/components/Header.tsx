import { type HealthResponse } from '../lib/api'

interface HeaderProps {
  health: HealthResponse | null
  onRunScan: () => void
  isScanning: boolean
  scanMode: string
  onModeChange: (mode: string) => void
  availableModes: string[]
}

const SCAN_MODES = [
  { id: 'demo', label: 'Demo (20 symbols)', risk: 'low' },
  { id: 'mega_large_focus', label: 'Mega/Large Caps', risk: 'low' },
  { id: 'mid_cap_focus', label: 'Mid Caps', risk: 'medium' },
  { id: 'high_vol_small_caps', label: 'High Vol Small Caps', risk: 'high' },
  { id: 'low_float_runners', label: 'Low Float Runners', risk: 'extreme' },
  { id: 'momentum_runners', label: 'Momentum Runners', risk: 'high' },
  { id: 'full_us_equities', label: 'Full US (4000)', risk: 'mixed' },
  { id: 'ci_test', label: 'CI Test (5)', risk: 'low' },
]

export function Header({ health, onRunScan, isScanning, scanMode, onModeChange }: HeaderProps) {
  const currentTime = new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

  const selectedMode = SCAN_MODES.find(m => m.id === scanMode) || SCAN_MODES[0]
  const isHighRisk = selectedMode.risk === 'high' || selectedMode.risk === 'extreme'

  return (
    <header className="h-20 bg-gradient-to-r from-[#030508] via-[#0a0f1a] to-[#030508] border-b border-[#0096ff]/30 px-6 flex items-center justify-between relative overflow-hidden">
      <div className="circuit-line top-0 left-0 w-full"></div>
      
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4">
          <div className="relative w-12 h-12 flex items-center justify-center">
            <div className="absolute inset-0 bg-gradient-to-br from-[#0096ff]/20 to-[#00d4ff]/10 rounded-lg"></div>
            <div className="lamont-logo text-2xl font-bold text-[#00d4ff]">
              <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" stroke="url(#logoGrad)" strokeWidth="2" fill="none"/>
                <path d="M12 20 L20 12 L28 20 L20 28 Z" stroke="#00d4ff" strokeWidth="1.5" fill="rgba(0, 212, 255, 0.1)"/>
                <circle cx="20" cy="20" r="4" fill="#00d4ff"/>
                <defs>
                  <linearGradient id="logoGrad" x1="0" y1="0" x2="40" y2="40">
                    <stop offset="0%" stopColor="#0096ff"/>
                    <stop offset="100%" stopColor="#00d4ff"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </div>
          
          <div className="flex flex-col">
            <h1 className="text-xl font-bold tracking-wider">
              <span className="neon-text text-[#00d4ff]">LAMONT</span>
              <span className="text-slate-400 ml-1">LABS</span>
            </h1>
            <div className="flex items-center gap-2">
              <span className="text-[#00d4ff] text-xs font-semibold tracking-widest">QUANTRACORE</span>
              <span className="text-slate-500 text-xs">APEX v9.0-A</span>
            </div>
          </div>
        </div>

        <div className="h-8 w-px bg-gradient-to-b from-transparent via-[#0096ff]/40 to-transparent"></div>

        <div className="flex items-center gap-3 text-sm">
          <span className={`px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider
            ${health?.status === 'healthy'
              ? 'bg-[#00d4ff]/10 text-[#00d4ff] border border-[#00d4ff]/30 shadow-sm shadow-[#00d4ff]/20'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
            }`}
          >
            {health?.engine || 'connecting...'}
          </span>

          <span className="px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-[#0096ff]/10 text-[#0096ff] border border-[#0096ff]/30">
            RESEARCH MODE
          </span>

          <span className="px-3 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            DESKTOP ONLY
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex flex-col items-end gap-1">
          <label className="text-xs text-slate-500 uppercase tracking-wider">Scan Mode</label>
          <div className="flex items-center gap-2">
            <select
              value={scanMode}
              onChange={(e) => onModeChange(e.target.value)}
              className="apex-select min-w-[180px]"
              disabled={isScanning}
            >
              {SCAN_MODES.map(mode => (
                <option key={mode.id} value={mode.id}>
                  {mode.label}
                </option>
              ))}
            </select>
            {isHighRisk && (
              <span className="badge-extreme">
                {selectedMode.risk === 'extreme' ? 'EXTREME' : 'HIGH RISK'}
              </span>
            )}
          </div>
        </div>

        <div className="h-8 w-px bg-gradient-to-b from-transparent via-[#0096ff]/40 to-transparent"></div>

        <div className="text-sm text-slate-500 font-mono tracking-wider">
          {currentTime}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onRunScan}
            disabled={isScanning}
            className="apex-button disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isScanning ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Scanning...
              </span>
            ) : (
              'Run Scan'
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
