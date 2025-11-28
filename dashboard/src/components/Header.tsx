import { type HealthResponse } from '../lib/api'

interface HeaderProps {
  health: HealthResponse | null
  onRunScan: () => void
  isScanning: boolean
}

export function Header({ health, onRunScan, isScanning }: HeaderProps) {
  const currentTime = new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })

  return (
    <header className="h-16 bg-gradient-to-r from-[#0a0f1a] to-[#050810] border-b border-[#1e3a5f]/40 px-6 flex items-center justify-between">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-wider">
            <span className="text-[#00d4ff]">QUANTRACORE</span>
            <span className="text-slate-300 ml-2 font-normal">Apex</span>
          </h1>
        </div>

        <div className="flex items-center gap-3 text-sm">
          <span className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
            ${health?.status === 'healthy'
              ? 'bg-[#00d4ff]/10 text-[#00d4ff] border border-[#00d4ff]/30 shadow-sm shadow-[#00d4ff]/20'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
            }`}
          >
            {health?.engine || 'connecting...'}
          </span>

          <span className="px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-blue-500/10 text-blue-400 border border-blue-500/30">
            RESEARCH MODE
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
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

          <button className="apex-button-secondary">
            Run Stress Test
          </button>
        </div>
      </div>
    </header>
  )
}
