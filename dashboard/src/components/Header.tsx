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
    <header className="h-16 bg-slate-900/70 border-b border-slate-700/30 px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <img
            src="/assets/quantra-q-icon.png"
            alt="Q"
            className="w-6 h-6"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
            }}
          />
          <h1 className="text-xl font-semibold bg-gradient-to-r from-cyan-400 to-violet-500 bg-clip-text text-transparent">
            QuantraCore Apex
          </h1>
        </div>

        <div className="flex items-center gap-3 text-sm">
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium
            ${health?.status === 'healthy'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
            }`}
          >
            {health?.engine || 'connecting...'}
          </span>

          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-violet-500/20 text-violet-300 border border-violet-500/30">
            RESEARCH MODE
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="text-sm text-slate-400 font-mono">
          {currentTime}
        </div>

        <div className="flex items-center gap-2">
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
