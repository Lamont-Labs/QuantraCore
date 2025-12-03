import { type NavItem } from '../App'

interface LeftRailProps {
  activeNav: NavItem
  onNavChange: (nav: NavItem) => void
}

const navItems: { id: NavItem; label: string; icon: string; description?: string }[] = [
  { id: 'dashboard', label: 'Command Center', icon: '◉', description: 'Real-time overview' },
  { id: 'swing', label: 'Swing Scanner', icon: '◐', description: 'Multi-day setups' },
  { id: 'research', label: 'Research / Backtests', icon: '◎', description: 'Historical analysis' },
  { id: 'apexlab', label: 'ApexLab', icon: '⬡', description: 'Model training' },
  { id: 'models', label: 'ApexCore Models', icon: '◈', description: 'Neural network status' },
  { id: 'logs', label: 'Logs & Provenance', icon: '≡', description: 'Audit trail' },
  { id: 'investor', label: 'Investor Metrics', icon: '◇', description: 'Public dashboard' },
]

export function LeftRail({ activeNav, onNavChange }: LeftRailProps) {
  return (
    <aside className="w-72 bg-gradient-to-b from-[#0a0f1a] to-[#050810] border-r border-[#1e3a5f]/40 flex flex-col">
      <div className="p-5 border-b border-[#1e3a5f]/40">
        <div className="flex items-center gap-4 mb-5">
          <div className="w-14 h-14 rounded-lg overflow-hidden ring-2 ring-cyan-500/30 shadow-lg shadow-cyan-500/20 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
            <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none">
              <circle cx="20" cy="20" r="16" stroke="url(#llGrad)" strokeWidth="2" fill="none"/>
              <path d="M14 20 L20 14 L26 20 L20 26 Z" stroke="#00d4ff" strokeWidth="1.5" fill="rgba(0, 212, 255, 0.15)"/>
              <circle cx="20" cy="20" r="3" fill="#00d4ff"/>
              <defs>
                <linearGradient id="llGrad" x1="0" y1="0" x2="40" y2="40">
                  <stop offset="0%" stopColor="#0096ff"/>
                  <stop offset="100%" stopColor="#00d4ff"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-wide">LAMONT LABS</div>
            <div className="text-xs text-cyan-400/70 italic">Obsession turned into systems.</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-lg overflow-hidden ring-2 ring-cyan-400/40 shadow-lg shadow-cyan-400/20 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center">
            <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none">
              <circle cx="20" cy="20" r="14" stroke="#00d4ff" strokeWidth="1" fill="none" opacity="0.5"/>
              <circle cx="20" cy="20" r="8" stroke="#0096ff" strokeWidth="2" fill="rgba(0, 150, 255, 0.1)"/>
              <text x="20" y="24" textAnchor="middle" fill="#00d4ff" fontSize="12" fontWeight="bold">Q</text>
            </svg>
          </div>
          <div>
            <div className="text-sm font-bold text-cyan-400 tracking-wider">QUANTRACORE</div>
            <div className="text-xs text-slate-400">AI Trading Intelligence</div>
            <div className="text-[10px] text-slate-500 mt-0.5">Apex Engine v9.0-A</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4 mt-2">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onNavChange(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-300
                  ${activeNav === item.id
                    ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-cyan-300 hover:bg-[#0d1526] border border-transparent hover:border-cyan-500/20'
                  }`}
              >
                <span className={`text-lg ${activeNav === item.id ? 'text-cyan-400' : 'text-cyan-500/50'}`}>{item.icon}</span>
                <div className="flex-1 text-left">
                  <div>{item.label}</div>
                  {item.description && (
                    <div className={`text-[10px] ${activeNav === item.id ? 'text-cyan-400/70' : 'text-slate-500'}`}>
                      {item.description}
                    </div>
                  )}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 border-t border-[#1e3a5f]/40 space-y-3">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded bg-slate-800/50">
            <div className="text-[10px] text-slate-500">API</div>
            <div className="w-2 h-2 rounded-full bg-emerald-400 mx-auto mt-1 animate-pulse"></div>
          </div>
          <div className="p-2 rounded bg-slate-800/50">
            <div className="text-[10px] text-slate-500">ML</div>
            <div className="w-2 h-2 rounded-full bg-emerald-400 mx-auto mt-1 animate-pulse"></div>
          </div>
          <div className="p-2 rounded bg-slate-800/50">
            <div className="text-[10px] text-slate-500">Data</div>
            <div className="w-2 h-2 rounded-full bg-emerald-400 mx-auto mt-1 animate-pulse"></div>
          </div>
        </div>

        <div className="text-center">
          <div className="text-[10px] uppercase tracking-widest text-cyan-500/50 mb-1">Institutional Research Platform</div>
          <div className="text-xs text-slate-500 font-mono">Desktop Build Only</div>
        </div>
      </div>
    </aside>
  )
}
