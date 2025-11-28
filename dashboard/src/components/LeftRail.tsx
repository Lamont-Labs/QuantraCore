import { type NavItem } from '../App'

interface LeftRailProps {
  activeNav: NavItem
  onNavChange: (nav: NavItem) => void
}

const navItems: { id: NavItem; label: string; icon: string }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: '◉' },
  { id: 'research', label: 'Research / Backtests', icon: '◎' },
  { id: 'apexlab', label: 'ApexLab', icon: '⬡' },
  { id: 'models', label: 'ApexCore Models', icon: '◈' },
  { id: 'logs', label: 'Logs & Provenance', icon: '≡' },
]

export function LeftRail({ activeNav, onNavChange }: LeftRailProps) {
  return (
    <aside className="w-64 bg-slate-900/70 border-r border-slate-700/30 flex flex-col">
      <div className="p-5 border-b border-slate-700/30">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center overflow-hidden">
            <img
              src="/assets/lamont-labs-logo.png"
              alt="Lamont Labs"
              className="w-full h-full object-cover"
              onError={(e) => {
                e.currentTarget.style.display = 'none'
                e.currentTarget.parentElement!.innerHTML = '<span class="text-white font-bold text-lg">L</span>'
              }}
            />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">Lamont Labs</div>
            <div className="text-xs text-slate-500">Research Division</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-slate-800 border border-cyan-500/30 flex items-center justify-center overflow-hidden">
            <img
              src="/assets/quantracore-disk.png"
              alt="QuantraCore"
              className="w-full h-full object-contain"
              onError={(e) => {
                e.currentTarget.style.display = 'none'
                e.currentTarget.parentElement!.innerHTML = '<span class="text-cyan-400 font-bold text-lg">Q</span>'
              }}
            />
          </div>
          <div>
            <div className="text-sm font-semibold text-cyan-400">QuantraCore</div>
            <div className="text-xs text-slate-500">Apex Engine v8.2</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3">
        <ul className="space-y-1">
          {navItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onNavChange(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200
                  ${activeNav === item.id
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
              >
                <span className="text-lg opacity-70">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 border-t border-slate-700/30">
        <div className="text-xs text-slate-500 text-center">
          <div className="mb-1">Desktop Build Only</div>
          <div className="text-slate-600">GMKtec NucBox K6</div>
        </div>
      </div>
    </aside>
  )
}
