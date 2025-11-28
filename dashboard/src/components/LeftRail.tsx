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
    <aside className="w-72 bg-gradient-to-b from-[#0a0f1a] to-[#050810] border-r border-[#1e3a5f]/40 flex flex-col">
      <div className="p-5 border-b border-[#1e3a5f]/40">
        <div className="flex items-center gap-4 mb-5">
          <div className="w-14 h-14 rounded-lg overflow-hidden ring-2 ring-cyan-500/30 shadow-lg shadow-cyan-500/20">
            <img
              src="/assets/lamont_labs_logo.png"
              alt="Lamont Labs"
              className="w-full h-full object-cover"
            />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-wide">LAMONT LABS</div>
            <div className="text-xs text-cyan-400/70 italic">Obsession turned into systems.</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-lg overflow-hidden ring-2 ring-cyan-400/40 shadow-lg shadow-cyan-400/20">
            <img
              src="/assets/quantracore_disk.png"
              alt="QuantraCore"
              className="w-full h-full object-cover"
            />
          </div>
          <div>
            <div className="text-sm font-bold text-cyan-400 tracking-wider">QUANTRACORE</div>
            <div className="text-xs text-slate-400">AI Trading Intelligence Engine</div>
            <div className="text-[10px] text-slate-500 mt-0.5">Apex Engine v8.2</div>
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
                <span>{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 border-t border-[#1e3a5f]/40">
        <div className="text-center">
          <div className="text-[10px] uppercase tracking-widest text-cyan-500/50 mb-1">Desktop Build Only</div>
          <div className="text-xs text-slate-500 font-mono">GMKtec NucBox K6</div>
        </div>
      </div>
    </aside>
  )
}
