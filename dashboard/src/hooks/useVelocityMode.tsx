import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

export type VelocityMode = 'standard' | 'high' | 'turbo'

interface VelocityConfig {
  mode: VelocityMode
  refreshIntervals: {
    system: number
    portfolio: number
    setups: number
    models: number
  }
  label: string
  description: string
}

const VELOCITY_CONFIGS: Record<VelocityMode, VelocityConfig> = {
  standard: {
    mode: 'standard',
    refreshIntervals: {
      system: 30000,
      portfolio: 15000,
      setups: 60000,
      models: 60000,
    },
    label: 'Standard',
    description: 'Normal refresh rates for research',
  },
  high: {
    mode: 'high',
    refreshIntervals: {
      system: 5000,
      portfolio: 3000,
      setups: 10000,
      models: 30000,
    },
    label: 'High Velocity',
    description: 'Fast updates for active trading',
  },
  turbo: {
    mode: 'turbo',
    refreshIntervals: {
      system: 2000,
      portfolio: 1000,
      setups: 5000,
      models: 15000,
    },
    label: 'Turbo',
    description: 'Maximum speed for scalping',
  },
}

interface VelocityContextType {
  mode: VelocityMode
  config: VelocityConfig
  setMode: (mode: VelocityMode) => void
  isHighVelocity: boolean
  isTurbo: boolean
}

const VelocityContext = createContext<VelocityContextType | null>(null)

export function useVelocityMode(): VelocityContextType {
  const context = useContext(VelocityContext)
  if (!context) {
    return {
      mode: 'standard',
      config: VELOCITY_CONFIGS.standard,
      setMode: () => {},
      isHighVelocity: false,
      isTurbo: false,
    }
  }
  return context
}

export function VelocityProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<VelocityMode>('standard')

  const config = VELOCITY_CONFIGS[mode]
  const isHighVelocity = mode === 'high' || mode === 'turbo'
  const isTurbo = mode === 'turbo'

  return (
    <VelocityContext.Provider value={{ mode, config, setMode, isHighVelocity, isTurbo }}>
      {children}
    </VelocityContext.Provider>
  )
}

export function usePriceChange(currentValue: number, _key?: string) {
  const [prevValue, setPrevValue] = useState<number>(currentValue)
  const [changeDirection, setChangeDirection] = useState<'up' | 'down' | 'none'>('none')
  const [changeAmount, setChangeAmount] = useState<number>(0)
  const [isFlashing, setIsFlashing] = useState(false)

  useEffect(() => {
    if (currentValue !== prevValue) {
      const diff = currentValue - prevValue
      setChangeAmount(diff)
      setChangeDirection(diff > 0 ? 'up' : diff < 0 ? 'down' : 'none')
      setIsFlashing(true)
      setPrevValue(currentValue)

      const timeout = setTimeout(() => {
        setIsFlashing(false)
      }, 500)

      return () => clearTimeout(timeout)
    }
  }, [currentValue, prevValue])

  return { changeDirection, changeAmount, isFlashing }
}

export function useRefreshInterval(
  callback: () => void,
  intervalType: keyof VelocityConfig['refreshIntervals']
) {
  const { config } = useVelocityMode()
  const interval = config.refreshIntervals[intervalType]

  useEffect(() => {
    callback()
    const id = setInterval(callback, interval)
    return () => clearInterval(id)
  }, [callback, interval])
}

export { VelocityContext, VELOCITY_CONFIGS }
