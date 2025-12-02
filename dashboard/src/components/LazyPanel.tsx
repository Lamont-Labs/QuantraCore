import { useState, useEffect, useRef, ReactNode } from 'react'

interface LazyPanelProps {
  children: ReactNode
  fallback?: ReactNode
  delay?: number
  rootMargin?: string
}

export function LazyPanel({ 
  children, 
  fallback = null, 
  delay = 0,
  rootMargin = '100px'
}: LazyPanelProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [shouldRender, setShouldRender] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true)
        }
      },
      { rootMargin }
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [isVisible, rootMargin])

  useEffect(() => {
    if (isVisible) {
      if (delay > 0) {
        const timer = setTimeout(() => setShouldRender(true), delay)
        return () => clearTimeout(timer)
      } else {
        setShouldRender(true)
      }
    }
  }, [isVisible, delay])

  return (
    <div ref={ref}>
      {shouldRender ? children : fallback}
    </div>
  )
}

export function PanelSkeleton({ height = 'h-48' }: { height?: string }) {
  return (
    <div className={`${height} bg-slate-900/50 rounded-xl border border-slate-700/30 animate-pulse flex items-center justify-center`}>
      <div className="text-slate-600 text-xs">Loading...</div>
    </div>
  )
}
