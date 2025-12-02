import { useState, useEffect, useCallback } from 'react'

interface PushStatus {
  enabled: boolean
  subscribers: number
  alerts_this_hour: number
  max_alerts_per_hour: number
  thresholds: {
    min_quantrascore: number
    min_runner_probability: number
    max_avoid_probability: number
  }
  recent_alerts: number
}

interface PushConfig {
  enabled: boolean
  min_quantrascore: number
  min_runner_probability: number
  max_avoid_probability: number
  min_timing_confidence: number
  only_immediate_timing: boolean
  max_alerts_per_hour: number
  cooldown_minutes: number
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - base64String.length % 4) % 4)
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/')

  const rawData = window.atob(base64)
  const outputArray = new Uint8Array(rawData.length)

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i)
  }
  return outputArray
}

export function PushNotificationPanel() {
  const [status, setStatus] = useState<PushStatus | null>(null)
  const [config, setConfig] = useState<PushConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSubscribed, setIsSubscribed] = useState(false)
  const [isSubscribing, setIsSubscribing] = useState(false)
  const [isTesting, setIsTesting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)

  const checkSubscription = useCallback(async () => {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      return false
    }

    try {
      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.getSubscription()
      return subscription !== null
    } catch {
      return false
    }
  }, [])

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/push/status')
      if (res.ok) {
        const data = await res.json()
        setStatus(data.status)
        setConfig(data.config)
      }
      
      const subscribed = await checkSubscription()
      setIsSubscribed(subscribed)
    } catch (err) {
      console.error('Failed to fetch push status:', err)
    } finally {
      setIsLoading(false)
    }
  }, [checkSubscription])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then(() => console.log('Service Worker registered'))
        .catch(err => console.error('SW registration failed:', err))
    }
  }, [])

  const subscribe = async () => {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      setError('Push notifications are not supported in this browser')
      return
    }

    setIsSubscribing(true)
    setError(null)
    setSuccessMessage(null)

    try {
      const permission = await Notification.requestPermission()
      if (permission !== 'granted') {
        setError('Notification permission denied. Please enable notifications in your browser settings.')
        setIsSubscribing(false)
        return
      }

      const vapidRes = await fetch('/push/vapid-key')
      if (!vapidRes.ok) {
        throw new Error('Failed to get VAPID key')
      }
      const { public_key } = await vapidRes.json()

      const registration = await navigator.serviceWorker.ready
      
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(public_key) as BufferSource
      })

      const subRes = await fetch('/push/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(subscription.toJSON())
      })

      if (!subRes.ok) {
        throw new Error('Failed to register subscription with server')
      }

      setIsSubscribed(true)
      setSuccessMessage('Successfully subscribed to push notifications!')
      fetchStatus()
    } catch (err) {
      console.error('Subscription error:', err)
      setError(err instanceof Error ? err.message : 'Failed to subscribe')
    } finally {
      setIsSubscribing(false)
    }
  }

  const unsubscribe = async () => {
    setIsSubscribing(true)
    setError(null)

    try {
      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.getSubscription()
      
      if (subscription) {
        await subscription.unsubscribe()
        
        await fetch('/push/unsubscribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ endpoint: subscription.endpoint })
        })
      }

      setIsSubscribed(false)
      setSuccessMessage('Successfully unsubscribed from notifications')
      fetchStatus()
    } catch (err) {
      console.error('Unsubscribe error:', err)
      setError('Failed to unsubscribe')
    } finally {
      setIsSubscribing(false)
    }
  }

  const sendTest = async () => {
    setIsTesting(true)
    setError(null)
    setSuccessMessage(null)

    try {
      const res = await fetch('/push/test', { method: 'POST' })
      const data = await res.json()

      if (data.success) {
        setSuccessMessage(`Test notification sent to ${data.sent} subscriber(s)`)
      } else {
        setError(data.error || 'Failed to send test notification')
      }
    } catch (err) {
      setError('Failed to send test notification')
    } finally {
      setIsTesting(false)
    }
  }

  const updateConfig = async (updates: Partial<PushConfig>) => {
    try {
      const params = new URLSearchParams()
      Object.entries(updates).forEach(([key, value]) => {
        if (value !== undefined) {
          params.append(key, String(value))
        }
      })

      const res = await fetch(`/push/config?${params.toString()}`, { method: 'POST' })
      const data = await res.json()

      if (data.success) {
        setConfig(data.config)
        setSuccessMessage('Settings updated')
        setTimeout(() => setSuccessMessage(null), 2000)
      }
    } catch (err) {
      setError('Failed to update settings')
    }
  }

  const notificationsSupported = 'serviceWorker' in navigator && 'PushManager' in window

  return (
    <div className="bg-gradient-to-br from-[#0a1628] to-[#050a14] border border-[#0096ff]/30 rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-medium">Push Notifications</h3>
            <span className="text-xs text-gray-400">Direct alerts - no SMS costs</span>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {isSubscribed && (
            <span className="px-2 py-1 text-xs font-medium bg-green-500/20 text-green-400 rounded-full">
              Active
            </span>
          )}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>

      {!notificationsSupported ? (
        <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
          <p className="text-yellow-400 text-sm">
            Push notifications are not supported in this browser.
            Try using Chrome, Firefox, or Edge.
          </p>
        </div>
      ) : isLoading ? (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-[#0096ff]"></div>
        </div>
      ) : (
        <div className="space-y-4">
          {error && (
            <div className="p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
              {error}
            </div>
          )}
          
          {successMessage && (
            <div className="p-2 bg-green-500/10 border border-green-500/30 rounded text-green-400 text-sm">
              {successMessage}
            </div>
          )}

          {!isSubscribed ? (
            <div className="space-y-3">
              <p className="text-gray-400 text-sm">
                Get instant notifications when high-quality trading signals are detected.
                Works even when your browser is closed.
              </p>
              
              <button
                onClick={subscribe}
                disabled={isSubscribing}
                className="w-full px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-medium rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSubscribing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Enabling...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                    </svg>
                    Enable Notifications
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-[#0a1628] rounded-lg p-2 text-center">
                  <div className="text-lg font-bold text-[#0096ff]">{status?.subscribers || 0}</div>
                  <div className="text-xs text-gray-500">Devices</div>
                </div>
                <div className="bg-[#0a1628] rounded-lg p-2 text-center">
                  <div className="text-lg font-bold text-green-400">{status?.alerts_this_hour || 0}</div>
                  <div className="text-xs text-gray-500">This Hour</div>
                </div>
                <div className="bg-[#0a1628] rounded-lg p-2 text-center">
                  <div className="text-lg font-bold text-yellow-400">{status?.recent_alerts || 0}</div>
                  <div className="text-xs text-gray-500">Recent</div>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={sendTest}
                  disabled={isTesting}
                  className="flex-1 px-3 py-2 bg-[#0a1628] hover:bg-[#0f1f38] text-[#0096ff] font-medium rounded-lg transition-colors text-sm disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isTesting ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-[#0096ff]"></div>
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                  Test
                </button>
                <button
                  onClick={unsubscribe}
                  disabled={isSubscribing}
                  className="px-3 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 font-medium rounded-lg transition-colors text-sm disabled:opacity-50"
                >
                  Disable
                </button>
              </div>
            </div>
          )}

          {showSettings && config && (
            <div className="mt-4 pt-4 border-t border-[#0096ff]/20 space-y-3">
              <h4 className="text-white text-sm font-medium">Alert Thresholds</h4>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-gray-400 text-xs">Min QuantraScore</label>
                  <select
                    value={config.min_quantrascore}
                    onChange={(e) => updateConfig({ min_quantrascore: parseFloat(e.target.value) })}
                    className="bg-[#0a1628] text-white text-xs rounded px-2 py-1 border border-[#0096ff]/30"
                  >
                    <option value="0.50">50%</option>
                    <option value="0.60">60%</option>
                    <option value="0.65">65%</option>
                    <option value="0.70">70%</option>
                    <option value="0.75">75%</option>
                    <option value="0.80">80%</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-gray-400 text-xs">Min Runner Prob</label>
                  <select
                    value={config.min_runner_probability}
                    onChange={(e) => updateConfig({ min_runner_probability: parseFloat(e.target.value) })}
                    className="bg-[#0a1628] text-white text-xs rounded px-2 py-1 border border-[#0096ff]/30"
                  >
                    <option value="0.40">40%</option>
                    <option value="0.50">50%</option>
                    <option value="0.60">60%</option>
                    <option value="0.70">70%</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-gray-400 text-xs">Max Alerts/Hour</label>
                  <select
                    value={config.max_alerts_per_hour}
                    onChange={(e) => updateConfig({ max_alerts_per_hour: parseInt(e.target.value) })}
                    className="bg-[#0a1628] text-white text-xs rounded px-2 py-1 border border-[#0096ff]/30"
                  >
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="15">15</option>
                    <option value="20">20</option>
                    <option value="30">30</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-gray-400 text-xs">Only Immediate Timing</label>
                  <button
                    onClick={() => updateConfig({ only_immediate_timing: !config.only_immediate_timing })}
                    className={`w-10 h-5 rounded-full transition-colors ${
                      config.only_immediate_timing ? 'bg-[#0096ff]' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`w-4 h-4 rounded-full bg-white transform transition-transform ${
                      config.only_immediate_timing ? 'translate-x-5' : 'translate-x-0.5'
                    }`} />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
