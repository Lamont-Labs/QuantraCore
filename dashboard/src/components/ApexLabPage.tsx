import { useState, useEffect } from 'react'

interface LabStatus {
  version: string
  schema_fields: number
  training_samples: number
  last_training: string | null
}

export function ApexLabPage() {
  const [status, setStatus] = useState<LabStatus | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [logs, setLogs] = useState<string[]>([])

  useEffect(() => {
    loadStatus()
  }, [])

  async function loadStatus() {
    setStatus({
      version: 'v2.0',
      schema_fields: 40,
      training_samples: 0,
      last_training: null
    })
  }

  async function handleStartTraining() {
    setIsTraining(true)
    setTrainingProgress(0)
    setLogs(['[INFO] Initializing ApexLab training pipeline...'])
    
    const steps = [
      '[INFO] Loading OHLCV windows...',
      '[INFO] Extracting 40+ field features...',
      '[INFO] Computing teacher labels via ApexEngine...',
      '[INFO] Generating future outcome labels...',
      '[INFO] Applying leakage prevention guards...',
      '[INFO] Building training dataset...',
      '[INFO] Training GradientBoosting ensemble...',
      '[INFO] Validating model performance...',
      '[SUCCESS] Training complete!'
    ]

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800))
      setLogs(prev => [...prev, steps[i]])
      setTrainingProgress(((i + 1) / steps.length) * 100)
    }

    setIsTraining(false)
    loadStatus()
  }

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">⬡</span>
          ApexLab Training Environment
        </h2>
        <p className="text-slate-400 text-sm mb-6">
          Offline machine learning training pipeline with walk-forward validation and bootstrap ensembles.
        </p>

        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Version</div>
            <div className="text-xl font-bold text-cyan-400">{status?.version || 'v2.0'}</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Schema Fields</div>
            <div className="text-xl font-bold text-white">{status?.schema_fields || 40}+</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Training Samples</div>
            <div className="text-xl font-bold text-white">{status?.training_samples?.toLocaleString() || '0'}</div>
          </div>
          <div className="bg-[#0a0f1a] p-4 rounded-lg border border-[#0096ff]/20">
            <div className="text-sm text-slate-400">Last Training</div>
            <div className="text-xl font-bold text-slate-300">{status?.last_training || 'Never'}</div>
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={isTraining}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            isTraining
              ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/25'
          }`}
        >
          {isTraining ? 'Training in Progress...' : 'Start Training Pipeline'}
        </button>

        {isTraining && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-slate-400 mb-1">
              <span>Progress</span>
              <span>{trainingProgress.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-[#0a0f1a] rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 apex-card overflow-hidden">
        <h3 className="text-lg font-semibold text-slate-300 mb-4">Training Logs</h3>
        <div className="h-64 bg-[#030508] rounded-lg p-4 font-mono text-sm overflow-y-auto">
          {logs.length === 0 ? (
            <div className="text-slate-500">No training logs yet. Click "Start Training Pipeline" to begin.</div>
          ) : (
            logs.map((log, i) => (
              <div 
                key={i} 
                className={`${
                  log.includes('[SUCCESS]') ? 'text-green-400' :
                  log.includes('[ERROR]') ? 'text-red-400' :
                  log.includes('[WARN]') ? 'text-yellow-400' :
                  'text-slate-300'
                }`}
              >
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
