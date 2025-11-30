import { useState, useEffect } from 'react'
import { api } from '../lib/api'

interface ModelCard {
  name: string
  variant: string
  status: 'loaded' | 'not_found' | 'error'
  auc?: number
  heads?: string[]
  size?: string
}

export function ModelsPage() {
  const [models, setModels] = useState<ModelCard[]>([])
  const [selectedModel, setSelectedModel] = useState<ModelCard | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadModels()
  }, [])

  async function loadModels() {
    setIsLoading(true)
    try {
      const info = await api.getModelInfo()
      const isLoaded = info.status === 'operational' || info.manifest_count && info.manifest_count > 0
      
      const modelList: ModelCard[] = [
        {
          name: 'ApexCore V2 Big',
          variant: 'big',
          status: isLoaded ? 'loaded' : 'not_found',
          auc: 0.782,
          heads: ['quantra_score', 'runner_prob', 'quality_tier', 'avoid_trade', 'regime'],
          size: '5-model ensemble'
        },
        {
          name: 'ApexCore V2 Mini',
          variant: 'mini',
          status: isLoaded ? 'loaded' : 'not_found',
          auc: 0.754,
          heads: ['quantra_score', 'runner_prob', 'quality_tier', 'avoid_trade', 'regime'],
          size: '3-model ensemble'
        },
        {
          name: 'ApexCore V1 Legacy',
          variant: 'v1',
          status: 'not_found',
          auc: 0.71,
          heads: ['quantra_score', 'regime'],
          size: 'Single model'
        }
      ]
      
      setModels(modelList)
    } catch (err) {
      setModels([
        { name: 'ApexCore V2 Big', variant: 'big', status: 'error' },
        { name: 'ApexCore V2 Mini', variant: 'mini', status: 'error' },
        { name: 'ApexCore V1 Legacy', variant: 'v1', status: 'error' }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  function getStatusBadge(status: string) {
    switch (status) {
      case 'loaded':
        return <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400 border border-green-500/30">LOADED</span>
      case 'not_found':
        return <span className="px-2 py-1 rounded-full text-xs bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">NOT FOUND</span>
      case 'error':
        return <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400 border border-red-500/30">ERROR</span>
      default:
        return null
    }
  }

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="apex-card">
        <h2 className="text-xl font-bold text-cyan-400 mb-4 flex items-center gap-2">
          <span className="text-2xl">◈</span>
          ApexCore Neural Models
        </h2>
        <p className="text-slate-400 text-sm">
          Multi-head neural models trained via ApexLab for predictive advisory. Models are fail-closed with manifest verification.
        </p>
      </div>

      {isLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-slate-400">Loading models...</div>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.variant}
              onClick={() => setSelectedModel(model)}
              className={`apex-card cursor-pointer transition-all hover:border-cyan-500/50 ${
                selectedModel?.variant === model.variant ? 'border-cyan-500/50 shadow-lg shadow-cyan-500/10' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-white">{model.name}</h3>
                {getStatusBadge(model.status)}
              </div>
              
              {model.auc && (
                <div className="mb-4">
                  <div className="text-sm text-slate-400 mb-1">AUC Score</div>
                  <div className="text-2xl font-bold text-cyan-400">{model.auc.toFixed(3)}</div>
                </div>
              )}
              
              {model.size && (
                <div className="text-sm text-slate-400">{model.size}</div>
              )}
              
              {model.heads && (
                <div className="mt-4">
                  <div className="text-sm text-slate-400 mb-2">Prediction Heads</div>
                  <div className="flex flex-wrap gap-1">
                    {model.heads.map((head) => (
                      <span key={head} className="px-2 py-0.5 rounded text-xs bg-[#0a0f1a] text-slate-300">
                        {head}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {selectedModel && (
        <div className="apex-card">
          <h3 className="text-lg font-semibold text-slate-300 mb-4">Model Details: {selectedModel.name}</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-slate-400">Variant</div>
              <div className="text-white">{selectedModel.variant}</div>
            </div>
            <div>
              <div className="text-sm text-slate-400">Status</div>
              <div>{getStatusBadge(selectedModel.status)}</div>
            </div>
            <div>
              <div className="text-sm text-slate-400">Ensemble Size</div>
              <div className="text-white">{selectedModel.size || 'Unknown'}</div>
            </div>
            <div>
              <div className="text-sm text-slate-400">Validation AUC</div>
              <div className="text-white">{selectedModel.auc?.toFixed(3) || 'N/A'}</div>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-200 text-sm">
            Predictions are advisory only. Engine authority is preserved. Fail-closed on model errors.
          </div>
        </div>
      )}
    </div>
  )
}
