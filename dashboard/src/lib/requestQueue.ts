type QueuedRequest = {
  id: string
  execute: () => Promise<unknown>
  resolve: (value: unknown) => void
  reject: (error: unknown) => void
  priority: number
}

class RequestQueue {
  private queue: QueuedRequest[] = []
  private activeRequests = 0
  private maxConcurrent: number
  private requestId = 0
  private isProcessing = false

  constructor(maxConcurrent = 4) {
    this.maxConcurrent = maxConcurrent
  }

  async enqueue<T>(
    execute: () => Promise<T>,
    priority = 0
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = `req-${++this.requestId}`
      this.queue.push({
        id,
        execute,
        resolve: resolve as (value: unknown) => void,
        reject,
        priority
      })
      this.queue.sort((a, b) => b.priority - a.priority)
      this.processQueue()
    })
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessing) return
    this.isProcessing = true

    while (this.queue.length > 0 && this.activeRequests < this.maxConcurrent) {
      const request = this.queue.shift()
      if (!request) continue

      this.activeRequests++
      
      request.execute()
        .then(result => {
          request.resolve(result)
        })
        .catch(error => {
          request.reject(error)
        })
        .finally(() => {
          this.activeRequests--
          if (this.queue.length > 0) {
            this.processQueue()
          }
        })
    }

    this.isProcessing = false
  }

  getStats() {
    return {
      queued: this.queue.length,
      active: this.activeRequests,
      maxConcurrent: this.maxConcurrent
    }
  }

  setMaxConcurrent(max: number) {
    this.maxConcurrent = max
    this.processQueue()
  }
}

export const requestQueue = new RequestQueue(4)

export function throttledFetch<T>(
  fetchFn: () => Promise<T>,
  priority = 0
): Promise<T> {
  return requestQueue.enqueue(fetchFn, priority) as Promise<T>
}
