import '@testing-library/react'
import { expect, afterEach, vi, beforeAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import * as matchers from '@testing-library/jest-dom/matchers'

expect.extend(matchers)

class IntersectionObserverMock implements IntersectionObserver {
  readonly root: Element | null = null
  readonly rootMargin: string = '0px'
  readonly thresholds: ReadonlyArray<number> = [0]
  
  constructor(callback: IntersectionObserverCallback) {
    setTimeout(() => {
      callback([{ isIntersecting: true } as IntersectionObserverEntry], this)
    }, 0)
  }
  disconnect() {}
  observe() {}
  takeRecords(): IntersectionObserverEntry[] { return [] }
  unobserve() {}
}
vi.stubGlobal('IntersectionObserver', IntersectionObserverMock)

const mockFetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  })
)
vi.stubGlobal('fetch', mockFetch)

beforeAll(() => {
  mockFetch.mockClear()
})

afterEach(() => {
  cleanup()
  mockFetch.mockClear()
})
