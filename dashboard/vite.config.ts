import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5000,
    strictPort: true,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/scan_symbol': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/scan_universe': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/trace': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/portfolio': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/monster_runner': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/risk': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/signal': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/oms': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/predictive': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/compliance': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
