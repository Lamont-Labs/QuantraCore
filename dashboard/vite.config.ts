import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@assets': path.resolve(__dirname, '../attached_assets'),
    },
  },
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
      '/scan_universe_mode': {
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
      '/signals': {
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
      '/broker': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/data_providers': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/trading': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/apexlab': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/model': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/autotrader': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/screener': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/sms': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/backtest': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/logs': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/drift': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/alpha-factory': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/battle-simulator': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/eeo': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/estimated_move': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/market': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/positions': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
