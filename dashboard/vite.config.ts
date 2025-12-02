import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

const backendTarget = 'http://127.0.0.1:8000'

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
      '/api': { target: backendTarget, changeOrigin: true },
      '/health': { target: backendTarget, changeOrigin: true },
      '/scan_symbol': { target: backendTarget, changeOrigin: true },
      '/scan_universe': { target: backendTarget, changeOrigin: true },
      '/scan_universe_mode': { target: backendTarget, changeOrigin: true },
      '/trace': { target: backendTarget, changeOrigin: true },
      '/portfolio': { target: backendTarget, changeOrigin: true },
      '/monster_runner': { target: backendTarget, changeOrigin: true },
      '/risk': { target: backendTarget, changeOrigin: true },
      '/signal': { target: backendTarget, changeOrigin: true },
      '/signals': { target: backendTarget, changeOrigin: true },
      '/oms': { target: backendTarget, changeOrigin: true },
      '/predictive': { target: backendTarget, changeOrigin: true },
      '/compliance': { target: backendTarget, changeOrigin: true },
      '/broker': { target: backendTarget, changeOrigin: true },
      '/data_providers': { target: backendTarget, changeOrigin: true },
      '/trading': { target: backendTarget, changeOrigin: true },
      '/apexlab': { target: backendTarget, changeOrigin: true },
      '/model': { target: backendTarget, changeOrigin: true },
      '/autotrader': { target: backendTarget, changeOrigin: true },
      '/screener': { target: backendTarget, changeOrigin: true },
      '/sms': { target: backendTarget, changeOrigin: true },
      '/push': { target: backendTarget, changeOrigin: true },
      '/backtest': { target: backendTarget, changeOrigin: true },
      '/logs': { target: backendTarget, changeOrigin: true },
      '/drift': { target: backendTarget, changeOrigin: true },
      '/alpha-factory': { target: backendTarget, changeOrigin: true },
      '/battle-simulator': { target: backendTarget, changeOrigin: true },
      '/eeo': { target: backendTarget, changeOrigin: true },
      '/estimated_move': { target: backendTarget, changeOrigin: true },
      '/market': { target: backendTarget, changeOrigin: true },
      '/positions': { target: backendTarget, changeOrigin: true },
      '/hyperspeed': { target: backendTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
  },
})
