import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  root: 'dashboard',
  publicDir: 'public',
  server: {
    host: '0.0.0.0',
    port: 5000,
    strictPort: true,
    allowedHosts: true,
    proxy: {
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
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/desk': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../dist',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'dashboard/src'),
    },
  },
})
