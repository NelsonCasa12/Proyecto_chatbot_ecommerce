import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy to avoid CORS issues with FastAPI running on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/search': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000'
    }
  }
})
