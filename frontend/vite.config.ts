import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const proxy = { '/api': process.env.VITE_API_TARGET || 'http://127.0.0.1:8000' }

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5175,
    strictPort: true,
    proxy,
  },
  preview: {
    host: '127.0.0.1',
    port: 5175,
    strictPort: true,
    proxy,
  },
})
