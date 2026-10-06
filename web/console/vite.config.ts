import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: {
    // The console serves dist while building. Keep hashed chunks used by open
    // tabs until a maintenance cleanup instead of breaking lazy navigation.
    emptyOutDir: false,
    chunkSizeWarningLimit: 1100,
    rollupOptions: {
      output: { manualChunks: { charts: ['echarts'], ui: ['antd', '@ant-design/icons'] } },
    },
  },
});
