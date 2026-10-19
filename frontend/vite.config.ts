import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve, extname } from 'path'
import fs from 'fs'

function ortWasmPlugin(): Plugin {
  return {
    name: 'ort-wasm-handler',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/ort-wasm/')) {
          const fileName = req.url.split('?')[0].replace('/ort-wasm/', '');
          const filePath = resolve(__dirname, 'node_modules/onnxruntime-web/dist', fileName);
          if (fs.existsSync(filePath)) {
            const ext = extname(filePath);
            const mimeType = ext === '.wasm' ? 'application/wasm' : 'text/javascript';
            res.setHeader('Content-Type', mimeType);
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            res.setHeader('Cache-Control', 'no-cache');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
        }
        next();
      });
    },
    closeBundle() {
      // Copy wasm & mjs to dist/ort-wasm for production build
      const distOrtDir = resolve(__dirname, 'dist/ort-wasm');
      const srcOrtDir = resolve(__dirname, 'node_modules/onnxruntime-web/dist');
      if (!fs.existsSync(distOrtDir)) {
        fs.mkdirSync(distOrtDir, { recursive: true });
      }
      const files = fs.readdirSync(srcOrtDir);
      for (const file of files) {
        if (file.endsWith('.wasm') || file.endsWith('.mjs')) {
          fs.copyFileSync(resolve(srcOrtDir, file), resolve(distOrtDir, file));
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [vue(), ortWasmPlugin()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
