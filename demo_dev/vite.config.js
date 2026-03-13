// Copyright (c) Microsoft Corporation.
//  Licensed under the MIT license.

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { ReactInjectorVitePlugin } from 'yunji-tagger'

export default defineConfig({
  plugins: [react(), ReactInjectorVitePlugin()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts: [
      'ununified-outrightly-nickole.ngrok-free.dev',
    ]
  }
})