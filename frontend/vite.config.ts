import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";
const proxied = ["/health", "/library", "/papers", "/analyze", "/jobs", "/topics", "/pairs", "/labels", "/search"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(proxied.map((path) => [path, api])),
  },
  build: { chunkSizeWarningLimit: 2000 },
});
