import { defineConfig, loadEnv } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsConfigPaths from "vite-tsconfig-paths";
import { nitro } from "nitro/vite";

export default defineConfig(({ mode }) => {
  // Expose .env / .env.local values (e.g. GEMINI_API_KEY) to server code via process.env.
  for (const [key, value] of Object.entries(loadEnv(mode, process.cwd(), ""))) {
    process.env[key] ??= value;
  }
  return {
    server: { port: 8080 },
    plugins: [
      tsConfigPaths(),
      tailwindcss(),
      // Server entry is src/server.ts (our SSR error wrapper).
      tanstackStart({ server: { entry: "server" } }),
      nitro(),
      viteReact(),
    ],
  };
});
