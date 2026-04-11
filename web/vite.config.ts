import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "../docs/play",
    emptyOutDir: true,
    target: "es2022",
  },
  publicDir: "public",
  test: {
    globals: true,
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
