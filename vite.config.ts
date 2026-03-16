import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  base: "path-tracer",
  resolve: {
    alias: {
      "@shaders": path.resolve(__dirname, "./shaders/"),
      "@assets": path.resolve(__dirname, "./assets/"),
    }
  },
  // build: {
  //   minify: false
  // }
});
