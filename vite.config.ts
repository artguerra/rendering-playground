import path from "path";

export default {
  resolve: {
    alias: {
      "@shaders": path.resolve(__dirname, "./shaders/"),
      "@assets": path.resolve(__dirname, "./assets/"),
    }
  },
};
