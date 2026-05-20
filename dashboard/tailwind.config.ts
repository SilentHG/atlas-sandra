import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}", "./lib/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        atlas: {
          bg: "#070b12",
          panel: "#0d1420",
          panel2: "#111b2a",
          line: "#223049",
          text: "#e6edf7",
          muted: "#93a4ba",
          green: "#2dd4a0",
          amber: "#f5b84b",
          red: "#f87171",
          blue: "#60a5fa"
        }
      }
    }
  },
  plugins: []
};

export default config;
