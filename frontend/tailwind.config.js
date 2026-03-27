/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        hollow: {
          bg: '#0a0a0f',
          panel: '#13131a',
          border: '#2a2a3a',
          accent: '#6c8cbf',
          text: '#c8d0e0',
          muted: '#6a6a8a',
        },
      },
    },
  },
  plugins: [],
}
