/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
      },
      colors: {
        panel: {
          950: '#0a0f1e',
          900: '#0d1424',
          800: '#131c30',
          700: '#1a2540',
          600: '#243050',
        },
      },
      keyframes: {
        typing: {
          '0%, 60%, 100%': { transform: 'translateY(0)' },
          '30%': { transform: 'translateY(-6px)' },
        },
      },
      animation: {
        typing: 'typing 1.2s infinite',
      },
    },
  },
  plugins: [],
}
