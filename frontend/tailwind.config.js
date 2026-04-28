/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        app: {
          primary: '#059669',       // Emerald 600
          primaryHover: '#047857',  // Emerald 700
          primaryLight: '#ecfdf5',  // Emerald 50
          secondary: '#64748b',     // Slate 500
          secondaryHover: '#475569',// Slate 600
          secondaryLight: '#f8fafc',// Slate 50
          bg: '#f8fafc',            // Slate 50
          surface: '#ffffff',       // White
          border: '#e5e7eb',        // Gray 200
          text: '#111827',          // Gray 900
          textSecondary: '#6b7280', // Gray 500
          error: '#dc2626',         // Red 600
          errorLight: '#fef2f2',    // Red 50
          success: '#16a34a',       // Green 600
          successLight: '#f0fdf4',  // Green 50
          warning: '#d97706',       // Amber 600
          warningLight: '#fffbeb',  // Amber 50
        }
      }
    },
  },
  plugins: [],
}
