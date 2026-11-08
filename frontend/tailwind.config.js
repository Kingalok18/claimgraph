/** Colours are CSS variables from src/styles.css, so the dark theme needs no extra classes. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        subtle: "var(--subtle)",
        ink: "var(--text)",
        muted: "var(--text-2)",
        line: "var(--border)",
        accent: "var(--accent)",
        "accent-ink": "var(--accent-ink)",
        supports: "var(--supports)",
        contradicts: "var(--contradicts)",
        setting: "var(--setting)",
        unclear: "var(--unclear)",
      },
      fontFamily: {
        sans: ["Inter Variable", "system-ui", "sans-serif"],
        serif: ["Source Serif 4 Variable", "Georgia", "serif"],
      },
      fontSize: { base: ["14.5px", "1.5"] },
      borderRadius: { DEFAULT: "6px" },
      transitionDuration: { DEFAULT: "150ms" },
      transitionTimingFunction: { DEFAULT: "ease-out" },
    },
  },
  plugins: [],
};
