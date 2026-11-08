// Light, dark, or follow the system. The choice is remembered in this browser.

export type ThemeChoice = "light" | "dark" | "system";
const KEY = "claimgraph-theme";
const DEFAULT: ThemeChoice = "light"; // the app's default, from the setup answers

function read(): ThemeChoice {
  try {
    const value = localStorage.getItem(KEY);
    if (value === "light" || value === "dark" || value === "system") return value;
  } catch {
    /* storage can be blocked; the default still works */
  }
  return DEFAULT;
}

function apply(choice: ThemeChoice) {
  const dark = choice === "dark" || (choice === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  window.dispatchEvent(new Event("claimgraph-theme"));
}

export function getTheme(): ThemeChoice {
  return read();
}

export function setTheme(choice: ThemeChoice) {
  try {
    localStorage.setItem(KEY, choice);
  } catch {
    /* applies for this visit only */
  }
  apply(choice);
}

export function initTheme() {
  apply(read());
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (read() === "system") apply("system");
  });
}
