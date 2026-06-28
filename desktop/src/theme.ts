import type { ActualTheme, ThemeMode, ThemeSettings } from "./types";

const SETTINGS_KEY = "ibl2svs.theme.settings";

export const defaultThemeSettings: ThemeSettings = {
  mode: "auto",
};

export function loadThemeSettings(): ThemeSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return defaultThemeSettings;
    return { ...defaultThemeSettings, ...JSON.parse(raw) };
  } catch {
    return defaultThemeSettings;
  }
}

export function saveThemeSettings(settings: ThemeSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export function resolveTheme(settings: ThemeSettings, now = new Date()): { theme: ActualTheme; hint: string } {
  const mode: ThemeMode = settings.mode;
  if (mode === "light" || mode === "dark") {
    return { theme: mode, hint: mode === "light" ? "手动浅色模式" : "手动深色模式" };
  }

  const current = now.getHours() + now.getMinutes() / 60;
  return {
    theme: current >= 7 && current < 20 ? "light" : "dark",
    hint: "自动模式根据系统时间切换：07:00 后浅色，20:00 后深色",
  };
}
