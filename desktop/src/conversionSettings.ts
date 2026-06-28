import type { ConversionSettings } from "./types";

const SETTINGS_KEY = "ibl2svs.conversion.settings";

export const defaultConversionSettings: ConversionSettings = {
  parallel_wsi: 1,
  jpeg_quality: 90,
  memory_budget_mb: 6144,
};

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

export function normalizeConversionSettings(settings: Partial<ConversionSettings>): ConversionSettings {
  return {
    parallel_wsi: clampNumber(settings.parallel_wsi, defaultConversionSettings.parallel_wsi, 1, 4),
    jpeg_quality: clampNumber(settings.jpeg_quality, defaultConversionSettings.jpeg_quality, 1, 100),
    memory_budget_mb: clampNumber(settings.memory_budget_mb, defaultConversionSettings.memory_budget_mb, 1024, 65536),
  };
}

export function loadConversionSettings(): ConversionSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return defaultConversionSettings;
    return normalizeConversionSettings(JSON.parse(raw));
  } catch {
    return defaultConversionSettings;
  }
}

export function saveConversionSettings(settings: ConversionSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(normalizeConversionSettings(settings)));
}
