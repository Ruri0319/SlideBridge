import type { ConversionSettings } from "./types";

const SETTINGS_KEY = "slidebridge.conversion.settings";
const LEGACY_SETTINGS_KEY = "ibl2svs.conversion.settings";

export const defaultConversionSettings: ConversionSettings = {
  parallel_wsi: 1,
  main_quality: 90,
  preview_quality: 70,
  pyramid_quality: 60,
  memory_budget_mb: 6144,
};

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === "number" ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

export function normalizeConversionSettings(settings: Partial<ConversionSettings>): ConversionSettings {
  return {
    parallel_wsi: clampNumber(settings.parallel_wsi, defaultConversionSettings.parallel_wsi, 1, 8),
    main_quality: clampNumber(settings.main_quality, defaultConversionSettings.main_quality, 1, 100),
    preview_quality: clampNumber(settings.preview_quality, defaultConversionSettings.preview_quality, 1, 100),
    pyramid_quality: clampNumber(settings.pyramid_quality, defaultConversionSettings.pyramid_quality, 1, 100),
    memory_budget_mb: clampNumber(settings.memory_budget_mb, defaultConversionSettings.memory_budget_mb, 1024, 65536),
  };
}

export function loadConversionSettings(): ConversionSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY) ?? localStorage.getItem(LEGACY_SETTINGS_KEY);
    if (!raw) return defaultConversionSettings;
    return normalizeConversionSettings(JSON.parse(raw));
  } catch {
    return defaultConversionSettings;
  }
}

export function saveConversionSettings(settings: ConversionSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(normalizeConversionSettings(settings)));
}
