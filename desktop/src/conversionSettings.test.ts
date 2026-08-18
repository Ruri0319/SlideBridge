import { beforeEach, describe, expect, it } from "vitest";
import {
  defaultConversionSettings,
  loadConversionSettings,
  normalizeConversionSettings,
} from "./conversionSettings";

describe("conversion settings", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("uses separate quality defaults for main, preview, and pyramid images", () => {
    expect(defaultConversionSettings).toMatchObject({
      main_quality: 90,
      preview_quality: 70,
      pyramid_quality: 60,
    });
    expect(loadConversionSettings()).toEqual(defaultConversionSettings);
  });

  it("normalizes an older saved setting to the new quality fields", () => {
    localStorage.setItem(
      "slidebridge.conversion.settings",
      JSON.stringify({ parallel_wsi: 2, jpeg_quality: 95, memory_budget_mb: 4096 }),
    );

    expect(loadConversionSettings()).toEqual({
      parallel_wsi: 2,
      main_quality: 90,
      preview_quality: 70,
      pyramid_quality: 60,
      memory_budget_mb: 4096,
    });
    expect(normalizeConversionSettings({ main_quality: 101, preview_quality: 0, pyramid_quality: 55 })).toMatchObject({
      main_quality: 100,
      preview_quality: 1,
      pyramid_quality: 55,
    });
  });
});
