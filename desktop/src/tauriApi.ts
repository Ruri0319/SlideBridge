import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import type { ConversionEvent, ConversionRequest } from "./types";

function isTauri(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

export async function chooseDirectory(): Promise<string | null> {
  if (!isTauri()) return null;
  const selected = await openDialog({ directory: true, multiple: false });
  return typeof selected === "string" ? selected : null;
}

export async function openFilesystemPath(path: string): Promise<void> {
  if (!path || !isTauri()) return;
  await openPath(path);
}

export async function getApplicationVersion(): Promise<string> {
  if (!isTauri()) return "0.4.5";
  return getVersion();
}

export async function startConversion(request: ConversionRequest): Promise<void> {
  if (!isTauri()) {
    throw new Error("Tauri runtime is not available. Use npm run tauri dev for conversions.");
  }
  await invoke("start_conversion", { request });
}

export async function cancelConversion(): Promise<void> {
  if (!isTauri()) return;
  await invoke("cancel_conversion");
}

export async function onConversionEvent(callback: (event: ConversionEvent) => void): Promise<() => void> {
  if (!isTauri()) return () => undefined;
  return listen<ConversionEvent>("conversion:event", (event) => callback(event.payload));
}
