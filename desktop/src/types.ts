export type OutputFormat = "generic_tiff" | "svs";
export type ThemeMode = "auto" | "light" | "dark";
export type ActualTheme = "light" | "dark";
export type ViewKey = "new" | "settings";

export interface ThemeSettings {
  mode: ThemeMode;
}

export interface ConversionSettings {
  parallel_wsi: number;
  jpeg_quality: number;
  memory_budget_mb: number;
}

export interface ConversionRequest {
  job_id: string;
  input_dir: string;
  output_dir: string;
  output_format: OutputFormat;
  recursive: boolean;
  memory_budget_mb: number;
  tile_size: number;
  jpeg_quality: number;
  parallel_wsi: number;
}

export interface ConvertResultPayload {
  input_path: string;
  output_path: string | null;
  success: boolean;
  input_format: string;
  status: string;
  output_format: OutputFormat;
  backend: string;
  width: number | null;
  height: number | null;
  pyramid_levels: number | null;
  mpp: number | null;
  duration_sec: number;
  peak_memory_mb: number;
  failure_stage: string | null;
  error_code: string | null;
  error: string | null;
}

export interface BatchPayload {
  total_files: number;
  success_count: number;
  failed_count: number;
  cancelled_count: number;
  cancelled: boolean;
  report_path: string | null;
  results: ConvertResultPayload[];
}

export type ConversionEvent =
  | { type: "ready"; banner?: string }
  | { type: "started"; job_id: string }
  | { type: "log"; job_id?: string | null; message: string }
  | { type: "report_path"; job_id?: string; path: string }
  | { type: "overall"; job_id?: string; done: number; total: number; current: string }
  | { type: "performance"; job_id?: string; memory_mb: number; cpu_percent: number }
  | {
      type: "file_progress";
      job_id?: string;
      current: string;
      level: string;
      done: number;
      total: number;
      overall_done: number;
      overall_total: number;
    }
  | { type: "done"; job_id: string; batch: BatchPayload }
  | { type: "error"; job_id?: string | null; message: string; traceback?: string }
  | { type: "worker_terminated"; code?: number | null; signal?: number | null };

export interface TaskEvent {
  id: string;
  title: string;
  detail: string;
  status: "idle" | "active" | "success" | "warning" | "error";
}

export interface ProgressState {
  running: boolean;
  statusText: string;
  currentFile: string;
  currentPhase: string;
  stagePercent: number;
  filePercent: number;
  batchPercent: number;
  batchDone: number;
  batchTotal: number;
  etaText: string;
  backend: string;
  memoryMb: number;
  cpuPercent: number;
  reportPath: string;
  outputDir: string;
  startedAt: number | null;
}
