export type OutputFormat = "ome_tiff" | "svs" | "fluorescence_svs" | "afi";
export type ThemeMode = "auto" | "light" | "dark";
export type ActualTheme = "light" | "dark";
export type ViewKey = "new" | "settings";
export type TaskStatus = "idle" | "inspecting" | "ready" | "starting" | "running" | "success" | "error" | "cancelled";

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
  selected_input_paths: string[] | null;
  convert_compatible_only: boolean;
  channel_overrides: Record<string, ChannelDefinition[]>;
  input_signatures: Record<string, { size: number; mtime_ns: string }>;
  preflight_files: InputInspection[];
}

export interface WorkerStatus {
  alive: boolean;
  ready: boolean;
  activity: "initializing" | "idle" | "inspecting" | "converting" | "unavailable";
  job_id: string | null;
}

export interface InspectionRequest {
  job_id: string;
  input_dir: string;
  recursive: boolean;
}

export interface ChannelDefinition {
  index: number;
  name: string;
  fluor: string | null;
  color: [number, number, number];
  excitation_nm: number | null;
  emission_nm: number | null;
  exposure: number | null;
  identity_source: "source_metadata" | "documented_vendor_id" | "user_supplied" | "unknown";
}

export interface InputInspection {
  input_path: string;
  file_size: number;
  file_mtime_ns: string;
  input_format: string;
  source_modality: "brightfield" | "fluorescence" | "unknown";
  source_container: string | null;
  source_version: string | null;
  source_codec: string | null;
  source_bit_depth: number;
  field_count: number;
  channel_count: number;
  z_count: number;
  t_count: number;
  channel_definitions: ChannelDefinition[];
  allowed_output_formats: OutputFormat[];
  incompatible_reasons: Partial<Record<OutputFormat, string>>;
  error: string | null;
}

export interface BatchInspection {
  input_dir: string;
  recursive: boolean;
  files: InputInspection[];
}

export interface ConvertResultPayload {
  input_path: string;
  output_path: string | null;
  output_files: string[];
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
  native_path: boolean;
  native_level_dimensions: [number, number][] | null;
  native_resource_dimensions: Record<string, [number, number] | null> | null;
  native_tile_mode: string | null;
  native_fallback_reason: string | null;
  source_container: string | null;
  source_version: string | null;
  source_codec: string | null;
  source_bit_depth: number | null;
  source_channel_count: number | null;
  source_axes: string | null;
  compatibility_level: string | null;
  diagnostic_code: string | null;
  diagnostic_stage: string | null;
  svs_omitted_native_data: string | null;
  source_modality: string | null;
  channel_definitions: ChannelDefinition[] | null;
  channel_identity_source: string[] | null;
  channel_override_applied: boolean;
  skipped_reason: string | null;
  failure_stage: string | null;
  error_code: string | null;
  error: string | null;
}

export interface BatchPayload {
  total_files: number;
  success_count: number;
  failed_count: number;
  skipped_count: number;
  cancelled_count: number;
  cancelled: boolean;
  report_path: string | null;
  results: ConvertResultPayload[];
}

export type ConversionEvent =
  | { type: "ready"; banner?: string }
  | { type: "started"; job_id: string }
  | { type: "inspection_started"; job_id: string }
  | { type: "inspection_discovered"; job_id: string; total: number; format_counts: Record<string, number> }
  | { type: "inspection_file_done"; job_id: string; done: number; total: number; current: string; file: InputInspection }
  | { type: "inspection_progress"; job_id: string; done: number; total: number; current: string }
  | { type: "inspection_done"; job_id: string; inspection: BatchInspection; duration_ms?: number }
  | { type: "inspection_error"; job_id?: string | null; message: string; traceback?: string }
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
  | { type: "error"; job_id?: string | null; message: string; traceback?: string; diagnostic_code?: string | null; diagnostic_stage?: string | null }
  | {
      type: "worker_terminated";
      code?: number | null;
      signal?: number | null;
      busy?: boolean;
      activity?: WorkerStatus["activity"];
      job_id?: string | null;
    };

export interface ProgressState {
  running: boolean;
  status: TaskStatus;
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
  finishedAt: number | null;
  successCount: number;
  failedCount: number;
  skippedCount: number;
}
