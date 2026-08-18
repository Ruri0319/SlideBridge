import * as Select from "@radix-ui/react-select";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Folder,
  LoaderCircle,
  Minus,
  Moon,
  Play,
  Plus,
  RotateCcw,
  Settings,
  Square,
  Sun,
} from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  cancelConversion,
  chooseDirectory,
  ensureWorker,
  getApplicationVersion,
  onConversionEvent,
  openFilesystemPath,
  startConversion,
  startInspection,
} from "./tauriApi";
import {
  defaultConversionSettings,
  loadConversionSettings,
  normalizeConversionSettings,
  saveConversionSettings,
} from "./conversionSettings";
import { defaultThemeSettings, loadThemeSettings, resolveTheme, saveThemeSettings } from "./theme";
import type {
  ActualTheme,
  BatchInspection,
  ChannelDefinition,
  ConversionSettings,
  ConversionEvent,
  InputInspection,
  OutputFormat,
  ProgressState,
  TaskStatus,
  ThemeSettings,
  ViewKey,
} from "./types";

const phases = ["待开始", "解析输入", "构建主图", "生成金字塔", "生成缩略图", "生成附属图像", "写出文件", "完成"];
type PhaseUiState = "done" | "active" | "idle";
type PhaseDisplayState = {
  phase: string;
  status: PhaseUiState;
  detail: string;
};

const initialProgress: ProgressState = {
  running: false,
  status: "idle",
  currentFile: "未选择任务",
  currentPhase: "待开始",
  stagePercent: 0,
  filePercent: 0,
  batchPercent: 0,
  batchDone: 0,
  batchTotal: 0,
  etaText: "—",
  backend: "—",
  memoryMb: 0,
  cpuPercent: 0,
  reportPath: "",
  outputDir: "",
  startedAt: null,
  finishedAt: null,
  successCount: 0,
  failedCount: 0,
  skippedCount: 0,
};

const taskStatusLabels: Record<TaskStatus, string> = {
  idle: "待机",
  inspecting: "预检中",
  ready: "就绪",
  starting: "启动中",
  running: "转换中",
  success: "已完成",
  error: "失败",
  cancelled: "已取消",
};

function basename(path: string): string {
  return path.replace(/\\/g, "/").split("/").filter(Boolean).pop() || path || "未选择任务";
}

function formatPercent(done: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, (done / total) * 100));
}

function formatEta(startedAt: number | null, percent: number): string {
  if (!startedAt || percent <= 0) return "计算中";
  if (percent >= 100) return "已完成";
  const elapsedSeconds = Math.max(0, (Date.now() - startedAt) / 1000);
  if (elapsedSeconds < 1) return "计算中";
  const remainingSeconds = Math.max(0, Math.round((elapsedSeconds * (100 - percent)) / percent));
  if (remainingSeconds < 60) return `${remainingSeconds} 秒`;
  if (remainingSeconds < 3600) return `${Math.floor(remainingSeconds / 60)} 分 ${remainingSeconds % 60} 秒`;
  const hours = Math.floor(remainingSeconds / 3600);
  const minutes = Math.floor((remainingSeconds % 3600) / 60);
  return `${hours} 小时 ${minutes} 分`;
}

function formatDuration(startedAt: number | null, finishedAt: number | null): string {
  if (!startedAt) return "—";
  const seconds = Math.max(0, Math.round(((finishedAt ?? Date.now()) - startedAt) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
}

function normalizePhase(value: string): string {
  if (value.includes("解析")) return "解析输入";
  if (value.includes("主图")) return "构建主图";
  if (value.includes("金字塔")) return "生成金字塔";
  if (value.includes("缩略图")) return "生成缩略图";
  if (value.includes("附属") || value.toLowerCase().includes("label") || value.toLowerCase().includes("macro")) return "生成附属图像";
  if (value.includes("写") || value.includes("导出") || value.includes("重排")) return "写出文件";
  if (value.includes("取消") || value.includes("完成") || value.includes("异常")) return "完成";
  return phases.includes(value) ? value : "待开始";
}

function phaseToIndex(value: string): number {
  const index = phases.indexOf(normalizePhase(value));
  return index >= 0 ? index : 0;
}

function initialPhaseStates(): PhaseDisplayState[] {
  return phases.map((phase, index) => ({
    phase,
    status: index === 0 ? "active" : "idle",
    detail: index === 0 ? "等待任务启动" : "等待前序文件",
  }));
}

function buildPhaseStates(
  total: number,
  filePhaseByPath: Map<string, number>,
  completedFiles: Set<string>,
  batchFinished: boolean,
  cancelled: boolean,
): PhaseDisplayState[] {
  const safeTotal = Math.max(0, total);
  const finalIndex = phases.length - 1;
  if (safeTotal <= 0) return initialPhaseStates();

  return phases.map((phase, index) => {
    if (index === finalIndex) {
      if (batchFinished && !cancelled) {
        return { phase, status: "done", detail: "全部完成" };
      }
      if (completedFiles.size > 0) {
        return { phase, status: "active", detail: `${Math.min(completedFiles.size, safeTotal)}/${safeTotal} 已完成` };
      }
      return { phase, status: "idle", detail: "等待批次完成" };
    }

    let passed = 0;
    let current = 0;
    for (const [path, phaseIndex] of filePhaseByPath.entries()) {
      const isCompleted = completedFiles.has(path);
      if (isCompleted || phaseIndex > index) {
        passed += 1;
      } else if (phaseIndex === index) {
        current += 1;
      }
    }
    for (const path of completedFiles) {
      if (!filePhaseByPath.has(path)) passed += 1;
    }

    const passedClamped = Math.min(passed, safeTotal);
    if (passedClamped >= safeTotal) {
      return { phase, status: "done", detail: `${safeTotal}/${safeTotal} 已通过` };
    }
    if (current > 0) {
      return { phase, status: "active", detail: `${current} 个处理中` };
    }
    if (passedClamped > 0 || index === 0) {
      return { phase, status: "active", detail: `${passedClamped}/${safeTotal} 已通过` };
    }
    return { phase, status: "idle", detail: "等待前序文件" };
  });
}

function summarizeBatchPhase(states: PhaseDisplayState[]): string {
  return states.find((state) => state.status === "active")?.phase || states.find((state) => state.status === "done")?.phase || "待开始";
}

export default function App() {
  const [view, setView] = useState<ViewKey>("new");
  const [inputDir, setInputDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("ome_tiff");
  const [recursive, setRecursive] = useState(true);
  const [progress, setProgress] = useState<ProgressState>(initialProgress);
  const [phaseStates, setPhaseStates] = useState<PhaseDisplayState[]>(() => initialPhaseStates());
  const [logs, setLogs] = useState<string[]>([]);
  const [themeSettings, setThemeSettings] = useState<ThemeSettings>(() => loadThemeSettings());
  const [conversionSettings, setConversionSettings] = useState<ConversionSettings>(() => loadConversionSettings());
  const [taskSettings, setTaskSettings] = useState<ConversionSettings | null>(null);
  const [appVersion, setAppVersion] = useState("0.4.5");
  const [actualTheme, setActualTheme] = useState<ActualTheme>(() => resolveTheme(loadThemeSettings()).theme);
  const [inspection, setInspection] = useState<BatchInspection | null>(null);
  const [inspectionStatus, setInspectionStatus] = useState<"idle" | "running" | "ready" | "error">("idle");
  const [inspectionMessage, setInspectionMessage] = useState("");
  const [inspectionRevision, setInspectionRevision] = useState(0);
  const [inspectionTotal, setInspectionTotal] = useState(0);
  const [inspectionDone, setInspectionDone] = useState(0);
  const [candidateFormatCounts, setCandidateFormatCounts] = useState<Record<string, number>>({});
  const [partialInspectionFiles, setPartialInspectionFiles] = useState<InputInspection[]>([]);
  const [dialog, setDialog] = useState<"channels" | "compatibility" | null>(null);
  const [pendingOverrides, setPendingOverrides] = useState<Record<string, ChannelDefinition[]>>({});
  const themeResolution = useMemo(() => resolveTheme(themeSettings), [themeSettings]);
  const inspectionJob = useRef("");
  const inspectionSequence = useRef(0);
  const fileProgressByPath = useRef<Map<string, number>>(new Map());
  const filePhaseByPath = useRef<Map<string, number>>(new Map());
  const completedFiles = useRef<Set<string>>(new Set());
  const batchTotal = useRef(0);
  const completedCount = useRef(0);
  const lastBatchPercent = useRef(0);
  const batchFinished = useRef(false);
  const batchCancelled = useRef(false);
  const workerReady = useRef(false);

  useEffect(() => {
    void getApplicationVersion().then(setAppVersion);
  }, []);

  useEffect(() => {
    setActualTheme(themeResolution.theme);
    document.documentElement.dataset.theme = themeResolution.theme;
  }, [themeResolution.theme]);

  useEffect(() => {
    const id = window.setInterval(() => {
      const resolution = resolveTheme(themeSettings);
      setActualTheme(resolution.theme);
      document.documentElement.dataset.theme = resolution.theme;
    }, 60_000);
    return () => window.clearInterval(id);
  }, [themeSettings]);

  useEffect(() => {
    let disposed = false;
    let unsubscribe: (() => void) | undefined;
    void (async () => {
      const unlisten = await onConversionEvent(handleConversionEvent);
      if (disposed) {
        unlisten();
        return;
      }
      unsubscribe = unlisten;
      try {
        const status = await ensureWorker();
        if (disposed) return;
        workerReady.current = status.ready;
      } catch (error) {
        if (disposed) return;
        appendLog(`转换引擎启动失败: ${error instanceof Error ? error.message : String(error)}`);
      }
    })();
    return () => {
      disposed = true;
      unsubscribe?.();
    };
  }, []);

  useEffect(() => {
    if (!inputDir) {
      setInspection(null);
      setInspectionStatus("idle");
      setInspectionMessage("");
      return;
    }
    inspectionSequence.current += 1;
    const jobId = `inspect-${Date.now()}-${inspectionSequence.current}`;
    inspectionJob.current = jobId;
    setInspection(null);
    setInspectionStatus("running");
    setInspectionTotal(0);
    setInspectionDone(0);
    setCandidateFormatCounts({});
    setPartialInspectionFiles([]);
    setInspectionMessage(workerReady.current ? "正在扫描输入目录…" : "正在初始化转换引擎…");
    setProgress((state) => state.running ? state : { ...state, status: "inspecting" });
    const timer = window.setTimeout(() => {
      void startInspection({ job_id: jobId, input_dir: inputDir, recursive }).catch((error) => {
        if (inspectionJob.current !== jobId) return;
        const message = error instanceof Error ? error.message : String(error);
        setInspectionStatus("error");
        setInspectionMessage(message);
        setProgress((state) => state.running ? state : {
          ...state,
          status: "error",
          currentPhase: "预检失败",
          finishedAt: Date.now(),
        });
      });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [inputDir, recursive, inspectionRevision]);

  function appendLog(message: string) {
    setLogs((items) => [...items.slice(-299), message]);
  }

  function resetBatchAggregation() {
    fileProgressByPath.current = new Map();
    filePhaseByPath.current = new Map();
    completedFiles.current = new Set();
    batchTotal.current = 0;
    completedCount.current = 0;
    lastBatchPercent.current = 0;
    batchFinished.current = false;
    batchCancelled.current = false;
    setPhaseStates(initialPhaseStates());
  }

  function computeBatchPercent(totalOverride?: number): number {
    const total = totalOverride || batchTotal.current;
    if (total <= 0) return lastBatchPercent.current;
    let activeProgress = 0;
    for (const [path, fraction] of fileProgressByPath.current.entries()) {
      if (!completedFiles.current.has(path)) {
        activeProgress += Math.max(0, Math.min(1, fraction));
      }
    }
    const computed = ((completedCount.current + activeProgress) / total) * 100;
    const next = Math.min(99.9, Math.max(lastBatchPercent.current, computed));
    lastBatchPercent.current = next;
    return next;
  }

  function updatePhaseStates(totalOverride?: number): PhaseDisplayState[] {
    const next = buildPhaseStates(
      totalOverride || batchTotal.current,
      filePhaseByPath.current,
      completedFiles.current,
      batchFinished.current,
      batchCancelled.current,
    );
    setPhaseStates(next);
    return next;
  }

  function handleConversionEvent(event: ConversionEvent) {
    if (event.type === "inspection_started") {
      if (event.job_id === inspectionJob.current) setInspectionStatus("running");
      return;
    }
    if (event.type === "inspection_discovered") {
      if (event.job_id !== inspectionJob.current) return;
      setInspectionTotal(event.total);
      setCandidateFormatCounts(event.format_counts);
      setInspectionMessage(`发现 ${event.total} 张候选 WSI · 正在读取通道 0/${event.total}`);
      return;
    }
    if (event.type === "inspection_file_done") {
      if (event.job_id !== inspectionJob.current) return;
      setInspectionDone(event.done);
      setPartialInspectionFiles((files) => [
        ...files.filter((file) => file.input_path !== event.file.input_path),
        event.file,
      ]);
      setInspectionMessage(`发现 ${event.total} 张候选 WSI · 已读取通道 ${event.done}/${event.total}`);
      return;
    }
    if (event.type === "inspection_progress") {
      if (event.job_id !== inspectionJob.current) return;
      setInspectionMessage(
        event.total > 0
          ? `正在预检 ${event.done}/${event.total} · ${basename(event.current)}`
          : "正在扫描输入目录…",
      );
      return;
    }
    if (event.type === "inspection_done") {
      if (event.job_id !== inspectionJob.current) return;
      setInspection(event.inspection);
      setInspectionStatus("ready");
      setInspectionTotal(event.inspection.files.length);
      setInspectionDone(event.inspection.files.length);
      setPartialInspectionFiles(event.inspection.files);
      setInspectionMessage(`检测到 ${event.inspection.files.length} 张 WSI`);
      setProgress((state) => state.running ? state : { ...state, status: "ready" });
      if (event.duration_ms !== undefined) appendLog(`inspection_ms=${event.duration_ms.toFixed(1)}`);
      return;
    }
    if (event.type === "inspection_error") {
      if (event.job_id && event.job_id !== inspectionJob.current) return;
      setInspectionStatus("error");
      setInspectionMessage(event.message);
      setProgress((state) => state.running ? state : { ...state, status: "error" });
      return;
    }
    if (event.type === "ready") {
      workerReady.current = true;
      appendLog(event.banner || "Python worker ready");
      return;
    }
    if (event.type === "started") {
      setProgress((state) => ({ ...state, running: true, status: "running" }));
      return;
    }
    if (event.type === "report_path") {
      setProgress((state) => ({ ...state, reportPath: event.path }));
      return;
    }
    if (event.type === "log") {
      appendLog(event.message);
      return;
    }
    if (event.type === "overall") {
      batchTotal.current = Math.max(batchTotal.current, event.total);
      if (event.done > completedCount.current && event.current) {
        completedFiles.current.add(event.current);
        fileProgressByPath.current.set(event.current, 1);
        filePhaseByPath.current.set(event.current, phases.length - 1);
      }
      completedCount.current = Math.max(completedCount.current, event.done, completedFiles.current.size);
      const nextPhaseStates = updatePhaseStates(event.total);
      const nextPercent = computeBatchPercent(event.total);
      setProgress((state) => ({
        ...state,
        currentFile: basename(event.current),
        currentPhase: summarizeBatchPhase(nextPhaseStates),
        batchDone: completedCount.current,
        batchTotal: event.total,
        batchPercent: nextPercent,
        etaText: formatEta(state.startedAt, nextPercent),
      }));
      return;
    }
    if (event.type === "performance") {
      setProgress((state) => ({
        ...state,
        memoryMb: Math.max(0, event.memory_mb),
        cpuPercent: Math.min(100, Math.max(0, event.cpu_percent)),
        etaText: state.running ? formatEta(state.startedAt, state.batchPercent) : state.etaText,
      }));
      return;
    }
    if (event.type === "file_progress") {
      const phase = normalizePhase(event.level);
      const phaseIndex = phaseToIndex(phase);
      const fileFraction = event.overall_total > 0 ? event.overall_done / event.overall_total : 0;
      const previousFraction = fileProgressByPath.current.get(event.current) || 0;
      const previousPhase = filePhaseByPath.current.get(event.current) ?? 0;
      fileProgressByPath.current.set(event.current, Math.max(previousFraction, Math.max(0, Math.min(1, fileFraction))));
      filePhaseByPath.current.set(event.current, Math.max(previousPhase, phaseIndex));
      const nextPhaseStates = updatePhaseStates();
      const nextPercent = computeBatchPercent();
      setProgress((state) => {
        return {
          ...state,
          running: true,
          status: "running",
          currentFile: basename(event.current),
          currentPhase: summarizeBatchPhase(nextPhaseStates),
          stagePercent: formatPercent(event.done, event.total),
          filePercent: formatPercent(event.overall_done, event.overall_total),
          batchDone: completedCount.current,
          batchTotal: batchTotal.current || state.batchTotal,
          batchPercent: nextPercent,
          etaText: formatEta(state.startedAt, nextPercent),
        };
      });
      return;
    }
    if (event.type === "done") {
      const batch = event.batch;
      const failed = batch.failed_count > 0;
      batchFinished.current = true;
      batchCancelled.current = batch.cancelled;
      batchTotal.current = batch.total_files;
      for (const result of batch.results) {
        completedFiles.current.add(result.input_path);
        fileProgressByPath.current.set(result.input_path, 1);
        filePhaseByPath.current.set(result.input_path, phases.length - 1);
        if (result.compatibility_level === "static_unverified") {
          appendLog(
            `${basename(result.input_path)}: static_unverified · ${result.source_container || "KFB"} ${result.source_version || "unknown"}`,
          );
        }
        if (result.diagnostic_code) {
          appendLog(
            `${basename(result.input_path)}: ${result.diagnostic_code} · ${result.diagnostic_stage || "parse"}`,
          );
        }
        if (result.svs_omitted_native_data) {
          appendLog(
            `${basename(result.input_path)}: SVS 未保存原始数据 · ${result.svs_omitted_native_data}`,
          );
        }
      }
      completedCount.current = Math.max(completedFiles.current.size, batch.results.length);
      if (!batch.cancelled) lastBatchPercent.current = 100;
      const nextPhaseStates = updatePhaseStates(batch.total_files);
      setProgress((state) => ({
        ...state,
        running: false,
        status: batch.cancelled ? "cancelled" : failed ? "error" : "success",
        currentPhase: batch.cancelled ? "已取消" : failed ? "转换失败" : summarizeBatchPhase(nextPhaseStates),
        batchDone: completedCount.current,
        batchTotal: batch.total_files,
        batchPercent: batch.cancelled ? state.batchPercent : 100,
        etaText: batch.cancelled || failed ? "—" : "已完成",
        reportPath: batch.report_path || state.reportPath,
        backend: batch.results.length ? batch.results[batch.results.length - 1].backend : state.backend,
        finishedAt: Date.now(),
        successCount: batch.success_count,
        failedCount: batch.failed_count,
        skippedCount: batch.skipped_count,
      }));
      return;
    }
    if (event.type === "error") {
      batchFinished.current = true;
      batchCancelled.current = false;
      setProgress((state) => ({
        ...state,
        running: false,
        status: "error",
        currentPhase: "转换失败",
        etaText: "—",
        finishedAt: Date.now(),
      }));
      appendLog(event.traceback || event.message);
      if (event.diagnostic_code) {
        appendLog(`${event.diagnostic_code} · ${event.diagnostic_stage || "unknown"}`);
      }
      return;
    }
    if (event.type === "worker_terminated") {
      workerReady.current = false;
      const message = `转换引擎异常退出: ${event.code ?? "-"} ${event.signal ?? ""}`.trim();
      if (event.activity === "inspecting") {
        appendLog(message);
        if (event.job_id && event.job_id !== inspectionJob.current) return;
        setInspectionStatus("error");
        setInspectionMessage("转换引擎异常退出，请重新预检");
        setProgress((state) => ({
          ...state,
          running: false,
          status: "error",
          currentPhase: "预检失败",
          etaText: "—",
          finishedAt: Date.now(),
        }));
        return;
      }
      if (!event.busy) return;
      batchFinished.current = true;
      batchCancelled.current = false;
      setProgress((state) => ({
        ...state,
        running: false,
        status: "error",
        currentPhase: "异常中止",
        etaText: "—",
        finishedAt: Date.now(),
      }));
      appendLog(message);
      return;
    }
  }

  async function pickInput() {
    const path = await chooseDirectory();
    if (!path) return;
    resetBatchAggregation();
    setLogs([]);
    setTaskSettings(null);
    setDialog(null);
    setPendingOverrides({});
    setProgress({
      ...initialProgress,
      status: "inspecting",
      currentFile: basename(path),
      outputDir,
    });
    setInputDir(path);
    if (path === inputDir) setInspectionRevision((revision) => revision + 1);
  }

  async function pickOutput() {
    const path = await chooseDirectory();
    if (!path) return;
    setOutputDir(path);
    setProgress((state) => ({ ...state, outputDir: path }));
  }

  function incompatibleFiles(): number {
    if (!inspection) return 0;
    return inspection.files.filter(
      (file) => Boolean(file.error) || !file.allowed_output_formats.includes(outputFormat),
    ).length;
  }

  function unknownChannelFiles() {
    return (inspection?.files || []).filter(
      (file) => file.source_modality === "fluorescence"
        && file.channel_definitions.some((channel) => channel.identity_source === "unknown"),
    );
  }

  async function beginConversion(
    convertCompatibleOnly: boolean,
    channelOverrides: Record<string, ChannelDefinition[]>,
  ) {
    if (!inputDir || !outputDir || progress.running || !inspection) return;
    const jobId = `job-${Date.now()}`;
    const settingsSnapshot = { ...conversionSettings };
    resetBatchAggregation();
    setLogs([]);
    setTaskSettings(settingsSnapshot);
    setProgress({
      ...initialProgress,
      running: true,
      status: "starting",
      currentFile: basename(inputDir),
      outputDir,
      startedAt: Date.now(),
      backend: outputFormat === "svs"
        ? "svs-streaming-direct"
        : outputFormat === "fluorescence_svs"
          ? "tifffile-fluorescence-svs"
          : outputFormat === "afi"
            ? "tifffile-afi"
            : "tifffile-ome",
    });
    try {
      await startConversion({
        job_id: jobId,
        input_dir: inputDir,
        output_dir: outputDir,
        output_format: outputFormat,
        recursive,
        memory_budget_mb: settingsSnapshot.memory_budget_mb,
        tile_size: 256,
        jpeg_quality: settingsSnapshot.jpeg_quality,
        parallel_wsi: settingsSnapshot.parallel_wsi,
        selected_input_paths: inspection.files.map((file) => file.input_path),
        convert_compatible_only: convertCompatibleOnly,
        channel_overrides: channelOverrides,
        input_signatures: Object.fromEntries(
          inspection.files.map((file) => [
            file.input_path,
            { size: file.file_size, mtime_ns: file.file_mtime_ns },
          ]),
        ),
        preflight_files: inspection.files,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setProgress((state) => ({
        ...state,
        running: false,
        status: "error",
        finishedAt: Date.now(),
      }));
      appendLog(message);
    }
  }

  function runConversion() {
    if (!inspection || inspectionStatus !== "ready") return;
    setPendingOverrides({});
    if (unknownChannelFiles().length > 0) {
      setDialog("channels");
      return;
    }
    if (incompatibleFiles() > 0) {
      setDialog("compatibility");
      return;
    }
    void beginConversion(false, {});
  }

  function editChannels() {
    if (inspection) setDialog("channels");
  }

  function acceptChannels(overrides: Record<string, ChannelDefinition[]>) {
    setPendingOverrides(overrides);
    setDialog(null);
    if (incompatibleFiles() > 0) {
      setDialog("compatibility");
    } else {
      void beginConversion(false, overrides);
    }
  }

  async function cancelCurrent() {
    await cancelConversion();
    appendLog("已请求取消，等待当前写入步骤安全结束");
  }

  function resetTask() {
    if (progress.running) return;
    resetBatchAggregation();
    setLogs([]);
    setTaskSettings(null);
    setDialog(null);
    setPendingOverrides({});
    setProgress({
      ...initialProgress,
      status: inputDir ? "inspecting" : "idle",
      currentFile: inputDir ? basename(inputDir) : initialProgress.currentFile,
      outputDir,
    });
    if (inputDir) setInspectionRevision((revision) => revision + 1);
  }

  function updateTheme(next: ThemeSettings) {
    setThemeSettings(next);
    saveThemeSettings(next);
  }

  function updateConversionSettings(next: ConversionSettings) {
    const normalized = normalizeConversionSettings(next);
    setConversionSettings(normalized);
    saveConversionSettings(normalized);
  }

  async function openPathWithFeedback(path: string, label: string) {
    try {
      await openFilesystemPath(path);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendLog(`${label}失败: ${message}`);
    }
  }

  const formatCounts: Record<OutputFormat, number> = {
    ome_tiff: inspection?.files.filter((file) => file.allowed_output_formats.includes("ome_tiff")).length || 0,
    svs: inspection?.files.filter((file) => file.allowed_output_formats.includes("svs")).length || 0,
    fluorescence_svs: inspection?.files.filter((file) => file.allowed_output_formats.includes("fluorescence_svs")).length || 0,
    afi: inspection?.files.filter((file) => file.allowed_output_formats.includes("afi")).length || 0,
  };
  const canStart = Boolean(
    inputDir
    && outputDir
    && !progress.running
    && inspectionStatus === "ready"
    && formatCounts[outputFormat] > 0,
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">I</div>
          <div>
            <strong>镜渡 SlideBridge</strong>
            <span>Universal WSI Converter</span>
          </div>
        </div>
        <nav className="nav-list">
          <NavItem active={view === "new"} icon={<ChevronRight size={16} />} label="转换任务" onClick={() => setView("new")} />
          <NavItem active={view === "settings"} icon={<Settings size={16} />} label="设置" onClick={() => setView("settings")} />
        </nav>
        <div className="sidebar-footer">
          <span>v{appVersion}</span>
          <span>{actualTheme === "light" ? "Light" : "Dark"}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{inputDir ? basename(inputDir) : "未选择任务"}</h1>
            <p>{inputDir ? "确认输出目录和格式后即可开始转换。" : "选择一个输入目录后开始转换。"}</p>
          </div>
          <div className={`status-chip ${progress.status}`}>{taskStatusLabels[progress.status]}</div>
        </header>

        {view === "settings" ? (
          <SettingsView
            settings={themeSettings}
            conversionSettings={conversionSettings}
            onChange={updateTheme}
            onConversionChange={updateConversionSettings}
          />
        ) : (
          <div className="work-grid">
            <section className="task-flow">
              <TaskStatusBanner
                progress={progress}
                inspectionMessage={inspectionMessage}
                onOpenOutput={() => openPathWithFeedback(outputDir, "打开输出目录")}
                onOpenReport={() => openPathWithFeedback(progress.reportPath, "打开转换报告")}
              />
              <div className="phase-list">
                {phaseStates.map((phaseState) => (
                  <div
                    key={phaseState.phase}
                    className={`phase-item ${phaseState.status}`}
                  >
                    <span className="phase-dot" />
                    <div>
                      <strong>{phaseState.phase}</strong>
                      <p>{phaseState.detail}</p>
                    </div>
                  </div>
                ))}
              </div>

              <section className="transcript">
                <div className="transcript-header">
                  <span>Transcript</span>
                  <span>{logs.length} lines</span>
                </div>
                <pre>{logs.join("\n") || "等待 Python worker 输出日志。"}</pre>
              </section>
            </section>

            <aside className="summary-panel">
              <ProgressSummary progress={progress} />
              <PerformanceSummary progress={progress} settings={taskSettings ?? conversionSettings} />
            </aside>
          </div>
        )}

        {view !== "settings" && (
          <TaskComposer
            inputDir={inputDir}
            outputDir={outputDir}
            outputFormat={outputFormat}
            conversionSettings={conversionSettings}
            recursive={recursive}
            canStart={canStart}
            running={progress.running}
            taskStatus={progress.status}
            reportPath={progress.reportPath}
            inspection={inspection}
            inspectionStatus={inspectionStatus}
            inspectionMessage={inspectionMessage}
            inspectionTotal={inspectionTotal}
            inspectionDone={inspectionDone}
            candidateFormatCounts={candidateFormatCounts}
            partialInspectionFiles={partialInspectionFiles}
            formatCounts={formatCounts}
            onInput={pickInput}
            onOutput={pickOutput}
            onFormat={setOutputFormat}
            onRecursive={setRecursive}
            onStart={runConversion}
            onCancel={cancelCurrent}
            onReset={resetTask}
            onEditChannels={editChannels}
            onOpenOutput={() => openPathWithFeedback(outputDir, "打开输出目录")}
            onOpenReport={() => openPathWithFeedback(progress.reportPath, "打开转换报告")}
          />
        )}
      </main>
      {dialog === "channels" && inspection && (
        <ChannelDialog
          files={inspection.files}
          onCancel={() => setDialog(null)}
          onContinue={acceptChannels}
        />
      )}
      {dialog === "compatibility" && inspection && (
        <CompatibilityDialog
          incompatibleCount={incompatibleFiles()}
          compatibleCount={formatCounts[outputFormat]}
          outputFormat={outputFormat}
          onBack={() => setDialog(null)}
          onContinue={() => {
            setDialog(null);
            void beginConversion(true, pendingOverrides);
          }}
        />
      )}
    </div>
  );
}

function NavItem({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

export function TaskStatusBanner({
  progress,
  inspectionMessage,
  onOpenOutput,
  onOpenReport,
}: {
  progress: ProgressState;
  inspectionMessage: string;
  onOpenOutput: () => void;
  onOpenReport: () => void;
}) {
  if (progress.status === "idle") return null;

  const isBusy = progress.status === "inspecting" || progress.status === "starting" || progress.status === "running";
  const isPreflightError = progress.status === "error" && progress.startedAt === null;
  const icon = isBusy
    ? <LoaderCircle className="status-banner-spinner" size={20} />
    : progress.status === "ready" || progress.status === "success"
      ? <CheckCircle2 size={20} />
      : <CircleAlert size={20} />;
  const title = progress.status === "inspecting"
    ? "正在预检输入文件"
    : progress.status === "ready"
      ? "预检完成，可以开始转换"
      : progress.status === "starting"
        ? "正在启动转换"
        : progress.status === "running"
          ? `正在转换 ${progress.currentFile}`
          : progress.status === "success"
            ? "转换完成"
            : progress.status === "cancelled"
              ? "转换已取消"
              : isPreflightError ? "预检失败" : "转换失败";
  const detail = progress.status === "inspecting" || progress.status === "ready"
    ? inspectionMessage
    : progress.status === "starting"
      ? "正在复用转换引擎并准备任务…"
      : progress.status === "running"
        ? `${progress.currentPhase} · ${Math.round(progress.batchPercent)}% · ETA ${progress.etaText}`
        : progress.status === "success"
          ? `成功 ${progress.successCount} · 失败 ${progress.failedCount} · 跳过 ${progress.skippedCount} · 用时 ${formatDuration(progress.startedAt, progress.finishedAt)}`
          : isPreflightError ? inspectionMessage : progress.currentPhase;

  return (
    <section className={`task-status-banner ${progress.status}`} aria-live="polite">
      <span className="task-status-icon">{icon}</span>
      <div>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
      {(progress.status === "success" || progress.status === "error" || progress.status === "cancelled") && (
        <div className="task-status-actions">
          <button className="ghost" disabled={!progress.outputDir} onClick={onOpenOutput}>打开输出</button>
          <button className="ghost" disabled={!progress.reportPath} onClick={onOpenReport}>查看报告</button>
        </div>
      )}
    </section>
  );
}

function ProgressSummary({ progress }: { progress: ProgressState }) {
  return (
    <div className="progress-card">
      <span>进度摘要</span>
      <strong>{Math.round(progress.batchPercent)}%</strong>
      <div className="progress-bar">
        <i style={{ width: `${Math.max(0, Math.min(100, progress.batchPercent))}%` }} />
      </div>
      <dl>
        <div>
          <dt>当前文件</dt>
          <dd>{progress.currentFile}</dd>
        </div>
        <div>
          <dt>阶段</dt>
          <dd>{progress.currentPhase}</dd>
        </div>
        <div>
          <dt>批次</dt>
          <dd>
            {progress.batchDone}/{progress.batchTotal || 0}
          </dd>
        </div>
        <div>
          <dt>后端</dt>
          <dd>{progress.backend}</dd>
        </div>
      </dl>
    </div>
  );
}

function PerformanceSummary({
  progress,
  settings,
}: {
  progress: ProgressState;
  settings: ConversionSettings;
}) {
  const memoryPercent = settings.memory_budget_mb > 0
    ? Math.min(100, (progress.memoryMb / settings.memory_budget_mb) * 100)
    : 0;
  const cpuPercent = Math.min(100, Math.max(0, progress.cpuPercent));

  return (
    <section className="performance-card">
      <h2>性能监控</h2>
      <dl className="performance-text-metrics">
        <div>
          <dt>并发 WSI 数量</dt>
          <dd>{settings.parallel_wsi}</dd>
        </div>
        <div>
          <dt>JPG 质量</dt>
          <dd>{settings.jpeg_quality}</dd>
        </div>
        <div>
          <dt>ETA</dt>
          <dd>{progress.running ? progress.etaText : progress.status === "success" ? "已完成" : "—"}</dd>
        </div>
      </dl>
      <PerformanceBar
        label="内存消耗"
        value={`${Math.round(progress.memoryMb)} / ${settings.memory_budget_mb} MB`}
        percent={memoryPercent}
      />
      <PerformanceBar label="CPU 占用" value={`${Math.round(cpuPercent)}%`} percent={cpuPercent} />
    </section>
  );
}

function PerformanceBar({ label, value, percent }: { label: string; value: string; percent: number }) {
  return (
    <div className="performance-metric">
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <div
        className="performance-bar"
        role="progressbar"
        aria-label={`${label} ${value}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(Math.min(100, Math.max(0, percent)))}
      >
        <i style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
      </div>
    </div>
  );
}

export function TaskComposer({
  inputDir,
  outputDir,
  outputFormat,
  conversionSettings,
  recursive,
  canStart,
  running,
  taskStatus,
  reportPath,
  inspection,
  inspectionStatus,
  inspectionMessage,
  inspectionTotal,
  inspectionDone,
  candidateFormatCounts,
  partialInspectionFiles,
  formatCounts,
  onInput,
  onOutput,
  onFormat,
  onRecursive,
  onStart,
  onCancel,
  onReset,
  onEditChannels,
  onOpenOutput,
  onOpenReport,
}: {
  inputDir: string;
  outputDir: string;
  outputFormat: OutputFormat;
  conversionSettings: ConversionSettings;
  recursive: boolean;
  canStart: boolean;
  running: boolean;
  taskStatus: TaskStatus;
  reportPath: string;
  inspection: BatchInspection | null;
  inspectionStatus: "idle" | "running" | "ready" | "error";
  inspectionMessage: string;
  inspectionTotal: number;
  inspectionDone: number;
  candidateFormatCounts: Record<string, number>;
  partialInspectionFiles: InputInspection[];
  formatCounts: Record<OutputFormat, number>;
  onInput: () => void;
  onOutput: () => void;
  onFormat: (format: OutputFormat) => void;
  onRecursive: (value: boolean) => void;
  onStart: () => void;
  onCancel: () => void;
  onReset: () => void;
  onEditChannels: () => void;
  onOpenOutput: () => void;
  onOpenReport: () => void;
}) {
  return (
    <section className="composer">
      <div className="path-row">
        <PathButton label="输入目录" value={inputDir || "选择包含 IBL / KFB / IMAGE / SVS / TIFF / AFI 的文件夹"} onClick={onInput} disabled={running} />
        <PathButton label="输出目录" value={outputDir || "选择转换结果保存位置"} onClick={onOutput} disabled={running} />
      </div>
      <div className="format-note" aria-label="支持的输入格式">
        <span className="format-note-label">输入格式</span>
        <span className="format-note-values">
          .ibl · .kfb/.kfbl/.kfbf/.kfba/.kfbx · .image · .svs · .tif/.tiff · .afi
        </span>
        <span className="format-note-auto">自动识别</span>
      </div>
      <InspectionSummary
        inspection={inspection}
        status={inspectionStatus}
        message={inspectionMessage}
        total={inspectionTotal}
        done={inspectionDone}
        candidateFormatCounts={candidateFormatCounts}
        partialFiles={partialInspectionFiles}
      />
      <div className="action-row">
        <div className="segment format-segment">
          {([
            ["ome_tiff", "Pyramidal OME-TIFF"],
            ["svs", "明场 SVS"],
            ["fluorescence_svs", "荧光 SVS"],
            ["afi", "AFI"],
          ] as [OutputFormat, string][]).map(([format, label]) => (
            <button
              key={format}
              className={outputFormat === format ? "selected" : ""}
              disabled={running || (inspectionStatus === "ready" && formatCounts[format] === 0)}
              onClick={() => onFormat(format)}
            >
              {label}
              {inspectionStatus === "ready" && <small>{formatCounts[format]}/{inspection?.files.length || 0}</small>}
            </button>
          ))}
        </div>
        <label className="check-row">
          <input type="checkbox" checked={recursive} disabled={running} onChange={(event) => onRecursive(event.target.checked)} />
          包含子文件夹
        </label>
        <button className="primary" disabled={!canStart} onClick={onStart}>
          {taskStatus === "starting" || taskStatus === "running"
            ? <LoaderCircle className="button-spinner" size={16} />
            : <Play size={16} />}
          {taskStatus === "starting" ? "正在启动" : taskStatus === "running" ? "转换中" : "开始转换"}
        </button>
        <button className="soft" disabled={!running} onClick={onCancel}>
          <Square size={14} />
          取消
        </button>
        <button className="soft" disabled={running} onClick={onReset}>
          <RotateCcw size={14} />
          重置任务
        </button>
        <button className="soft" disabled={running || !inspection?.files.some((file) => file.source_modality === "fluorescence")} onClick={onEditChannels}>
          编辑通道
        </button>
        <button className="ghost" disabled={!outputDir} onClick={onOpenOutput}>
          打开输出
        </button>
        <button className="ghost" disabled={!reportPath} onClick={onOpenReport}>
          查看报告
        </button>
      </div>
      <div className="composer-meta">
        并行 {conversionSettings.parallel_wsi} · JPEG {conversionSettings.jpeg_quality} · 内存 {conversionSettings.memory_budget_mb} MB
      </div>
    </section>
  );
}

export function InspectionSummary({
  inspection,
  status,
  message,
  total,
  done,
  candidateFormatCounts,
  partialFiles,
}: {
  inspection: BatchInspection | null;
  status: "idle" | "running" | "ready" | "error";
  message: string;
  total: number;
  done: number;
  candidateFormatCounts: Record<string, number>;
  partialFiles: InputInspection[];
}) {
  const files = status === "ready" && inspection ? inspection.files : partialFiles;
  const formatCounts = files.length > 0
    ? files.reduce<Record<string, number>>((counts, file) => {
        const label = inspectionFormatLabel(file);
        counts[label] = (counts[label] || 0) + 1;
        return counts;
      }, {})
    : Object.fromEntries(
        Object.entries(candidateFormatCounts).map(([extension, count]) => [extension.toUpperCase(), count]),
      );
  const brightfield = files.filter((file) => file.source_modality === "brightfield").length;
  const fluorescence = files.filter((file) => file.source_modality === "fluorescence").length;
  const failed = files.filter((file) => Boolean(file.error)).length;
  const unknownChannels = files.reduce(
    (count, file) => count + (file.source_modality === "fluorescence"
      ? file.channel_definitions.filter((channel) => channel.identity_source === "unknown").length
      : 0),
    0,
  );
  const recognizedChannels = files.reduce(
    (count, file) => count + (file.source_modality === "fluorescence"
      ? file.channel_definitions.filter((channel) => channel.identity_source !== "unknown").length
      : 0),
    0,
  );
  if (status === "idle") {
    return <div className="inspection-summary idle">{message || "选择输入目录后自动预检"}</div>;
  }
  return (
    <div className={`inspection-summary ${status}`} aria-live="polite">
      <span>{status === "ready" ? "检测完成" : status === "error" ? "检测失败" : "正在检测"}</span>
      <strong>{status === "ready" ? `${files.length} 张 WSI` : `${done}/${total || 0}`}</strong>
      <div className="inspection-formats">
        {Object.entries(formatCounts)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([label, count]) => <span className="format-badge" key={label}>{label} {count}</span>)}
      </div>
      {files.length > 0 && <span>明场 {brightfield} · 荧光 {fluorescence}</span>}
      {recognizedChannels > 0 && <span>已识别通道 {recognizedChannels}</span>}
      {unknownChannels > 0 && <span className="warning">未知通道 {unknownChannels}</span>}
      {failed > 0 && <span className="warning">无法识别 {failed}</span>}
      {message && status === "error" && <span className="warning">{message}</span>}
    </div>
  );
}

function inspectionFormatLabel(file: InputInspection): string {
  const container = (file.source_container || "").toLowerCase();
  const labels: Record<string, string> = {
    ome_tiff: "OME-TIFF",
    fluorescence_svs: "荧光 SVS",
    generic_tiff: "TIFF",
    svs: "SVS",
    kfb: "KFB",
    kfbl: "KFBL",
    kfbf: "KFBF",
    kfba: "KFBA",
    kfbx: "KFBX",
    afi: "AFI",
  };
  return labels[container] || labels[file.input_format] || file.input_format.toUpperCase();
}

const channelPresets: Record<string, { color: [number, number, number]; excitation_nm: number | null; emission_nm: number | null }> = {
  DAPI: { color: [0, 0, 255], excitation_nm: 358, emission_nm: 461 },
  FITC: { color: [0, 255, 0], excitation_nm: 495, emission_nm: 519 },
  TRITC: { color: [255, 80, 0], excitation_nm: 550, emission_nm: 570 },
  Cy3: { color: [255, 165, 0], excitation_nm: 550, emission_nm: 570 },
  Cy5: { color: [255, 0, 80], excitation_nm: 650, emission_nm: 670 },
  AF: { color: [255, 255, 255], excitation_nm: null, emission_nm: null },
};

type ChannelDraft = ChannelDefinition & { preset: string };

function colorHex(color: [number, number, number]): string {
  return `#${color.map((value) => Math.max(0, Math.min(255, value)).toString(16).padStart(2, "0")).join("")}`;
}

function parseColor(value: string): [number, number, number] {
  const cleaned = value.replace("#", "");
  return [0, 2, 4].map((index) => Number.parseInt(cleaned.slice(index, index + 2), 16)) as [number, number, number];
}

function parentPath(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  return normalized.slice(0, normalized.lastIndexOf("/"));
}

export function ChannelDialog({
  files,
  onCancel,
  onContinue,
}: {
  files: InputInspection[];
  onCancel: () => void;
  onContinue: (overrides: Record<string, ChannelDefinition[]>) => void;
}) {
  const [search, setSearch] = useState("");
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const nameInputs = useRef<Record<string, HTMLInputElement | null>>({});
  const [drafts, setDrafts] = useState<Record<string, ChannelDraft[]>>(() =>
    Object.fromEntries(
      files
        .filter((file) => file.source_modality === "fluorescence" && file.channel_definitions.length > 0)
        .map((file) => [
          file.input_path,
          file.channel_definitions.map((channel) => ({
            ...channel,
            color: [...channel.color] as [number, number, number],
            preset: channelPresets[channel.name] ? channel.name : "custom",
          })),
        ]),
    ),
  );
  const visibleFiles = files.filter(
    (file) => file.source_modality === "fluorescence"
      && file.channel_definitions.length > 0
      && file.input_path.toLowerCase().includes(search.trim().toLowerCase()),
  );
  const folderGroups = Object.entries(
    visibleFiles.reduce<Record<string, InputInspection[]>>((groups, file) => {
      const folder = parentPath(file.input_path);
      (groups[folder] ||= []).push(file);
      return groups;
    }, {}),
  );

  function update(path: string, index: number, patch: Partial<ChannelDraft>) {
    setDrafts((current) => ({
      ...current,
      [path]: current[path].map((channel) => channel.index === index
        ? { ...channel, ...patch, identity_source: "user_supplied" }
        : channel),
    }));
  }

  function applyPreset(path: string, index: number, presetName: string) {
    if (presetName === "custom") {
      update(path, index, { preset: "custom" });
      window.setTimeout(() => {
        const input = nameInputs.current[`${path}:${index}`];
        input?.focus();
        input?.select();
      }, 50);
      return;
    }
    const preset = channelPresets[presetName];
    if (!preset) return;
    update(path, index, {
      preset: presetName,
      name: presetName,
      fluor: presetName,
      color: preset.color,
      excitation_nm: preset.excitation_nm,
      emission_nm: preset.emission_nm,
    });
  }

  function applyToFolder(path: string, definition: ChannelDraft) {
    const folder = parentPath(path);
    setDrafts((current) => Object.fromEntries(
      Object.entries(current).map(([filePath, channels]) => [
        filePath,
        parentPath(filePath) === folder
          ? channels.map((channel) => channel.index === definition.index
            ? { ...definition, identity_source: "user_supplied" as const }
            : channel)
          : channels,
      ]),
    ));
  }

  function applyToSelected(definition: ChannelDraft) {
    setDrafts((current) => Object.fromEntries(
      Object.entries(current).map(([filePath, channels]) => [
        filePath,
        selectedPaths.has(filePath)
          ? channels.map((channel) => channel.index === definition.index
            ? { ...definition, identity_source: "user_supplied" as const }
            : channel)
          : channels,
      ]),
    ));
  }

  function toggleSelected(path: string) {
    setSelectedPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function serializedDrafts(): Record<string, ChannelDefinition[]> {
    return Object.fromEntries(
      Object.entries(drafts).map(([path, channels]) => [
        path,
        channels.map(({ preset: _preset, ...channel }) => channel),
      ]),
    );
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="确认荧光通道定义">
      <section className="modal-card channel-modal">
        <header>
          <div>
            <h2>确认荧光通道定义</h2>
            <p>修改只影响输出元数据、显示颜色和文件名，不改变任何像素。</p>
          </div>
          <button className="ghost" onClick={onCancel}>关闭</button>
        </header>
        <input
          className="channel-search"
          placeholder="搜索文件或目录"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <div className="channel-files">
          {folderGroups.map(([folder, folderFiles]) => (
            <section className="channel-folder" key={folder}>
              <h3>{folder || "根目录"}</h3>
              {folderFiles.map((file) => (
                <details key={file.input_path} open>
                  <summary>
                    <input
                      type="checkbox"
                      checked={selectedPaths.has(file.input_path)}
                      aria-label={`选择 ${basename(file.input_path)}`}
                      onClick={(event) => event.stopPropagation()}
                      onChange={() => toggleSelected(file.input_path)}
                    />
                    <strong>{basename(file.input_path)}</strong>
                    <span>{file.source_container || file.input_format} · C={file.channel_count} Z={file.z_count} T={file.t_count}</span>
                  </summary>
                  {(drafts[file.input_path] || []).map((channel) => (
                    <div className="channel-row" key={`${file.input_path}-${channel.index}`}>
                      <span className={`channel-source ${channel.identity_source === "unknown" ? "unknown" : ""}`}>
                        C{channel.index + 1} · {channel.identity_source}
                      </span>
                      <Select.Root value={channel.preset} onValueChange={(value) => applyPreset(file.input_path, channel.index, value)}>
                        <Select.Trigger className="channel-select-trigger" aria-label={`C${channel.index + 1} 荧光预设`}>
                          <Select.Value />
                          <Select.Icon><ChevronDown size={14} /></Select.Icon>
                        </Select.Trigger>
                        <Select.Portal>
                          <Select.Content className="channel-select-content" position="popper" sideOffset={6}>
                            <Select.Viewport>
                              {["custom", ...Object.keys(channelPresets)].map((name) => (
                                <Select.Item className="channel-select-item" key={name} value={name}>
                                  <Select.ItemText>{name === "custom" ? "自定义" : name}</Select.ItemText>
                                  <Select.ItemIndicator><Check size={14} /></Select.ItemIndicator>
                                </Select.Item>
                              ))}
                            </Select.Viewport>
                          </Select.Content>
                        </Select.Portal>
                      </Select.Root>
                      <input
                        ref={(element) => { nameInputs.current[`${file.input_path}:${channel.index}`] = element; }}
                        value={channel.name}
                        aria-label="通道名称"
                        onChange={(event) => update(file.input_path, channel.index, {
                          name: event.target.value,
                          fluor: event.target.value,
                          preset: "custom",
                        })}
                      />
                      <input type="color" value={colorHex(channel.color)} aria-label="显示色" onChange={(event) => update(file.input_path, channel.index, { color: parseColor(event.target.value), preset: "custom" })} />
                      <input type="number" placeholder="激发 nm" value={channel.excitation_nm ?? ""} onChange={(event) => update(file.input_path, channel.index, { excitation_nm: event.target.value ? Number(event.target.value) : null, preset: "custom" })} />
                      <input type="number" placeholder="发射 nm" value={channel.emission_nm ?? ""} onChange={(event) => update(file.input_path, channel.index, { emission_nm: event.target.value ? Number(event.target.value) : null, preset: "custom" })} />
                      <button className="ghost" disabled={selectedPaths.size === 0} onClick={() => applyToSelected(channel)}>应用到选中文件</button>
                      <button className="ghost" onClick={() => applyToFolder(file.input_path, channel)}>应用到当前文件夹</button>
                    </div>
                  ))}
                  <p className="output-preview">输出预览：{basename(file.input_path).replace(/\.[^.]+$/, "")}_C01_{drafts[file.input_path]?.[0]?.name || "C1"}.svs</p>
                </details>
              ))}
            </section>
          ))}
        </div>
        <footer>
          <button className="soft" onClick={() => onContinue({})}>按序号编码并继续</button>
          <button className="primary" onClick={() => onContinue(serializedDrafts())}>使用当前定义并继续</button>
        </footer>
      </section>
    </div>
  );
}

function CompatibilityDialog({
  incompatibleCount,
  compatibleCount,
  outputFormat,
  onBack,
  onContinue,
}: {
  incompatibleCount: number;
  compatibleCount: number;
  outputFormat: OutputFormat;
  onBack: () => void;
  onContinue: () => void;
}) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="批量兼容性确认">
      <section className="modal-card compatibility-modal">
        <h2>部分文件与当前格式不兼容</h2>
        <p>{outputFormat} 可转换 {compatibleCount} 个文件，另有 {incompatibleCount} 个文件将进入报告并标记为 skipped_incompatible。</p>
        <footer>
          <button className="soft" onClick={onBack}>返回修改格式</button>
          <button className="primary" onClick={onContinue}>只转换兼容文件</button>
        </footer>
      </section>
    </div>
  );
}

function PathButton({ label, value, onClick, disabled = false }: { label: string; value: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button className="path-picker" onClick={onClick} disabled={disabled}>
      <span>{label}</span>
      <strong>
        <Folder size={16} />
        {value}
      </strong>
    </button>
  );
}

function SettingsView({
  settings,
  conversionSettings,
  onChange,
  onConversionChange,
}: {
  settings: ThemeSettings;
  conversionSettings: ConversionSettings;
  onChange: (settings: ThemeSettings) => void;
  onConversionChange: (settings: ConversionSettings) => void;
}) {
  return (
    <section className="settings-view">
      <div className="settings-head">
        <h2>设置</h2>
      </div>
      <div className="settings-group">
        <label>外观模式</label>
        <div className="segment compact">
          <button className={settings.mode === "auto" ? "selected" : ""} onClick={() => onChange({ ...settings, mode: "auto" })}>
            <RotateCcw size={14} />
            自动
          </button>
          <button className={settings.mode === "light" ? "selected" : ""} onClick={() => onChange({ ...settings, mode: "light" })}>
            <Sun size={14} />
            浅色
          </button>
          <button className={settings.mode === "dark" ? "selected" : ""} onClick={() => onChange({ ...settings, mode: "dark" })}>
            <Moon size={14} />
            深色
          </button>
        </div>
      </div>
      <div className="settings-group">
        <label>转换参数</label>
        <div className="settings-grid">
          <StepperField
            label="同时处理 WSI"
            value={conversionSettings.parallel_wsi}
            min={1}
            max={8}
            onChange={(value) => onConversionChange({ ...conversionSettings, parallel_wsi: value })}
          />
          <StepperField
            label="JPEG quality"
            value={conversionSettings.jpeg_quality}
            min={1}
            max={100}
            onChange={(value) => onConversionChange({ ...conversionSettings, jpeg_quality: value })}
          />
          <StepperField
            label="内存预算 MB"
            value={conversionSettings.memory_budget_mb}
            min={1024}
            max={65536}
            step={512}
            onChange={(value) => onConversionChange({ ...conversionSettings, memory_budget_mb: value })}
          />
        </div>
        <p className="settings-note">并行处理会同时转换多个 WSI，默认 1 最稳妥；内存预算会在并发 WSI 之间分配，不是操作系统硬限制。</p>
      </div>
      <div className="settings-actions">
        <button className="soft" onClick={() => onChange(defaultThemeSettings)}>
          重置外观
        </button>
        <button className="soft" onClick={() => onConversionChange(defaultConversionSettings)}>
          重置转换参数
        </button>
      </div>
    </section>
  );
}

export function StepperField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const inputId = useId();
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  function parseDraft(raw: string): number | null {
    if (!raw.trim()) return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? Math.round(parsed) : null;
  }

  function commitDraft(raw: string) {
    const parsed = parseDraft(raw);
    if (parsed === null) {
      setDraft(String(value));
      return;
    }
    const committed = Math.min(max, Math.max(min, parsed));
    setDraft(String(committed));
    onChange(committed);
  }

  function nudge(direction: -1 | 1) {
    const parsed = parseDraft(draft) ?? value;
    commitDraft(String(parsed + direction * step));
  }

  return (
    <div className="text-field">
      <label htmlFor={inputId}>{label}</label>
      <span className="stepper-field">
        <input
          id={inputId}
          type="number"
          min={min}
          max={max}
          step={step}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={(event) => commitDraft(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") event.currentTarget.blur();
            if (event.key === "ArrowUp") {
              event.preventDefault();
              nudge(1);
            }
            if (event.key === "ArrowDown") {
              event.preventDefault();
              nudge(-1);
            }
          }}
        />
        <span className="stepper-actions">
          <button type="button" aria-label={`减小${label}`} disabled={Number(draft) <= min} onClick={() => nudge(-1)}>
            <Minus size={14} />
          </button>
          <button type="button" aria-label={`增大${label}`} disabled={Number(draft) >= max} onClick={() => nudge(1)}>
            <Plus size={14} />
          </button>
        </span>
      </span>
    </div>
  );
}
