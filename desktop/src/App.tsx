import {
  ChevronRight,
  Folder,
  Moon,
  Play,
  RotateCcw,
  Settings,
  Square,
  Sun,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  cancelConversion,
  chooseDirectory,
  onConversionEvent,
  openFilesystemPath,
  startConversion,
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
  ConversionSettings,
  ConversionEvent,
  OutputFormat,
  ProgressState,
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
  statusText: "Idle",
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
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("generic_tiff");
  const [recursive, setRecursive] = useState(true);
  const [progress, setProgress] = useState<ProgressState>(initialProgress);
  const [phaseStates, setPhaseStates] = useState<PhaseDisplayState[]>(() => initialPhaseStates());
  const [logs, setLogs] = useState<string[]>([]);
  const [themeSettings, setThemeSettings] = useState<ThemeSettings>(() => loadThemeSettings());
  const [conversionSettings, setConversionSettings] = useState<ConversionSettings>(() => loadConversionSettings());
  const [taskSettings, setTaskSettings] = useState<ConversionSettings | null>(null);
  const [actualTheme, setActualTheme] = useState<ActualTheme>(() => resolveTheme(loadThemeSettings()).theme);
  const themeResolution = useMemo(() => resolveTheme(themeSettings), [themeSettings]);
  const fileProgressByPath = useRef<Map<string, number>>(new Map());
  const filePhaseByPath = useRef<Map<string, number>>(new Map());
  const completedFiles = useRef<Set<string>>(new Set());
  const batchTotal = useRef(0);
  const completedCount = useRef(0);
  const lastBatchPercent = useRef(0);
  const batchFinished = useRef(false);
  const batchCancelled = useRef(false);

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
    let unsubscribe: (() => void) | undefined;
    void onConversionEvent(handleConversionEvent).then((cleanup) => {
      unsubscribe = cleanup;
    });
    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

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
    if (event.type === "ready") {
      appendLog(event.banner || "Python worker ready");
      return;
    }
    if (event.type === "started") {
      setProgress((state) => ({ ...state, running: true, statusText: "Running" }));
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
          statusText: "Running",
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
      }
      completedCount.current = Math.max(completedFiles.current.size, batch.results.length);
      if (!batch.cancelled) lastBatchPercent.current = 100;
      const nextPhaseStates = updatePhaseStates(batch.total_files);
      setProgress((state) => ({
        ...state,
        running: false,
        statusText: batch.cancelled ? "Cancelled" : failed ? "Error" : "Ready",
        currentPhase: batch.cancelled ? "已取消" : failed ? "转换失败" : summarizeBatchPhase(nextPhaseStates),
        batchDone: completedCount.current,
        batchTotal: batch.total_files,
        batchPercent: batch.cancelled ? state.batchPercent : 100,
        etaText: batch.cancelled || failed ? "—" : "已完成",
        reportPath: batch.report_path || state.reportPath,
        backend: batch.results.length ? batch.results[batch.results.length - 1].backend : state.backend,
      }));
      return;
    }
    if (event.type === "error") {
      batchFinished.current = true;
      batchCancelled.current = false;
      setProgress((state) => ({ ...state, running: false, statusText: "Error", currentPhase: "完成", etaText: "—" }));
      appendLog(event.traceback || event.message);
      return;
    }
    if (event.type === "worker_terminated") {
      appendLog(`Worker terminated: ${event.code ?? "-"} ${event.signal ?? ""}`);
    }
  }

  async function pickInput() {
    const path = await chooseDirectory();
    if (!path) return;
    setInputDir(path);
    setProgress((state) => ({ ...state, currentFile: basename(path), statusText: "Idle" }));
  }

  async function pickOutput() {
    const path = await chooseDirectory();
    if (!path) return;
    setOutputDir(path);
    setProgress((state) => ({ ...state, outputDir: path }));
  }

  async function runConversion() {
    if (!inputDir || !outputDir || progress.running) return;
    const jobId = `job-${Date.now()}`;
    const settingsSnapshot = { ...conversionSettings };
    resetBatchAggregation();
    setLogs([]);
    setTaskSettings(settingsSnapshot);
    setProgress({
      ...initialProgress,
      running: true,
      statusText: "Starting",
      currentFile: basename(inputDir),
      outputDir,
      startedAt: Date.now(),
      backend: outputFormat === "svs" ? "svs-streaming-direct" : "tifffile-streaming",
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
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setProgress((state) => ({ ...state, running: false, statusText: "Error" }));
      appendLog(message);
    }
  }

  async function cancelCurrent() {
    await cancelConversion();
    appendLog("已请求取消，等待当前写入步骤安全结束");
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

  const canStart = Boolean(inputDir && outputDir && !progress.running);

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
          <span>v0.3.0</span>
          <span>{actualTheme === "light" ? "Light" : "Dark"}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{inputDir ? basename(inputDir) : "未选择任务"}</h1>
            <p>{inputDir ? "确认输出目录和格式后即可开始转换。" : "选择一个输入目录后开始转换。"}</p>
          </div>
          <div className={`status-chip ${progress.statusText.toLowerCase()}`}>{progress.statusText}</div>
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
            reportPath={progress.reportPath}
            onInput={pickInput}
            onOutput={pickOutput}
            onFormat={setOutputFormat}
            onRecursive={setRecursive}
            onStart={runConversion}
            onCancel={cancelCurrent}
            onOpenOutput={() => openPathWithFeedback(outputDir, "打开输出目录")}
            onOpenReport={() => openPathWithFeedback(progress.reportPath, "打开转换报告")}
          />
        )}
      </main>
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
          <dd>{progress.running ? progress.etaText : progress.statusText === "Ready" ? "已完成" : "—"}</dd>
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

function TaskComposer({
  inputDir,
  outputDir,
  outputFormat,
  conversionSettings,
  recursive,
  canStart,
  running,
  reportPath,
  onInput,
  onOutput,
  onFormat,
  onRecursive,
  onStart,
  onCancel,
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
  reportPath: string;
  onInput: () => void;
  onOutput: () => void;
  onFormat: (format: OutputFormat) => void;
  onRecursive: (value: boolean) => void;
  onStart: () => void;
  onCancel: () => void;
  onOpenOutput: () => void;
  onOpenReport: () => void;
}) {
  return (
    <section className="composer">
      <div className="path-row">
        <PathButton label="输入目录" value={inputDir || "选择包含 IBL / KFB / IMAGE / SVS / TIFF 的文件夹"} onClick={onInput} />
        <PathButton label="输出目录" value={outputDir || "选择转换结果保存位置"} onClick={onOutput} />
      </div>
      <div className="format-note" aria-label="支持的输入格式">
        <span className="format-note-label">输入格式</span>
        <span className="format-note-values">
          .ibl · .kfb · <strong>.image</strong> · .svs · .tif/.tiff
        </span>
        <span className="format-note-auto">自动识别</span>
      </div>
      <div className="action-row">
        <div className="segment">
          <button className={outputFormat === "generic_tiff" ? "selected" : ""} onClick={() => onFormat("generic_tiff")}>
            Generic TIFF
          </button>
          <button className={outputFormat === "svs" ? "selected" : ""} onClick={() => onFormat("svs")}>
            SVS
          </button>
        </div>
        <label className="check-row">
          <input type="checkbox" checked={recursive} onChange={(event) => onRecursive(event.target.checked)} />
          包含子文件夹
        </label>
        <button className="primary" disabled={!canStart} onClick={onStart}>
          <Play size={16} />
          开始转换
        </button>
        <button className="soft" disabled={!running} onClick={onCancel}>
          <Square size={14} />
          取消
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

function PathButton({ label, value, onClick }: { label: string; value: string; onClick: () => void }) {
  return (
    <button className="path-picker" onClick={onClick}>
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
          <NumberField
            label="同时处理 WSI"
            value={conversionSettings.parallel_wsi}
            min={1}
            max={8}
            onChange={(value) => onConversionChange({ ...conversionSettings, parallel_wsi: value })}
          />
          <NumberField
            label="JPEG quality"
            value={conversionSettings.jpeg_quality}
            min={1}
            max={100}
            onChange={(value) => onConversionChange({ ...conversionSettings, jpeg_quality: value })}
          />
          <NumberField
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

function NumberField({
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

  return (
    <label className="text-field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={(event) => commitDraft(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
      />
    </label>
  );
}
