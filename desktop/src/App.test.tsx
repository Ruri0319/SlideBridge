import { useState } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App, {
  ChannelDialog,
  InspectionSummary,
  StepperField,
  TaskComposer,
  TaskStatusBanner,
} from "./App";
import type { BatchInspection, ConversionEvent, InputInspection, ProgressState } from "./types";

const apiMocks = vi.hoisted(() => ({
  cancelConversion: vi.fn(),
  chooseDirectory: vi.fn(),
  ensureWorker: vi.fn(),
  getApplicationVersion: vi.fn(),
  onConversionEvent: vi.fn(),
  openFilesystemPath: vi.fn(),
  startConversion: vi.fn(),
  startInspection: vi.fn(),
}));

vi.mock("./tauriApi", () => apiMocks);

let conversionEventHandler: ((event: ConversionEvent) => void) | null = null;

function inspectedFile(overrides: Partial<InputInspection> = {}): InputInspection {
  return {
    input_path: "/slides/sample.kfbf",
    file_size: 1024,
    file_mtime_ns: "123",
    input_format: "kfbf",
    source_modality: "fluorescence",
    source_container: "kfbf",
    source_version: "2.1",
    source_codec: "jpeg",
    source_bit_depth: 8,
    field_count: 1,
    channel_count: 1,
    z_count: 1,
    t_count: 1,
    channel_definitions: [{
      index: 0,
      name: "DAPI",
      fluor: "DAPI",
      color: [0, 0, 255],
      excitation_nm: 358,
      emission_nm: 461,
      exposure: 10,
      identity_source: "source_metadata",
    }],
    allowed_output_formats: ["ome_tiff", "fluorescence_svs", "afi"],
    incompatible_reasons: {},
    error: null,
    ...overrides,
  };
}

function inspection(files: InputInspection[]): BatchInspection {
  return { input_dir: "/slides", recursive: true, files };
}

function progress(status: ProgressState["status"]): ProgressState {
  return {
    running: status === "starting" || status === "running",
    status,
    currentFile: "sample.kfbf",
    currentPhase: status === "error" ? "写出失败" : "生成金字塔",
    stagePercent: 40,
    filePercent: 35,
    batchPercent: 35,
    batchDone: 0,
    batchTotal: 1,
    etaText: "12 秒",
    backend: "tifffile-ome",
    memoryMb: 256,
    cpuPercent: 20,
    reportPath: "/output/report.csv",
    outputDir: "/output",
    startedAt: Date.now() - 2_000,
    finishedAt: status === "success" ? Date.now() : null,
    successCount: status === "success" ? 1 : 0,
    failedCount: 0,
    skippedCount: 0,
  };
}

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  conversionEventHandler = null;
  apiMocks.getApplicationVersion.mockResolvedValue("0.4.5");
  apiMocks.ensureWorker.mockResolvedValue({ alive: true, ready: true, activity: "idle", job_id: null });
  apiMocks.onConversionEvent.mockImplementation(async (callback: (event: ConversionEvent) => void) => {
    conversionEventHandler = callback;
    return () => undefined;
  });
  apiMocks.startInspection.mockResolvedValue(undefined);
});

describe("ChannelDialog", () => {
  it("keeps custom selected, focuses the name, and can return to a preset", async () => {
    const user = userEvent.setup();
    render(<ChannelDialog files={[inspectedFile()]} onCancel={vi.fn()} onContinue={vi.fn()} />);

    const trigger = screen.getByLabelText("C1 荧光预设");
    expect(trigger.textContent).toContain("DAPI");
    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "自定义" }));

    const name = screen.getByLabelText("通道名称") as HTMLInputElement;
    await waitFor(() => expect(document.activeElement).toBe(name));
    expect(trigger.textContent).toContain("自定义");
    expect(name.value).toBe("DAPI");

    await user.clear(name);
    await user.type(name, "My channel");
    expect(trigger.textContent).toContain("自定义");

    await user.click(trigger);
    await user.click(await screen.findByRole("option", { name: "FITC" }));
    expect(name.value).toBe("FITC");
    expect((screen.getByLabelText("显示色") as HTMLInputElement).value).toBe("#00ff00");
  });
});

describe("StepperField", () => {
  function Harness() {
    const [value, setValue] = useState(2);
    return <StepperField label="并发" value={value} min={1} max={3} onChange={setValue} />;
  }

  it("uses the configured bounds for buttons and arrow keys", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = screen.getByLabelText("并发") as HTMLInputElement;

    await user.click(screen.getByLabelText("增大并发"));
    expect(input.value).toBe("3");
    expect((screen.getByLabelText("增大并发") as HTMLButtonElement).disabled).toBe(true);

    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(input.value).toBe("2");
    fireEvent.change(input, { target: { value: "0" } });
    fireEvent.blur(input);
    expect(input.value).toBe("1");
  });
});

describe("inspection and task feedback", () => {
  it("summarizes detected containers instead of highlighting a fixed extension", () => {
    const files = [
      inspectedFile(),
      inspectedFile({
        input_path: "/slides/he.ome.tif",
        input_format: "tiff",
        source_container: "ome_tiff",
        source_modality: "brightfield",
        channel_count: 3,
        channel_definitions: [],
      }),
    ];
    render(
      <InspectionSummary
        inspection={inspection(files)}
        status="ready"
        message="检测完成"
        total={2}
        done={2}
        candidateFormatCounts={{}}
        partialFiles={files}
      />,
    );
    expect(screen.getByText("KFBF 1")).toBeTruthy();
    expect(screen.getByText("OME-TIFF 1")).toBeTruthy();
    expect(screen.getByText("明场 1 · 荧光 1")).toBeTruthy();
  });

  it("shows persistent starting, running, and success states", () => {
    const { rerender } = render(
      <TaskStatusBanner progress={progress("starting")} inspectionMessage="" onOpenOutput={vi.fn()} onOpenReport={vi.fn()} />,
    );
    expect(screen.getByText("正在启动转换")).toBeTruthy();

    rerender(<TaskStatusBanner progress={progress("running")} inspectionMessage="" onOpenOutput={vi.fn()} onOpenReport={vi.fn()} />);
    expect(screen.getByText("正在转换 sample.kfbf")).toBeTruthy();

    rerender(<TaskStatusBanner progress={progress("success")} inspectionMessage="" onOpenOutput={vi.fn()} onOpenReport={vi.fn()} />);
    expect(screen.getByText("转换完成")).toBeTruthy();
    expect(screen.getByText(/成功 1/)).toBeTruthy();
  });

  it("locks output formats as soon as conversion is starting", () => {
    const file = inspectedFile();
    render(
      <TaskComposer
        inputDir="/slides"
        outputDir="/output"
        outputFormat="ome_tiff"
        conversionSettings={{ parallel_wsi: 1, jpeg_quality: 90, memory_budget_mb: 4096 }}
        recursive
        canStart={false}
        running
        taskStatus="starting"
        reportPath=""
        inspection={inspection([file])}
        inspectionStatus="ready"
        inspectionMessage="检测完成"
        inspectionTotal={1}
        inspectionDone={1}
        candidateFormatCounts={{}}
        partialInspectionFiles={[file]}
        formatCounts={{ ome_tiff: 1, svs: 0, fluorescence_svs: 1, afi: 1 }}
        onInput={vi.fn()}
        onOutput={vi.fn()}
        onFormat={vi.fn()}
        onRecursive={vi.fn()}
        onStart={vi.fn()}
        onCancel={vi.fn()}
        onReset={vi.fn()}
        onEditChannels={vi.fn()}
        onOpenOutput={vi.fn()}
        onOpenReport={vi.fn()}
      />,
    );
    for (const label of ["Pyramidal OME-TIFF", "明场 SVS", "荧光 SVS", "AFI"]) {
      expect((screen.getByRole("button", { name: new RegExp(label) }) as HTMLButtonElement).disabled).toBe(true);
    }
  });
});

describe("App reset", () => {
  it("starts a fresh inspection for the current input directory", async () => {
    const user = userEvent.setup();
    apiMocks.chooseDirectory.mockResolvedValue("/slides");
    render(<App />);

    await user.click(screen.getByRole("button", { name: /输入目录/ }));
    await waitFor(() => expect(apiMocks.startInspection).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: /重置任务/ }));
    await waitFor(() => expect(apiMocks.startInspection).toHaveBeenCalledTimes(2));

    const firstJob = apiMocks.startInspection.mock.calls[0][0].job_id;
    const secondJob = apiMocks.startInspection.mock.calls[1][0].job_id;
    expect(secondJob).not.toBe(firstJob);
  });

  it("ends the task banner when inspection startup fails", async () => {
    const user = userEvent.setup();
    apiMocks.chooseDirectory.mockResolvedValue("/slides");
    apiMocks.startInspection.mockRejectedValueOnce(new Error("worker unavailable"));
    render(<App />);

    await user.click(screen.getByRole("button", { name: /输入目录/ }));

    await waitFor(() => expect(screen.getAllByText("预检失败").length).toBeGreaterThan(0));
    expect(screen.getByText("检测失败")).toBeTruthy();
    expect(screen.getAllByText("worker unavailable").length).toBeGreaterThan(0);
  });

  it("marks inspection failed when the worker terminates", async () => {
    const user = userEvent.setup();
    apiMocks.chooseDirectory.mockResolvedValue("/slides");
    render(<App />);

    await user.click(screen.getByRole("button", { name: /输入目录/ }));
    await waitFor(() => expect(apiMocks.startInspection).toHaveBeenCalledTimes(1));
    const jobId = apiMocks.startInspection.mock.calls[0][0].job_id;

    act(() => {
      conversionEventHandler?.({
        type: "worker_terminated",
        code: 1,
        signal: null,
        busy: true,
        activity: "inspecting",
        job_id: jobId,
      });
    });

    expect(screen.getAllByText("预检失败").length).toBeGreaterThan(0);
    expect(screen.getByText("检测失败")).toBeTruthy();
    expect(screen.getAllByText("转换引擎异常退出，请重新预检").length).toBeGreaterThan(0);
  });
});

describe("App lifecycle", () => {
  it("unsubscribes when listener registration resolves after unmount", async () => {
    const unlisten = vi.fn();
    let resolveListener: ((value: () => void) => void) | undefined;
    apiMocks.onConversionEvent.mockReturnValueOnce(
      new Promise<() => void>((resolve) => {
        resolveListener = resolve;
      }),
    );

    const view = render(<App />);
    view.unmount();
    await act(async () => {
      resolveListener?.(unlisten);
      await Promise.resolve();
    });

    expect(unlisten).toHaveBeenCalledTimes(1);
    expect(apiMocks.ensureWorker).not.toHaveBeenCalled();
  });
});
