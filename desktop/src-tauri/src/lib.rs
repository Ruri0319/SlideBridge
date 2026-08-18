use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
enum OutputFormat {
    OmeTiff,
    Svs,
    FluorescenceSvs,
    Afi,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ConversionRequest {
    job_id: String,
    input_dir: String,
    output_dir: String,
    output_format: OutputFormat,
    recursive: bool,
    memory_budget_mb: Option<u32>,
    tile_size: Option<u32>,
    jpeg_quality: Option<u32>,
    parallel_wsi: Option<u32>,
    selected_input_paths: Option<Vec<String>>,
    convert_compatible_only: Option<bool>,
    channel_overrides: Option<Value>,
    input_signatures: Option<Value>,
    preflight_files: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct InspectionRequest {
    job_id: String,
    input_dir: String,
    recursive: bool,
}

#[derive(Debug, Serialize)]
struct WorkerStatus {
    alive: bool,
    ready: bool,
    activity: String,
    job_id: Option<String>,
}

#[derive(Debug)]
struct RunningWorker {
    child: CommandChild,
    generation: u64,
    ready: bool,
    activity: WorkerActivity,
    job_id: Option<String>,
    started_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerActivity {
    Initializing,
    Idle,
    Inspecting,
    Conversion,
}

#[derive(Default)]
struct ConversionManager {
    state: Mutex<WorkerManagerState>,
}

#[derive(Default)]
struct WorkerManagerState {
    worker: Option<RunningWorker>,
    next_generation: u64,
}

#[derive(Debug, thiserror::Error)]
enum CommandError {
    #[error("conversion already running")]
    AlreadyRunning,
    #[error("no conversion is running")]
    NotRunning,
    #[error("worker error: {0}")]
    Worker(String),
}

impl Serialize for CommandError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

fn encode_worker_message(payload: &Value) -> Result<String, CommandError> {
    let message =
        serde_json::to_string(payload).map_err(|error| CommandError::Worker(error.to_string()))?;
    Ok(format!("{message}\n"))
}

fn activity_name(activity: WorkerActivity) -> &'static str {
    match activity {
        WorkerActivity::Initializing => "initializing",
        WorkerActivity::Idle => "idle",
        WorkerActivity::Inspecting => "inspecting",
        WorkerActivity::Conversion => "converting",
    }
}

fn worker_terminated_event(
    code: Option<i32>,
    signal: Option<i32>,
    busy: bool,
    activity: WorkerActivity,
    job_id: Option<String>,
) -> Value {
    json!({
        "type": "worker_terminated",
        "code": code,
        "signal": signal,
        "busy": busy,
        "activity": activity_name(activity),
        "job_id": job_id
    })
}

fn current_worker_status(manager: &ConversionManager) -> WorkerStatus {
    let guard = manager.state.lock().expect("worker mutex poisoned");
    match guard.worker.as_ref() {
        Some(worker) => WorkerStatus {
            alive: true,
            ready: worker.ready,
            activity: activity_name(worker.activity).to_string(),
            job_id: worker.job_id.clone(),
        },
        None => WorkerStatus {
            alive: false,
            ready: false,
            activity: "unavailable".to_string(),
            job_id: None,
        },
    }
}

fn ensure_worker_process(app: &AppHandle, manager: &ConversionManager) -> Result<(), CommandError> {
    let mut guard = manager.state.lock().expect("worker mutex poisoned");
    if guard.worker.is_some() {
        return Ok(());
    }

    let command = app
        .shell()
        .sidecar("slidebridge-worker")
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    let (mut receiver, child) = command
        .spawn()
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    guard.next_generation += 1;
    let generation = guard.next_generation;
    guard.worker = Some(RunningWorker {
        child,
        generation,
        ready: false,
        activity: WorkerActivity::Initializing,
        job_id: None,
        started_at: Instant::now(),
    });
    drop(guard);

    let app_for_events = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    for line in String::from_utf8_lossy(&bytes).lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let parsed = serde_json::from_str::<Value>(line)
                            .unwrap_or_else(|_| json!({"type": "error", "message": line}));
                        let event_type = parsed
                            .get("type")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let event_job_id = parsed.get("job_id").and_then(|value| value.as_str());
                        let mut startup_ms = None;
                        if let Some(state) = app_for_events.try_state::<ConversionManager>() {
                            let mut state_guard =
                                state.state.lock().expect("worker mutex poisoned");
                            if let Some(worker) = state_guard
                                .worker
                                .as_mut()
                                .filter(|worker| worker.generation == generation)
                            {
                                if event_type == "ready" {
                                    worker.ready = true;
                                    if worker.activity == WorkerActivity::Initializing {
                                        worker.activity = WorkerActivity::Idle;
                                    }
                                    startup_ms =
                                        Some(worker.started_at.elapsed().as_secs_f64() * 1000.0);
                                }
                                let is_current_job = event_job_id
                                    .map(|job_id| worker.job_id.as_deref() == Some(job_id))
                                    .unwrap_or(false);
                                if is_current_job
                                    && matches!(
                                        event_type,
                                        "inspection_done" | "inspection_error" | "done" | "error"
                                    )
                                {
                                    worker.activity = WorkerActivity::Idle;
                                    worker.job_id = None;
                                }
                            }
                        }
                        let _ = app_for_events.emit("conversion:event", parsed);
                        if let Some(elapsed) = startup_ms {
                            let _ = app_for_events.emit(
                                "conversion:event",
                                json!({
                                    "type": "log",
                                    "message": format!("worker_startup_ms={elapsed:.1}")
                                }),
                            );
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let message = String::from_utf8_lossy(&bytes).trim().to_string();
                    if !message.is_empty() {
                        let _ = app_for_events.emit(
                            "conversion:event",
                            json!({"type": "log", "message": message}),
                        );
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let mut was_busy = false;
                    let mut terminated_activity = WorkerActivity::Idle;
                    let mut terminated_job_id = None;
                    if let Some(state) = app_for_events.try_state::<ConversionManager>() {
                        let mut state_guard = state.state.lock().expect("worker mutex poisoned");
                        if let Some(worker) = state_guard
                            .worker
                            .as_ref()
                            .filter(|worker| worker.generation == generation)
                        {
                            terminated_activity = worker.activity;
                            terminated_job_id = worker.job_id.clone();
                            was_busy = matches!(
                                worker.activity,
                                WorkerActivity::Inspecting | WorkerActivity::Conversion
                            );
                            state_guard.worker = None;
                        }
                    }
                    let _ = app_for_events.emit(
                        "conversion:event",
                        worker_terminated_event(
                            payload.code,
                            payload.signal,
                            was_busy,
                            terminated_activity,
                            terminated_job_id,
                        ),
                    );
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn worker_status(manager: State<'_, ConversionManager>) -> WorkerStatus {
    current_worker_status(manager.inner())
}

#[tauri::command]
fn ensure_worker(
    app: AppHandle,
    manager: State<'_, ConversionManager>,
) -> Result<WorkerStatus, CommandError> {
    ensure_worker_process(&app, manager.inner())?;
    Ok(current_worker_status(manager.inner()))
}

#[tauri::command]
fn cancel_conversion(manager: State<'_, ConversionManager>) -> Result<(), CommandError> {
    let mut guard = manager.state.lock().expect("worker mutex poisoned");
    let Some(worker) = guard.worker.as_mut() else {
        return Err(CommandError::NotRunning);
    };
    if worker.activity != WorkerActivity::Conversion {
        return Err(CommandError::NotRunning);
    }
    let payload = json!({
        "type": "cancel",
        "job_id": worker.job_id,
    });
    let message = encode_worker_message(&payload)?;
    worker
        .child
        .write(message.as_bytes())
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    Ok(())
}

#[tauri::command]
async fn start_conversion(
    app: AppHandle,
    manager: State<'_, ConversionManager>,
    request: ConversionRequest,
) -> Result<(), CommandError> {
    ensure_worker_process(&app, manager.inner())?;

    let output_format = match request.output_format {
        OutputFormat::OmeTiff => "ome_tiff",
        OutputFormat::Svs => "svs",
        OutputFormat::FluorescenceSvs => "fluorescence_svs",
        OutputFormat::Afi => "afi",
    };
    let start_payload = json!({
        "type": "start",
        "job_id": request.job_id,
        "payload": {
            "job_id": request.job_id,
            "input_dir": request.input_dir,
            "output_dir": request.output_dir,
            "output_format": output_format,
            "recursive": request.recursive,
            "memory_budget_mb": request.memory_budget_mb.unwrap_or(6144),
            "tile_size": request.tile_size.unwrap_or(256),
            "jpeg_quality": request.jpeg_quality.unwrap_or(90),
            "parallel_wsi": request.parallel_wsi.unwrap_or(1),
            "selected_input_paths": request.selected_input_paths,
            "convert_compatible_only": request.convert_compatible_only.unwrap_or(false),
            "channel_overrides": request.channel_overrides.unwrap_or_else(|| json!({})),
            "input_signatures": request.input_signatures.unwrap_or_else(|| json!({})),
            "preflight_files": request.preflight_files.unwrap_or_else(|| json!([]))
        }
    });
    let mut guard = manager.state.lock().expect("worker mutex poisoned");
    let worker = guard
        .worker
        .as_mut()
        .ok_or_else(|| CommandError::Worker("worker unavailable".into()))?;
    if matches!(
        worker.activity,
        WorkerActivity::Inspecting | WorkerActivity::Conversion
    ) {
        return Err(CommandError::AlreadyRunning);
    }
    worker
        .child
        .write(encode_worker_message(&start_payload)?.as_bytes())
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    worker.activity = WorkerActivity::Conversion;
    worker.job_id = Some(request.job_id);

    Ok(())
}

#[tauri::command]
async fn start_inspection(
    app: AppHandle,
    manager: State<'_, ConversionManager>,
    request: InspectionRequest,
) -> Result<(), CommandError> {
    ensure_worker_process(&app, manager.inner())?;
    let payload = json!({
        "type": "inspect",
        "job_id": request.job_id,
        "payload": {
            "input_dir": request.input_dir,
            "recursive": request.recursive
        }
    });
    let mut guard = manager.state.lock().expect("worker mutex poisoned");
    let worker = guard
        .worker
        .as_mut()
        .ok_or_else(|| CommandError::Worker("worker unavailable".into()))?;
    if worker.activity == WorkerActivity::Conversion {
        return Err(CommandError::AlreadyRunning);
    }
    worker
        .child
        .write(encode_worker_message(&payload)?.as_bytes())
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    worker.activity = WorkerActivity::Inspecting;
    worker.job_id = Some(request.job_id);
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(ConversionManager::default())
        .invoke_handler(tauri::generate_handler![
            ensure_worker,
            start_conversion,
            start_inspection,
            cancel_conversion,
            worker_status
        ])
        .build(tauri::generate_context!())
        .expect("error while building SlideBridge desktop app")
        .run(|app, event| {
            if matches!(event, RunEvent::Exit) {
                if let Some(manager) = app.try_state::<ConversionManager>() {
                    let mut guard = manager.state.lock().expect("worker mutex poisoned");
                    if let Some(worker) = guard.worker.take() {
                        let _ = worker.child.kill();
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_message_escapes_windows_paths() {
        let input_dir = r"C:\Users\Alice\Slides";
        let payload = json!({
            "type": "start",
            "payload": { "input_dir": input_dir },
        });

        let message = encode_worker_message(&payload).expect("serialize worker message");
        let parsed: Value = serde_json::from_str(&message).expect("parse worker message");

        assert!(message.contains(r#"C:\\Users\\Alice\\Slides"#));
        assert_eq!(parsed["payload"]["input_dir"], input_dir);
    }

    #[test]
    fn worker_termination_event_preserves_activity_and_job_id() {
        let event = worker_terminated_event(
            Some(1),
            None,
            true,
            WorkerActivity::Inspecting,
            Some("inspect-1".to_string()),
        );

        assert_eq!(event["type"], "worker_terminated");
        assert_eq!(event["activity"], "inspecting");
        assert_eq!(event["job_id"], "inspect-1");
        assert_eq!(event["busy"], true);
    }
}
