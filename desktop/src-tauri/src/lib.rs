use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
enum OutputFormat {
    GenericTiff,
    Svs,
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
}

#[derive(Debug, Serialize)]
struct WorkerStatus {
    running: bool,
    job_id: Option<String>,
}

#[derive(Debug)]
struct RunningWorker {
    job_id: String,
    child: CommandChild,
}

#[derive(Default)]
struct ConversionManager {
    worker: Mutex<Option<RunningWorker>>,
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

#[tauri::command]
fn worker_status(manager: State<'_, ConversionManager>) -> WorkerStatus {
    let guard = manager.worker.lock().expect("worker mutex poisoned");
    WorkerStatus {
        running: guard.is_some(),
        job_id: guard.as_ref().map(|worker| worker.job_id.clone()),
    }
}

#[tauri::command]
fn cancel_conversion(manager: State<'_, ConversionManager>) -> Result<(), CommandError> {
    let mut guard = manager.worker.lock().expect("worker mutex poisoned");
    let Some(worker) = guard.as_mut() else {
        return Err(CommandError::NotRunning);
    };
    let payload = json!({
        "type": "cancel",
        "job_id": worker.job_id,
    });
    worker
        .child
        .write(format!("{}\n", payload).as_bytes())
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    Ok(())
}

#[tauri::command]
async fn start_conversion(
    app: AppHandle,
    manager: State<'_, ConversionManager>,
    request: ConversionRequest,
) -> Result<(), CommandError> {
    {
        let guard = manager.worker.lock().expect("worker mutex poisoned");
        if guard.is_some() {
            return Err(CommandError::AlreadyRunning);
        }
    }

    let command = app
        .shell()
        .sidecar("ibl2svs-worker")
        .map_err(|error| CommandError::Worker(error.to_string()))?;
    let (mut receiver, mut child) = command
        .spawn()
        .map_err(|error| CommandError::Worker(error.to_string()))?;

    let output_format = match request.output_format {
        OutputFormat::GenericTiff => "generic_tiff",
        OutputFormat::Svs => "svs",
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
            "parallel_wsi": request.parallel_wsi.unwrap_or(1)
        }
    });
    child
        .write(format!("{}\n", start_payload).as_bytes())
        .map_err(|error| CommandError::Worker(error.to_string()))?;

    {
        let mut guard = manager.worker.lock().expect("worker mutex poisoned");
        *guard = Some(RunningWorker {
            job_id: request.job_id.clone(),
            child,
        });
    }

    let app_for_events = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let parsed = serde_json::from_str::<serde_json::Value>(&line)
                        .unwrap_or_else(|_| json!({"type": "error", "message": line}));
                    let _ = app_for_events.emit("conversion:event", parsed.clone());
                    let event_type = parsed
                        .get("type")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default();
                    if matches!(event_type, "done" | "error") {
                        if let Some(state) = app_for_events.try_state::<ConversionManager>() {
                            let mut guard = state.worker.lock().expect("worker mutex poisoned");
                            if let Some(worker) = guard.take() {
                                let _ = worker.child.kill();
                            }
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
                    let _ = app_for_events.emit(
                        "conversion:event",
                        json!({"type": "worker_terminated", "code": payload.code, "signal": payload.signal}),
                    );
                    if let Some(state) = app_for_events.try_state::<ConversionManager>() {
                        let mut guard = state.worker.lock().expect("worker mutex poisoned");
                        *guard = None;
                    }
                }
                _ => {}
            }
        }
    });

    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(ConversionManager::default())
        .invoke_handler(tauri::generate_handler![
            start_conversion,
            cancel_conversion,
            worker_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running IBL2SVS desktop app");
}
