#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{Manager, State};

#[cfg(target_family = "unix")]
use std::os::unix::process::CommandExt;

const BACKEND_PORT: u16 = 8000;
const STARTUP_TIMEOUT_SEC: u64 = 45;

type SharedState = Mutex<DesktopState>;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct RuntimeInfo {
    api_base: String,
    desktop_pid: u32,
    backend_pid: Option<u32>,
    startup_state: String,
    last_start_error: Option<String>,
    ui_python_path: String,
    backend_python_path: String,
    backend_command_fingerprint: String,
    logs_dir: String,
    run_dir: String,
    backend_started_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SidecarMarker {
    pid: u32,
    cmd_fingerprint: String,
    started_at: String,
}

#[derive(Debug)]
struct DesktopState {
    root: PathBuf,
    logs_dir: PathBuf,
    run_dir: PathBuf,
    api_base: String,
    startup_state: String,
    last_start_error: Option<String>,
    ui_python_path: String,
    backend_python_path: String,
    backend_command_fingerprint: String,
    backend_started_at: Option<String>,
    backend_child: Option<Child>,
}

impl DesktopState {
    fn new(root: PathBuf) -> Result<Self, String> {
        let logs_dir = dirs::home_dir()
            .ok_or_else(|| "Unable to resolve user home directory".to_string())?
            .join("Library")
            .join("Logs")
            .join("PaperEval");
        let run_dir = root.join(".run");
        fs::create_dir_all(&logs_dir).map_err(|err| format!("Failed to create logs dir: {err}"))?;
        fs::create_dir_all(&run_dir).map_err(|err| format!("Failed to create run dir: {err}"))?;

        let backend_python_path = resolve_backend_python(&root).display().to_string();
        let backend_command_fingerprint = fingerprint(&format!(
            "{} scripts/run_app.py --api-only --force --backend-port {}",
            backend_python_path, BACKEND_PORT
        ));

        Ok(Self {
            root,
            logs_dir,
            run_dir,
            api_base: format!("http://127.0.0.1:{BACKEND_PORT}/api"),
            startup_state: "booting".to_string(),
            last_start_error: None,
            ui_python_path: std::env::var("PYTHON").unwrap_or_else(|_| "n/a".to_string()),
            backend_python_path,
            backend_command_fingerprint,
            backend_started_at: None,
            backend_child: None,
        })
    }

    fn marker_path(&self) -> PathBuf {
        self.run_dir.join("desktop_sidecar.json")
    }

    fn desktop_log_path(&self) -> PathBuf {
        self.logs_dir.join("desktop_shell.log")
    }

    fn backend_log_path(&self) -> PathBuf {
        self.logs_dir.join("backend_desktop.log")
    }

    fn runtime_snapshot_path(&self) -> PathBuf {
        self.logs_dir.join("desktop_runtime.json")
    }

    fn lifecycle_path(&self) -> PathBuf {
        self.run_dir.join("lifecycle_events.jsonl")
    }

    fn runtime_info(&self) -> RuntimeInfo {
        RuntimeInfo {
            api_base: self.api_base.clone(),
            desktop_pid: std::process::id(),
            backend_pid: self.backend_child.as_ref().map(|child| child.id()),
            startup_state: self.startup_state.clone(),
            last_start_error: self.last_start_error.clone(),
            ui_python_path: self.ui_python_path.clone(),
            backend_python_path: self.backend_python_path.clone(),
            backend_command_fingerprint: self.backend_command_fingerprint.clone(),
            logs_dir: self.logs_dir.display().to_string(),
            run_dir: self.run_dir.display().to_string(),
            backend_started_at: self.backend_started_at.clone(),
        }
    }

    fn append_log(&self, message: &str) {
        let ts = unix_ts();
        if let Ok(mut handle) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.desktop_log_path())
        {
            let _ = writeln!(handle, "[{ts}] {message}");
        }
    }

    fn log_lifecycle(&self, event: &str, details: serde_json::Value) {
        let payload = json!({
            "event": event,
            "timestamp": unix_ts(),
            "details": details,
        });
        if let Ok(mut handle) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.lifecycle_path())
        {
            let _ = writeln!(handle, "{}", payload);
        }
    }

    fn write_runtime_snapshot(&self) {
        let payload = serde_json::to_vec_pretty(&self.runtime_info()).unwrap_or_else(|_| b"{}".to_vec());
        let _ = fs::write(self.runtime_snapshot_path(), payload);
    }

    fn cleanup_stale_marker(&self) {
        let marker_path = self.marker_path();
        if !marker_path.exists() {
            return;
        }

        let marker = fs::read_to_string(&marker_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<SidecarMarker>(&raw).ok());

        let Some(marker) = marker else {
            let _ = fs::remove_file(marker_path);
            return;
        };

        if !pid_alive(marker.pid) {
            let _ = fs::remove_file(self.marker_path());
            return;
        }

        if marker.cmd_fingerprint != self.backend_command_fingerprint {
            self.append_log("Stale marker skipped: command fingerprint mismatch.");
            let _ = fs::remove_file(self.marker_path());
            return;
        }

        self.append_log(&format!("Killing stale backend process group for pid {}", marker.pid));
        terminate_process_group(marker.pid as i32);
        let _ = fs::remove_file(self.marker_path());
    }

    fn write_marker(&self, pid: u32) {
        let payload = SidecarMarker {
            pid,
            cmd_fingerprint: self.backend_command_fingerprint.clone(),
            started_at: unix_ts(),
        };
        let serialized = serde_json::to_vec_pretty(&payload).unwrap_or_default();
        let _ = fs::write(self.marker_path(), serialized);
    }

    fn start_backend(&mut self) -> Result<(), String> {
        self.cleanup_stale_marker();

        if let Some(child) = self.backend_child.as_mut() {
            if child.try_wait().map_err(|err| err.to_string())?.is_none() {
                self.startup_state = "backend_ready".to_string();
                self.last_start_error = None;
                self.write_runtime_snapshot();
                return Ok(());
            }
        }

        self.startup_state = "backend_starting".to_string();
        self.last_start_error = None;
        self.write_runtime_snapshot();

        if wait_for_backend_ready(&self.api_base, 1) {
            self.append_log("Existing backend detected on port 8000; requesting shutdown before launch.");
            self.log_lifecycle(
                "desktop_backend_preexisting_detected",
                json!({"api_base": self.api_base.clone()}),
            );
            let _ = Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .and_then(|client| client.post(format!("{}/stop", self.api_base)).send());
            let _ = wait_for_backend_down(&self.api_base, 12);
        }

        let backend_log_path = self.backend_log_path();
        let mut backend_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&backend_log_path)
            .map_err(|err| format!("Failed to open backend log: {err}"))?;
        let _ = writeln!(backend_log, "\n---- {} backend start ----", unix_ts());

        let backend_python = PathBuf::from(&self.backend_python_path);
        let mut cmd = Command::new(backend_python);
        cmd.current_dir(&self.root)
            .arg("scripts/run_app.py")
            .arg("--api-only")
            .arg("--force")
            .arg("--backend-port")
            .arg(BACKEND_PORT.to_string())
            .env("PAPER_EVAL_ROOT", &self.root)
            .stdout(Stdio::from(
                backend_log
                    .try_clone()
                    .map_err(|err| format!("Failed to clone backend log handle: {err}"))?,
            ))
            .stderr(Stdio::from(backend_log));

        #[cfg(target_family = "unix")]
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }

        let child = cmd
            .spawn()
            .map_err(|err| format!("Failed to spawn backend process: {err}"))?;
        let pid = child.id();
        self.backend_child = Some(child);
        self.backend_started_at = Some(unix_ts());
        self.write_marker(pid);
        self.log_lifecycle(
            "desktop_backend_spawned",
            json!({"pid": pid, "api_base": self.api_base.clone()}),
        );

        if !wait_for_backend_ready(&self.api_base, STARTUP_TIMEOUT_SEC) {
            self.last_start_error = Some("Backend readiness timed out".to_string());
            self.startup_state = "degraded".to_string();
            self.append_log("Backend readiness timed out.");
            self.stop_backend();
            self.write_runtime_snapshot();
            return Err("Backend readiness timed out".to_string());
        }

        if let Some(child) = self.backend_child.as_mut() {
            if let Some(status) = child.try_wait().map_err(|err| err.to_string())? {
                self.last_start_error = Some(format!(
                    "Spawned backend process exited early with status {status}; an external backend likely occupied port 8000."
                ));
                self.startup_state = "degraded".to_string();
                self.append_log(
                    "Spawned backend exited early after readiness check; startup likely attached to pre-existing external backend.",
                );
                self.write_runtime_snapshot();
                return Err(
                    "Backend launch was hijacked by a pre-existing server on port 8000. Stop old backend and relaunch."
                        .to_string(),
                );
            }
        }

        self.startup_state = "backend_ready".to_string();
        self.last_start_error = None;
        self.append_log("Backend is ready.");
        self.log_lifecycle("desktop_backend_ready", json!({"pid": pid}));
        self.write_runtime_snapshot();

        let _ = trigger_processing_recover(&self.api_base);

        Ok(())
    }

    fn stop_backend(&mut self) {
        self.startup_state = "stopping".to_string();
        self.write_runtime_snapshot();

        let _ = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .and_then(|client| client.post(format!("{}/stop", self.api_base)).send());

        if let Some(mut child) = self.backend_child.take() {
            let pid = child.id() as i32;
            terminate_process_group(pid);
            let _ = child.wait();
            self.log_lifecycle("desktop_backend_stopped", json!({"pid": pid}));
        }

        self.backend_started_at = None;
        self.startup_state = "stopped".to_string();
        let _ = fs::remove_file(self.marker_path());
        self.write_runtime_snapshot();
    }
}

#[tauri::command]
fn runtime_info(state: State<'_, SharedState>) -> Result<RuntimeInfo, String> {
    let guard = state.lock().map_err(|_| "Failed to lock desktop state".to_string())?;
    Ok(guard.runtime_info())
}

#[tauri::command]
fn restart_backend(state: State<'_, SharedState>) -> Result<RuntimeInfo, String> {
    let mut guard = state.lock().map_err(|_| "Failed to lock desktop state".to_string())?;
    guard.stop_backend();
    guard.start_backend()?;
    Ok(guard.runtime_info())
}

#[tauri::command]
fn open_logs_folder(state: State<'_, SharedState>) -> Result<String, String> {
    let guard = state.lock().map_err(|_| "Failed to lock desktop state".to_string())?;
    Command::new("open")
        .arg(&guard.logs_dir)
        .spawn()
        .map_err(|err| format!("Failed to open logs folder: {err}"))?;
    Ok(guard.logs_dir.display().to_string())
}

#[tauri::command]
fn export_diagnostics(state: State<'_, SharedState>) -> Result<String, String> {
    let guard = state.lock().map_err(|_| "Failed to lock desktop state".to_string())?;

    let archive_path = guard
        .logs_dir
        .join(format!("papereval-diagnostics-{}.zip", unix_ts().replace(':', "-")));

    let candidates = [
        guard.desktop_log_path(),
        guard.backend_log_path(),
        guard.runtime_snapshot_path(),
        guard.lifecycle_path(),
        guard.run_dir.join("runtime_events.jsonl"),
    ];

    let existing: Vec<PathBuf> = candidates
        .iter()
        .filter(|path| path.exists())
        .cloned()
        .collect();

    if existing.is_empty() {
        return Err("No diagnostic files were found.".to_string());
    }

    let mut cmd = Command::new("zip");
    cmd.arg("-j").arg(&archive_path);
    for item in &existing {
        cmd.arg(item);
    }
    let output = cmd
        .output()
        .map_err(|err| format!("Failed to run zip command: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "zip command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(archive_path.display().to_string())
}

fn resolve_root() -> Result<PathBuf, String> {
    if let Ok(root) = std::env::var("PAPER_EVAL_ROOT") {
        let root_path = PathBuf::from(root);
        if is_project_root(&root_path) {
            return Ok(root_path);
        }
    }

    let cwd = std::env::current_dir().map_err(|err| format!("Failed to get cwd: {err}"))?;
    for candidate in cwd.ancestors() {
        if is_project_root(candidate) {
            return Ok(candidate.to_path_buf());
        }
    }

    let exe = std::env::current_exe().map_err(|err| format!("Failed to resolve current exe: {err}"))?;
    for candidate in exe.ancestors() {
        if is_project_root(candidate) {
            return Ok(candidate.to_path_buf());
        }
    }

    Err("Could not resolve PaperEval project root".to_string())
}

fn is_project_root(path: &Path) -> bool {
    path.join("backend").exists() && path.join("scripts").exists() && path.join("desktop_ui").exists()
}

fn resolve_backend_python(root: &Path) -> PathBuf {
    let preferred = root.join(".venv").join("bin").join("python");
    if preferred.exists() {
        return preferred;
    }
    PathBuf::from("/usr/bin/python3")
}

fn wait_for_backend_ready(api_base: &str, timeout_sec: u64) -> bool {
    let client = match Client::builder().timeout(Duration::from_secs(1)).build() {
        Ok(client) => client,
        Err(_) => return false,
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_sec);
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(format!("{api_base}/status")).send() {
            if resp.status().is_success() {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(350));
    }
    false
}

fn wait_for_backend_down(api_base: &str, timeout_sec: u64) -> bool {
    let client = match Client::builder().timeout(Duration::from_secs(1)).build() {
        Ok(client) => client,
        Err(_) => return false,
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_sec);
    while Instant::now() < deadline {
        match client.get(format!("{api_base}/status")).send() {
            Ok(resp) if resp.status().is_success() => {
                std::thread::sleep(Duration::from_millis(300));
            }
            _ => return true,
        }
    }
    false
}

fn trigger_processing_recover(api_base: &str) -> bool {
    let client = match Client::builder().timeout(Duration::from_secs(4)).build() {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client.post(format!("{api_base}/processing/recover")).send() {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

fn pid_alive(pid: u32) -> bool {
    #[cfg(target_family = "unix")]
    unsafe {
        libc::kill(pid as i32, 0) == 0
    }

    #[cfg(not(target_family = "unix"))]
    {
        let _ = pid;
        true
    }
}

fn terminate_process_group(pid: i32) {
    #[cfg(target_family = "unix")]
    unsafe {
        libc::killpg(pid, libc::SIGTERM);
        std::thread::sleep(Duration::from_millis(600));
        libc::killpg(pid, libc::SIGKILL);
    }

    #[cfg(not(target_family = "unix"))]
    {
        let _ = pid;
    }
}

fn fingerprint(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn unix_ts() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    now.to_string()
}

fn main() {
    let root = match resolve_root() {
        Ok(root) => root,
        Err(err) => {
            eprintln!("{err}");
            return;
        }
    };

    if let Err(err) = enforce_bundle_guard(&root) {
        eprintln!("{err}");
        return;
    }

    let state = match DesktopState::new(root) {
        Ok(state) => state,
        Err(err) => {
            eprintln!("{err}");
            return;
        }
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            let state = app.state::<SharedState>();
            if let Ok(mut guard) = state.lock() {
                guard.append_log("Secondary launch detected, recycling backend.");
                guard.log_lifecycle("desktop_shell_secondary_launch", json!({}));
                guard.stop_backend();
                if let Err(err) = guard.start_backend() {
                    guard.last_start_error = Some(err.clone());
                    guard.startup_state = "degraded".to_string();
                    guard.append_log(&format!("Backend restart on secondary launch failed: {err}"));
                }
                guard.write_runtime_snapshot();
            }

            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .manage(Mutex::new(state))
        .invoke_handler(tauri::generate_handler![
            runtime_info,
            restart_backend,
            open_logs_folder,
            export_diagnostics
        ])
        .setup(|app| {
            let state = app.state::<SharedState>();
            if let Ok(mut guard) = state.lock() {
                guard.append_log("Starting PaperEval Desktop shell.");
                guard.log_lifecycle(
                    "desktop_shell_startup",
                    json!({"api_base": guard.api_base.clone()}),
                );
                if let Err(err) = guard.start_backend() {
                    guard.last_start_error = Some(err.clone());
                    guard.startup_state = "degraded".to_string();
                    guard.append_log(&format!("Backend startup failed: {err}"));
                }
                guard.write_runtime_snapshot();
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let state = window.state::<SharedState>();
                let mut guard = match state.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                guard.append_log("Window close requested, stopping backend.");
                guard.stop_backend();
                window.app_handle().exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running PaperEval desktop shell");
}

fn enforce_bundle_guard(root: &Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|err| format!("Failed to resolve executable path: {err}"))?;
    let exe_str = exe.display().to_string();
    if exe_str.contains("/dist/") || exe_str.contains("/build/") {
        let canonical = root.join("PaperEval.app");
        show_alert(
            "PaperEval Desktop",
            &format!(
                "This launcher location is deprecated.\\n\\nUse only:\\n{}",
                canonical.display()
            ),
        );
        return Err(format!("Deprecated launcher path: {}", exe.display()));
    }
    Ok(())
}

fn show_alert(title: &str, message: &str) {
    let script = format!(
        "display alert \"{}\" message \"{}\"",
        title.replace('"', "'"),
        message.replace('"', "'")
    );
    let _ = Command::new("osascript").arg("-e").arg(script).status();
}
