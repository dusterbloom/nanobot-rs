//! Persistent stateful Python kernel via embedded CPython (PyO3).
//!
//! Unlike `execute_code` which spawns a fresh python3 process per call,
//! this tool holds a persistent CPython interpreter in-process.
//! Variables, imports, and function definitions survive across calls.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{json, Value};
use tokio::task;

use super::base::{PermissionLevel, Tool, ToolContext, ToolResult};
use crate::errors::ToolError;

// ---------------------------------------------------------------------------
// Capture stdout/stderr using only builtins (no import statements needed)
// ---------------------------------------------------------------------------

const CAPTURE_SETUP: &str = concat!(
    "__capture_out = __import__('io').StringIO()\n",
    "__capture_err = __import__('io').StringIO()\n",
    "__capture_old_out = __import__('sys').stdout\n",
    "__capture_old_err = __import__('sys').stderr\n",
    "__import__('sys').stdout = __capture_out\n",
    "__import__('sys').stderr = __capture_err\n",
);

const CAPTURE_TEARDOWN: &str = concat!(
    "__import__('sys').stdout = __capture_old_out\n",
    "__import__('sys').stderr = __capture_old_err\n",
    "__capture_result = (__capture_out.getvalue(), __capture_err.getvalue())\n",
    "__capture_out = __import__('io').StringIO()\n",
    "__capture_err = __import__('io').StringIO()\n",
);

// ---------------------------------------------------------------------------
// Tool struct
// ---------------------------------------------------------------------------

/// Point fd 0 at /dev/null while kernel code runs, restoring it when the
/// last concurrent shield drops.
///
/// User Python executes in-process: `open('/dev/stdin')` or `input()` would
/// issue a blocking read(2) on the process's real terminal — the same file
/// the UI's event loop reads — permanently stealing every other keystroke
/// (the "type each key twice" failure) and wedging the kernel. With the
/// shield, such reads return EOF instantly.
///
/// Refcounted because fd 0 is process-global: concurrent kernel calls (and
/// the tests that exercise them) would otherwise restore the terminal under
/// each other's running code. A wedged call simply never drops its count —
/// fd 0 stays /dev/null, which only makes later stray reads EOF.
struct KernelStdinShield;

static SHIELD_COUNT: std::sync::Mutex<(usize, Option<i32>)> = std::sync::Mutex::new((0, None));

impl KernelStdinShield {
    fn new() -> Self {
        // Poisoning only happens if a holder panicked mid-update; treat that
        // as "leave the shield on" rather than abort the kernel.
        let mut state = SHIELD_COUNT
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (count, saved_tty) = &mut *state;
        if *count == 0 {
            #[allow(unsafe_code)]
            unsafe {
                let saved = libc::dup(libc::STDIN_FILENO);
                if let Ok(null) = std::fs::File::open("/dev/null") {
                    use std::os::unix::io::AsRawFd;
                    libc::dup2(null.as_raw_fd(), libc::STDIN_FILENO);
                    // `null` closes on drop; fd 0 keeps the dup2'd copy.
                    *saved_tty = (saved >= 0).then_some(saved);
                } else if saved >= 0 {
                    libc::close(saved);
                }
            }
        }
        *count += 1;
        Self
    }
}

impl Drop for KernelStdinShield {
    fn drop(&mut self) {
        let mut state = SHIELD_COUNT
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (count, saved_tty) = &mut *state;
        *count = count.saturating_sub(1);
        if *count == 0 {
            if let Some(saved) = *saved_tty {
                #[allow(unsafe_code)]
                unsafe {
                    libc::dup2(saved, libc::STDIN_FILENO);
                    libc::close(saved);
                }
                *saved_tty = None;
            }
        }
    }
}

/// Process-wide count of wedged kernel threads (grace timeout fired, thread
/// never returned). Such threads can never be joined: the REPL must not wait
/// for them at shutdown.
static WEDGED_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn wedged_thread_count() -> usize {
    WEDGED_THREADS.load(Ordering::SeqCst)
}

pub struct PythonKernel {
    /// Held in a `RwLock` so a wedged kernel's globals can be swapped for a
    /// fresh dict: when a call misses the watchdog, its thread keeps the old
    /// inner mutex locked forever. Replacing the `Arc` frees later calls;
    /// the stuck thread is leaked but holds no GIL (blocking file/socket
    /// reads release it for the duration of the syscall).
    globals: std::sync::RwLock<Arc<Mutex<Py<PyDict>>>>,
    /// Set when a call timed out without responding to interruption. The
    /// next call replaces the globals instead of blocking on the dead mutex.
    wedged: AtomicBool,
    timeout: Duration,
}

impl PythonKernel {
    pub fn new(timeout_secs: u64) -> Self {
        let globals = Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.into()
        });
        Self {
            globals: std::sync::RwLock::new(Arc::new(Mutex::new(globals))),
            wedged: AtomicBool::new(false),
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Globals for this call, swapping in a fresh dict first if the previous
    /// call wedged the kernel.
    fn globals_for_call(&self) -> Arc<Mutex<Py<PyDict>>> {
        if self.wedged.swap(false, Ordering::SeqCst) {
            let fresh: Py<PyDict> = Python::with_gil(|py| PyDict::new(py).into());
            let new = Arc::new(Mutex::new(fresh));
            let mut slot = self.globals.write().unwrap_or_else(|e| e.into_inner());
            *slot = Arc::clone(&new);
            return new;
        }
        Arc::clone(&self.globals.read().unwrap_or_else(|e| e.into_inner()))
    }
}

#[async_trait]
impl Tool for PythonKernel {
    fn name(&self) -> &str {
        "python"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn description(&self) -> &str {
        "Execute Python code in a persistent kernel. \
         Variables, imports, and functions survive across calls. \
         Only explicit print() output is returned. \
         There is no stdin — never read input() or open('/dev/stdin'); \
         fetch web pages with urllib.request, read files by path. \
         Each call must finish within its timeout; a blocking call wedges \
         the kernel (it is then replaced, losing all variables)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. State persists across calls — variables, imports, and functions defined in one call are available in subsequent calls. Use print() to return output."
                }
            },
            "required": ["code"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        let code = match params.get("code").and_then(|v| v.as_str()) {
            Some(c) if !c.trim().is_empty() => c.to_string(),
            None => {
                return Err(ToolError::InvalidArgs {
                    message: "'code' parameter is required and must not be empty".to_string(),
                })
            }
            Some(c) => c.to_string(),
        };

        let c_code = match CString::new(code) {
            Ok(c) => c,
            Err(_) => {
                return Err(ToolError::InvalidArgs {
                    message: "'code' must not contain null bytes".to_string(),
                })
            }
        };

        let globals = self.globals_for_call();
        let timeout = self.timeout;

        // Watchdog handshake: the blocking thread publishes its CPython thread
        // id, the watchdog raises TimeoutError *inside* the interpreter once the
        // deadline passes, and `done` stops it from firing into the next call.
        let thread_id = Arc::new(AtomicI64::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let watchdog = spawn_watchdog(Arc::clone(&thread_id), Arc::clone(&done), timeout);

        let result = task::spawn_blocking(move || {
            // Block terminal-stealing reads for the whole run (setup, user
            // code, teardown). A blocking user call that never returns keeps
            // this thread (and the shield) alive — see KernelStdinShield.
            let _stdin_shield = KernelStdinShield::new();
            Python::with_gil(|py| {
                // A panic under the GIL would otherwise poison the mutex and
                // brick every later call.
                let guard = globals.lock().unwrap_or_else(|e| e.into_inner());
                let bound = (*guard).bind(py);
                let g = &*bound;

                thread_id.store(current_python_thread_id(py), Ordering::SeqCst);

                let setup = CString::new(CAPTURE_SETUP).unwrap();
                let teardown = CString::new(CAPTURE_TEARDOWN).unwrap();

                if let Err(e) = py.run(&setup, Some(g), None) {
                    return format!("Error: capture setup failed: {e}");
                }
                let run_err = py.run(&c_code, Some(g), None).err();
                // Stop the watchdog BEFORE teardown: teardown is real Python
                // bytecode with GIL release points; a `SetAsyncExc` landing
                // there would cause teardown to raise, leaving `sys.stdout`
                // pointed at a dead StringIO — the exact leak we claim to fix.
                done.store(true, Ordering::SeqCst);
                // Clear any pending async exception the watchdog may have just
                // raised, so teardown runs cleanly.
                #[allow(unsafe_code)]
                unsafe {
                    let id = current_python_thread_id(py);
                    // null → clear pending exception for this thread
                    pyo3::ffi::PyThreadState_SetAsyncExc(id, std::ptr::null_mut());
                }
                // Unconditional teardown: on the error path this is what
                // restores sys.stdout and refreshes `__capture_result`. Skipping
                // it left the next call reading the previous call's buffers.
                let teardown_err = py.run(&teardown, Some(g), None).err();

                if let Some(e) = teardown_err {
                    return format!("Error: capture teardown failed: {e}");
                }
                let captured = extract_output(py, g);
                match run_err {
                    None => captured,
                    Some(e) => format_exception(py, &e, &captured),
                }
            })
        });

        // The in-interpreter watchdog is the real timeout. This outer bound only
        // covers the case it cannot reach — a C extension holding the GIL
        // without releasing it, where no async exception is ever checked.
        let grace = timeout + Duration::from_secs(5);
        let output = match tokio::time::timeout(grace, result).await {
            Ok(Ok(output)) => output,
            Ok(Err(join_err)) if join_err.is_panic() => {
                format!("Error: kernel thread panicked: {join_err}")
            }
            Ok(Err(join_err)) => format!("Error: {join_err}"),
            Err(_) => {
                // Mark the kernel wedged: the runner thread never returned and
                // still holds the old globals mutex. The next call swaps in a
                // fresh interpreter namespace instead of blocking forever.
                self.wedged.store(true, Ordering::SeqCst);
                WEDGED_THREADS.fetch_add(1, Ordering::SeqCst);
                format!(
                    "Error: kernel did not respond to interruption {}s past its {}s \
                     timeout — a blocking or native call is holding the kernel \
                     (common cause: reading stdin, which does not exist here). \
                     The kernel will be replaced on the next call; variables \
                     defined so far are lost.",
                    grace.as_secs() - timeout.as_secs(),
                    timeout.as_secs()
                )
            }
        };
        watchdog.join().ok();
        // Thread output is a flat string; split the legacy error channel at
        // this one boundary exactly as the funnel did (byte-identical).
        match output.strip_prefix("Error:").map(str::trim) {
            Some(err) => Err(ToolError::Execution {
                message: err.to_string(),
            }),
            None => Ok(output.into()),
        }
    }
}

/// Poll for the deadline, then raise `TimeoutError` inside the running Python
/// thread via `PyThreadState_SetAsyncExc` — the same mechanism CPython's own
/// `ctypes` interrupt trick uses. The interpreter checks pending async
/// exceptions between bytecodes, so pure-Python loops and GIL-releasing calls
/// (`time.sleep`, IO) are interruptible.
///
/// ponytail: 50ms poll instead of a condvar. Upgrade only if kernel calls get
/// hot enough for the idle wakeups to matter.
fn spawn_watchdog(
    thread_id: Arc<AtomicI64>,
    done: Arc<AtomicBool>,
    timeout: Duration,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // Wait for the blocking task to publish its thread_id. A prior call
        // still holding the mutex + GIL causes the current task to queue;
        // starting the deadline immediately would let it expire before the
        // queued task ever runs (thread_id == 0 → return → no timeout at all).
        while thread_id.load(Ordering::SeqCst) == 0 {
            if done.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if done.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let id = thread_id.load(Ordering::SeqCst);
        if id == 0 || done.load(Ordering::SeqCst) {
            return;
        }
        // Acquiring the GIL here succeeds because the running code releases it
        // periodically (switch interval) or during blocking calls.
        Python::with_gil(|py| {
            if done.load(Ordering::SeqCst) {
                return;
            }
            let exc = py.get_type::<pyo3::exceptions::PyTimeoutError>();
            #[allow(unsafe_code)]
            // Safety: GIL held; `id` is a live CPython thread id and `exc` a
            // borrowed type object that outlives the call.
            unsafe {
                pyo3::ffi::PyThreadState_SetAsyncExc(id, exc.as_ptr());
            }
        });
    })
}

fn current_python_thread_id(py: Python<'_>) -> std::os::raw::c_long {
    py.import("threading")
        .and_then(|m| m.call_method0("get_ident"))
        .and_then(|v| v.extract())
        .unwrap_or(0)
}

/// Render a Python exception with its traceback, keeping whatever the code
/// managed to print before it raised.
fn format_exception(py: Python<'_>, e: &PyErr, captured: &str) -> String {
    let traceback = e
        .traceback(py)
        .and_then(|tb| tb.format().ok())
        .unwrap_or_default();
    let body = format!("{traceback}{e}");
    match captured {
        "(no output)" | "" => body,
        prior => format!("{prior}\n{body}"),
    }
}

fn extract_output(py: Python<'_>, g: &pyo3::Bound<'_, PyDict>) -> String {
    let out: String = py
        .eval(&CString::new("__capture_result[0]").unwrap(), Some(g), None)
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default();

    let err: String = py
        .eval(&CString::new("__capture_result[1]").unwrap(), Some(g), None)
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default();

    match (out.is_empty(), err.is_empty()) {
        (true, true) => "(no output)".to_string(),
        (false, true) => out.trim_end().to_string(),
        (true, false) => format!("stderr:\n{}", err.trim_end()),
        (false, false) => format!("{}\nstderr:\n{}", out.trim_end(), err.trim_end()),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> PythonKernel {
        PythonKernel::new(10)
    }

    #[tokio::test]
    async fn simple_print() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("print('hello')"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("hello"), "got: {}", result);
    }

    #[tokio::test]
    async fn no_output() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("x = 42"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("(no output)"), "got: {}", result);
    }

    #[tokio::test]
    async fn variable_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("x = 42"));
        crate::agent::tools::base::render_result(
            k.execute(
                params.clone(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        params.insert("code".to_string(), json!("print(x * 2)"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("84"), "got: {}", result);
    }

    #[tokio::test]
    async fn import_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("import json"));
        crate::agent::tools::base::render_result(
            k.execute(
                params.clone(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        params.insert("code".to_string(), json!("print(json.dumps({'a': 1}))"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("{\"a\": 1}"), "got: {}", result);
    }

    #[tokio::test]
    async fn function_definition_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert(
            "code".to_string(),
            json!("def double(n):\n    return n * 2"),
        );
        crate::agent::tools::base::render_result(
            k.execute(
                params.clone(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        params.insert("code".to_string(), json!("print(double(21))"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("42"), "got: {}", result);
    }

    #[tokio::test]
    async fn syntax_error_graceful() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("def (broken"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(!result.contains("panicked"), "got: {}", result);
        assert!(
            result.contains("SyntaxError") || result.contains("Error"),
            "got: {}",
            result
        );
    }

    #[tokio::test]
    async fn runtime_error_returns_traceback() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("1 / 0"));
        let result = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.contains("ZeroDivisionError") || result.contains("division by zero"),
            "got: {}",
            result
        );
    }

    #[tokio::test]
    async fn empty_code_rejected() {
        let k = kernel();
        let result = crate::agent::tools::base::render_result(
            k.execute(
                HashMap::new(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(result.starts_with("Error:"), "got: {}", result);
    }

    /// The timeout must actually stop the interpreter, not just abandon the
    /// task: a runaway loop is interrupted and the SAME kernel still works
    /// afterwards. Abandoning the thread would leave the GIL held and wedge the
    /// second call forever.
    #[tokio::test]
    async fn timeout_interrupts_and_kernel_survives() {
        let k = PythonKernel::new(1);
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("while True:\n    pass"));
        let result = crate::agent::tools::base::render_result(
            k.execute(
                params.clone(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            result.contains("TimeoutError"),
            "expected an in-interpreter TimeoutError, got: {}",
            result
        );

        params.insert("code".to_string(), json!("print('alive')"));
        let after = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            after.contains("alive"),
            "kernel wedged after timeout: {after}"
        );
    }

    /// Run one kernel call on a leaked detached runtime. Kernel tests must
    /// not own a joinable tokio runtime: any lingering blocking-pool thread
    /// (keep-alive, or a wedged call that can never return) would hang the
    /// runtime's shutdown and the test with it.
    fn run_kernel_detached(code: String) -> String {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let mut kernel = PythonKernel::new(30);
            let mut params = HashMap::new();
            params.insert("code".to_string(), json!(code));
            let rendered = {
                let ctx = crate::agent::tools::base::ToolContext::sandbox();
                let out = kernel.execute(params, &ctx);
                crate::agent::tools::base::render_result(rt.block_on(out))
            };
            std::mem::forget(rt);
            tx.send(rendered).unwrap();
        });
        rx.recv().expect("detached kernel call must report back")
    }

    /// `open('/dev/stdin')` must return EOF instantly, not issue a blocking
    /// read(2) on the process's real terminal — that steal-every-other-key
    /// failure bricked interactive sessions (2026-08-29). The verdict is
    /// written to a file: the embedded interpreter's stdout capture is
    /// process-global and other parallel kernel tests clobber it.
    #[test]
    fn stdin_reads_return_eof_instead_of_blocking() {
        let verdict = std::env::temp_dir().join(format!(
            "nanobot-kernel-stdin-{}.txt",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&verdict);
        let path = verdict.to_string_lossy().to_string();
        let code = format!(
            "import pathlib\nout = []\nout.append('len %d' % len(open('/dev/stdin').read()))\ntry:\n    input()\n    out.append('input got')\nexcept EOFError:\n    out.append('input eof')\npathlib.Path({path:?}).write_text(' | '.join(out))"
        );
        let result = run_kernel_detached(code.clone());
        let written = std::fs::read_to_string(&verdict)
            .unwrap_or_else(|_| format!("<no verdict; kernel said: {result}>"));
        let _ = std::fs::remove_file(&verdict);
        assert!(
            written.contains("len 0"),
            "stdin read must be EOF: {written} (kernel: {result})"
        );
        assert!(
            written.contains("input eof"),
            "input() must EOF: {written} (kernel: {result})"
        );
    }

    /// A call that blocks in a syscall (pipe read with no writer) can neither
    /// finish nor be interrupted — the grace timeout fires, the kernel is
    /// marked wedged, and the NEXT call must succeed on a fresh namespace
    /// instead of blocking on the dead mutex forever. This is the
    /// `open('/dev/stdin')` production failure: state is lost, the tool is not.
    ///
    /// Runs each call on a leaked detached runtime: the wedged blocking
    /// thread can never be joined, and a normal `#[tokio::test]` runtime
    /// would hang its own shutdown waiting for it (the zombie-process
    /// failure this suite exists to prevent).
    #[test]
    fn wedged_kernel_is_replaced_on_next_call() {
        // Both calls share one kernel on one leaked runtime — the wedge and
        // the recovery must hit the same instance.
        let (tx, rx) = std::sync::mpsc::channel::<(String, String)>();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let mut kernel = PythonKernel::new(1);
            let run = |code: String| {
                let mut params = HashMap::new();
            params.insert("code".to_string(), json!(code.as_str()));
                let ctx = crate::agent::tools::base::ToolContext::sandbox();
                let out = kernel.execute(params, &ctx);
                crate::agent::tools::base::render_result(rt.block_on(out))
            };
            let wedged = run(
                "import os\nprint('sentinel')\nos.read(os.pipe()[0], 1)".to_string(),
            );
            let recovered = run("print('recovered')".to_string());
            // Leaking the runtime keeps its shutdown from joining the wedged
            // blocking thread.
            std::mem::forget(rt);
            tx.send((wedged, recovered)).unwrap();
        });
        let (wedged, recovered) = rx.recv().expect("kernel calls must report back");

        assert!(wedged.contains("did not respond"), "got: {wedged}");
        assert!(wedged.contains("replaced"), "got: {wedged}");
        assert!(recovered.contains("recovered"), "kernel stayed wedged: {recovered}");
        assert!(super::wedged_thread_count() >= 1, "wedge counter must record the leak");
    }

    /// An exception must not leave sys.stdout redirected or `__capture_result`
    /// stale — the next call's output has to be its own.
    #[tokio::test]
    async fn capture_recovers_after_exception() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("print('before'); 1 / 0"));
        let failed = crate::agent::tools::base::render_result(
            k.execute(
                params.clone(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(failed.contains("ZeroDivisionError"), "got: {failed}");

        params.insert("code".to_string(), json!("print('after')"));
        let ok = crate::agent::tools::base::render_result(
            k.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(ok.contains("after"), "got: {ok}");
        assert!(!ok.contains("before"), "stale buffer leaked: {ok}");
    }
}
