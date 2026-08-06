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

use super::base::{PermissionLevel, Tool};

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

pub struct PythonKernel {
    globals: Arc<Mutex<Py<PyDict>>>,
    timeout: Duration,
}

impl PythonKernel {
    pub fn new(timeout_secs: u64) -> Self {
        let globals = Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.into()
        });
        Self {
            globals: Arc::new(Mutex::new(globals)),
            timeout: Duration::from_secs(timeout_secs),
        }
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
         Only explicit print() output is returned."
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

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let code = match params.get("code").and_then(|v| v.as_str()) {
            Some(c) if !c.trim().is_empty() => c.to_string(),
            _ => return "Error: 'code' parameter is required and must not be empty".to_string(),
        };

        let c_code = match CString::new(code) {
            Ok(c) => c,
            Err(_) => return "Error: 'code' must not contain null bytes".to_string(),
        };

        let globals = Arc::clone(&self.globals);
        let timeout = self.timeout;

        // Watchdog handshake: the blocking thread publishes its CPython thread
        // id, the watchdog raises TimeoutError *inside* the interpreter once the
        // deadline passes, and `done` stops it from firing into the next call.
        let thread_id = Arc::new(AtomicI64::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let watchdog = spawn_watchdog(Arc::clone(&thread_id), Arc::clone(&done), timeout);

        let result = task::spawn_blocking(move || {
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
            Err(_) => format!(
                "Error: kernel did not respond to interruption {}s past its {}s \
                 timeout — a blocking or native call is holding the GIL. Further \
                 `python` calls will block until it returns.",
                grace.as_secs() - timeout.as_secs(),
                timeout.as_secs()
            ),
        };
        watchdog.join().ok();
        output
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
    let out: String = py.eval(
        &CString::new("__capture_result[0]").unwrap(),
        Some(g),
        None,
    )
    .ok()
    .and_then(|v| v.extract::<String>().ok())
    .unwrap_or_default();

    let err: String = py.eval(
        &CString::new("__capture_result[1]").unwrap(),
        Some(g),
        None,
    )
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
        let result = k.execute(params).await;
        assert!(result.contains("hello"), "got: {}", result);
    }

    #[tokio::test]
    async fn no_output() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("x = 42"));
        let result = k.execute(params).await;
        assert!(result.contains("(no output)"), "got: {}", result);
    }

    #[tokio::test]
    async fn variable_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("x = 42"));
        k.execute(params.clone()).await;
        params.insert("code".to_string(), json!("print(x * 2)"));
        let result = k.execute(params).await;
        assert!(result.contains("84"), "got: {}", result);
    }

    #[tokio::test]
    async fn import_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("import json"));
        k.execute(params.clone()).await;
        params.insert("code".to_string(), json!("print(json.dumps({'a': 1}))"));
        let result = k.execute(params).await;
        assert!(result.contains("{\"a\": 1}"), "got: {}", result);
    }

    #[tokio::test]
    async fn function_definition_persistence() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("def double(n):\n    return n * 2"));
        k.execute(params.clone()).await;
        params.insert("code".to_string(), json!("print(double(21))"));
        let result = k.execute(params).await;
        assert!(result.contains("42"), "got: {}", result);
    }

    #[tokio::test]
    async fn syntax_error_graceful() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("def (broken"));
        let result = k.execute(params).await;
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
        let result = k.execute(params).await;
        assert!(
            result.contains("ZeroDivisionError") || result.contains("division by zero"),
            "got: {}",
            result
        );
    }

    #[tokio::test]
    async fn empty_code_rejected() {
        let k = kernel();
        let result = k.execute(HashMap::new()).await;
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
        let result = k.execute(params.clone()).await;
        assert!(
            result.contains("TimeoutError"),
            "expected an in-interpreter TimeoutError, got: {}",
            result
        );

        params.insert("code".to_string(), json!("print('alive')"));
        let after = k.execute(params).await;
        assert!(after.contains("alive"), "kernel wedged after timeout: {after}");
    }

    /// An exception must not leave sys.stdout redirected or `__capture_result`
    /// stale — the next call's output has to be its own.
    #[tokio::test]
    async fn capture_recovers_after_exception() {
        let k = kernel();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("print('before'); 1 / 0"));
        let failed = k.execute(params.clone()).await;
        assert!(failed.contains("ZeroDivisionError"), "got: {failed}");

        params.insert("code".to_string(), json!("print('after')"));
        let ok = k.execute(params).await;
        assert!(ok.contains("after"), "got: {ok}");
        assert!(!ok.contains("before"), "stale buffer leaked: {ok}");
    }
}
