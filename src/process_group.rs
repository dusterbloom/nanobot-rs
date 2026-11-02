//! Process-group guard for spawned subprocesses.
//!
//! `tokio::process::Child` kills only the immediate child on drop (and only
//! when `kill_on_drop(true)` was set) — it never reaches grandchildren. A
//! command like `sh -c "foo | bar"` forks `foo` and `bar` as children of
//! `sh`; killing `sh` alone leaves them to be reparented and keep running
//! forever (observed: orphaned `grep | head` pipelines alive for a day).
//! [`GroupChild`] puts the spawned process in its own process group (unix)
//! and kills the whole group on drop, so grandchildren die too.
//!
//! On non-unix platforms there is no process-group primitive here; `Drop`
//! falls back to killing just the immediate child.

use std::io;
use std::ops::{Deref, DerefMut};
use std::process::Output;

use tokio::process::{Child, Command};

/// Wraps a [`Child`] spawned into its own process group so [`Drop`] can kill
/// the whole group (the child and any grandchildren it forked), not just the
/// immediate child. Derefs to [`Child`] so call sites keep using
/// `.stdout.take()`, `.wait()`, etc. unchanged.
pub struct GroupChild(Child);

/// Spawn `cmd` into its own process group (unix) and return the guard.
///
/// On unix, this sets `process_group(0)`, which puts the new process in a
/// fresh group equal to its own pid — `Drop` sends `SIGKILL` to that pid's
/// group, reaching anything the command itself forked (e.g. a shell
/// pipeline's grandchildren). On non-unix platforms the process is spawned
/// normally; `Drop` kills only the immediate child.
pub fn spawn(cmd: &mut Command) -> io::Result<GroupChild> {
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.spawn().map(GroupChild)
}

impl Deref for GroupChild {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for GroupChild {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for GroupChild {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // `id()` returns `None` once the child has been reaped (`wait` /
        // `try_wait` observed its exit). That is the whole safety argument
        // for the unsafe call below: we only ever signal a pid tokio still
        // considers live, so a normally-exited command's own intentional
        // background jobs are not killed by a stale pid, and there is no
        // pid-reuse race (a reaped pid is never signaled again here).
        #[cfg(unix)]
        {
            if let Some(raw_pid) = self.0.id() {
                if let Ok(pgid) = i32::try_from(raw_pid) {
                    // Safety: `pgid` is a live pid tokio still tracks for
                    // this child (see above). Passing it to `killpg` with
                    // `SIGKILL` follows the documented POSIX contract and
                    // cannot violate memory safety regardless of process
                    // state.
                    unsafe {
                        libc::killpg(pgid, libc::SIGKILL);
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            if self.0.id().is_some() {
                let _ = self.0.start_kill();
            }
        }
    }
}

/// Run `cmd` to completion and collect its output, the same contract as
/// [`Child::wait_with_output`], but through a [`GroupChild`] so the whole
/// process group is killed if this future is dropped before completion
/// (e.g. by an enclosing `tokio::time::timeout` firing, or a `select!`
/// branch cancelling it).
///
/// Unlike [`std::process::Command::output`], this does not configure stdio
/// itself: call `.stdout(Stdio::piped())` / `.stderr(Stdio::piped())` on
/// `cmd` first for the corresponding stream to be captured (left as the
/// default, inherited setup, a stream reads back empty — same as
/// `Child::wait_with_output` when no pipe was requested for it).
///
/// `wait_with_output` is not reachable through `Deref` because it consumes
/// `Child` by value — consuming it here would drop the `GroupChild` guard
/// before the wait completes, defeating the whole point. So this instead
/// reads both pipes concurrently with the wait (the same technique the std
/// implementation uses internally), keeping the guard alive for the whole
/// operation.
pub async fn output(cmd: &mut Command) -> io::Result<Output> {
    use tokio::io::AsyncReadExt;

    let mut child = spawn(cmd)?;
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    let read_stdout = async {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            pipe.read_to_end(&mut buf).await?;
        }
        Ok::<_, io::Error>(buf)
    };
    let read_stderr = async {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            pipe.read_to_end(&mut buf).await?;
        }
        Ok::<_, io::Error>(buf)
    };

    let (status, stdout, stderr) = tokio::try_join!(child.wait(), read_stdout, read_stderr)?;
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}
