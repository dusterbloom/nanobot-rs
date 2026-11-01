//! Agent tool definitions, registry, and built-in tool implementations.

pub mod apply_patch;
pub mod base;
pub mod browser;
pub mod context_edit;
pub mod cron_tool;
pub mod cua;
pub mod email;
pub mod filesystem;
pub mod message;
#[cfg(feature = "python-kernel")]
pub mod python_kernel;
pub mod read_skill;
pub mod reasoning_tools;
pub mod recall;
pub mod registry;
pub mod remember;
pub mod shell;
pub mod spawn;
pub mod stash_search;
pub mod todo;
pub mod web;

pub use base::Tool;
pub use browser::BrowserTool;
pub use context_edit::ContextTool;
pub use cron_tool::CronScheduleTool;
pub use cua::CuaTool;
pub use email::{CheckInboxTool, SendEmailTool};
pub use filesystem::{EditFileTool, ListDirTool, ReadFileTool, SearchFilesTool, WriteFileTool};
pub use message::MessageTool;
#[cfg(feature = "python-kernel")]
pub use python_kernel::PythonKernel;
pub use read_skill::ReadSkillTool;
pub use recall::RecallTool;
pub use registry::ToolRegistry;
pub use remember::RememberTool;
pub use shell::ExecTool;
pub use spawn::{SpawnTool, SpawnToolLite};
pub use todo::TodoTool;
pub use web::{WebFetchTool, WebSearchTool};
