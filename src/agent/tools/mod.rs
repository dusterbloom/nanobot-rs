//! Agent tool definitions, registry, and built-in tool implementations.

pub mod apply_patch;
pub mod base;
pub mod browser;
pub mod code_execution;
pub mod cron_tool;
pub mod email;
pub mod file_preview;
pub mod filesystem;
pub mod message;
#[cfg(feature = "python-kernel")]
pub mod python_kernel;
pub mod read_skill;
pub mod reasoning_tools;
pub mod recall;
pub mod recall_tool_result;
pub mod registry;
pub mod remember;
pub mod shell;
pub mod spawn;
pub mod stash_search;
pub mod system_info;
pub mod todo;
pub mod tool_status;
pub mod web;

pub use apply_patch::ApplyPatchTool;
pub use base::Tool;
pub use browser::BrowserTool;
pub use code_execution::CodeExecutionTool;
pub use cron_tool::CronScheduleTool;
pub use email::{CheckInboxTool, SendEmailTool};
pub use file_preview::FilePreviewTool;
pub use filesystem::{
    EditFileTool, FileInfoTool, FindFilesTool, ListDirTool, ReadFileTool, SearchFilesTool,
    WorkspaceDiffTool, WriteFileTool,
};
pub use message::MessageTool;
#[cfg(feature = "python-kernel")]
pub use python_kernel::PythonKernel;
pub use read_skill::ReadSkillTool;
pub use recall::RecallTool;
pub use registry::ToolRegistry;
pub use remember::RememberTool;
pub use shell::ExecTool;
pub use spawn::{SpawnTool, SpawnToolLite};
pub use system_info::SystemInfoTool;
pub use todo::TodoTool;
pub use tool_status::ToolStatusTool;
pub use web::{WebFetchTool, WebSearchTool};
