//! Agent tool definitions, registry, and built-in tool implementations.

pub mod apply_patch;
pub mod base;
pub mod batch;
pub mod browser;
pub mod code_execution;
pub mod cron_tool;
pub mod email;
pub mod file_preview;
pub mod filesystem;
pub mod message;
pub mod read_skill;
pub mod reasoning_tools;
pub mod recall;
pub mod registry;
pub mod remember;
pub mod search_context;
pub mod session_search;
pub mod shell;
pub mod spawn;
pub mod system_info;
pub mod todo;
pub mod tool_status;
pub mod web;

pub use apply_patch::ApplyPatchTool;
pub use base::{PermissionLevel, Tool};
pub use batch::BatchTool;
pub use browser::BrowserTool;
pub use code_execution::CodeExecutionTool;
pub use cron_tool::CronScheduleTool;
pub use email::{CheckInboxTool, SendEmailTool};
pub use file_preview::FilePreviewTool;
pub use filesystem::{
    EditFileTool, FileInfoTool, FindFilesTool, ListDirTool, ReadFileTool, SearchFilesTool,
    WorkspaceDiffTool, WriteFileTool,
};
pub use message::{MessageTool, SendCallback};
pub use read_skill::ReadSkillTool;
pub use recall::RecallTool;
pub use registry::ToolRegistry;
pub use remember::RememberTool;
pub use search_context::SearchContextTool;
pub use session_search::SessionSearchTool;
pub use shell::ExecTool;
pub use spawn::{
    CancelCallback, CheckCallback, ListCallback, LoopCallback, PipelineCallback, SpawnCallback,
    SpawnTool, SpawnToolLite, WaitCallback,
};
pub use system_info::SystemInfoTool;
pub use todo::TodoTool;
pub use tool_status::ToolStatusTool;
pub use web::{WebFetchTool, WebSearchTool};
