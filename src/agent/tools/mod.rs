//! Agent tool definitions, registry, and built-in tool implementations.

pub mod base;
pub mod browser;
pub mod code_execution;
pub mod cron_tool;
pub mod email;
pub mod filesystem;
pub mod message;
pub mod read_skill;
pub mod reasoning_tools;
pub mod recall;
pub mod registry;
pub mod remember;
pub mod session_search;
pub mod shell;
pub mod spawn;
pub mod web;

pub use base::Tool;
pub use browser::BrowserTool;
pub use code_execution::CodeExecutionTool;
pub use cron_tool::CronScheduleTool;
pub use email::{CheckInboxTool, SendEmailTool};
pub use filesystem::{EditFileTool, ListDirTool, ReadFileTool, WriteFileTool};
pub use message::{MessageTool, SendCallback};
pub use read_skill::ReadSkillTool;
pub use recall::RecallTool;
pub use registry::ToolRegistry;
pub use remember::RememberTool;
pub use session_search::SessionSearchTool;
pub use shell::ExecTool;
pub use spawn::{
    CancelCallback, CheckCallback, ListCallback, LoopCallback, PipelineCallback, SpawnCallback,
    SpawnTool, SpawnToolLite, WaitCallback,
};
pub use web::{WebFetchTool, WebSearchTool};
