//! Agent tool definitions, registry, and built-in tool implementations.

pub mod base;
pub mod registry;
pub mod filesystem;
pub mod shell;
pub mod web;
pub mod message;
pub mod spawn;
pub mod cron_tool;
pub mod email;

pub use base::Tool;
pub use registry::ToolRegistry;
pub use filesystem::{ReadFileTool, WriteFileTool, EditFileTool, ListDirTool};
pub use shell::ExecTool;
pub use web::{WebSearchTool, WebFetchTool};
pub use message::{MessageTool, SendCallback};
pub use spawn::{SpawnTool, SpawnCallback};
pub use cron_tool::CronScheduleTool;
pub use email::{CheckInboxTool, SendEmailTool};
