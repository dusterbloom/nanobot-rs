//! Agent tool definitions, registry, and built-in tool implementations.

pub mod base;
pub mod cron_tool;
pub mod email;
pub mod filesystem;
pub mod message;
pub mod registry;
pub mod scratchpad_tool;
pub mod shell;
pub mod skill_manager;
pub mod spawn;
pub mod taskboard_tool;
pub mod web;

pub use base::Tool;
pub use cron_tool::CronScheduleTool;
pub use email::{CheckInboxTool, SendEmailTool};
pub use filesystem::{EditFileTool, ListDirTool, ReadFileTool, WriteFileTool};
pub use message::{MessageTool, SendCallback};
pub use registry::ToolRegistry;
pub use shell::ExecTool;
pub use scratchpad_tool::ScratchpadTool;
pub use skill_manager::SkillManagerTool;
pub use spawn::{SpawnCallback, SpawnTool};
pub use taskboard_tool::TaskBoardTool;
pub use web::{WebFetchTool, WebSearchTool};
