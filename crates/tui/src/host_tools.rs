//! Host-provided tools for embedders of the in-process engine (`exec::run_exec`).
//!
//! An app that runs CodeWhale in its own process (e.g. ThinkAI on iOS/Android,
//! where spawning helper executables is impossible) can implement [`ToolSpec`]
//! and hand the tools over via [`crate::exec::ExecRequest::host_tools`]; they are
//! registered into every turn's tool registry next to the built-in tools
//! (not in Plan mode, and not for sub-agents). They are always in the
//! model-visible catalog (never deferred). A host tool whose name collides with
//! a built-in is skipped with an error log — pick a distinct name.

use std::fmt;
use std::sync::Arc;

pub use crate::tools::spec::{ToolContext, ToolSpec};
pub use async_trait::async_trait;
pub use codewhale_tools::{ApprovalRequirement, ToolCapability, ToolError, ToolResult};

/// A list of host tools. Cheap to clone (shared `Arc`s).
#[derive(Clone, Default)]
pub struct HostTools(pub Vec<Arc<dyn ToolSpec>>);

impl HostTools {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for HostTools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter().map(|t| t.name())).finish()
    }
}
