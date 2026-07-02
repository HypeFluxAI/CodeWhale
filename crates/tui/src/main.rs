//! CLI entry point for CodeWhale.
//!
//! All functionality lives in the `codewhale_tui` library crate so that
//! host applications (e.g. the ThinkAI Kids desktop/iPad app) can drive
//! the agent in-process via `codewhale_tui::exec::run_exec` instead of
//! spawning this binary as a subprocess. This shim only forwards to the
//! library's CLI entry point.

fn main() -> anyhow::Result<()> {
    codewhale_tui::cli_main()
}
