//! Headless exec mode.
//!
//! This is the agent loop behind `codewhale exec` (extracted from the old
//! `main.rs` so it is reachable as a library). Two entry points:
//!
//! - [`run_exec_agent`]: the CLI path used by the `codewhale-tui` binary.
//!   Stream-json events go to stdout exactly as before the extraction.
//! - [`run_exec`]: the embeddable API. Host applications that cannot spawn
//!   subprocesses (e.g. iOS) call this in-process; stream events are
//!   delivered as typed [`ExecStreamEvent`] values over an mpsc channel and
//!   the run can be aborted through a [`CancellationToken`].

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, MAX_SUBAGENTS};
use crate::model_routing;
use crate::models::{Message, SystemPrompt};
use crate::session_manager::{self, SessionManager, truncate_id};
use crate::tui::history::{summarize_tool_args, summarize_tool_output};
use crate::{ExecArgs, ExecOutputFormat};

pub(crate) fn resolve_exec_model(config: &Config, explicit_model: Option<&str>) -> String {
    explicit_model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| config.default_model())
}

pub(crate) fn resolve_exec_resume_session_id(
    args: &ExecArgs,
    workspace: &Path,
) -> Result<Option<String>> {
    if let Some(id) = args.resume.as_ref().or(args.session_id.as_ref()) {
        return Ok(Some(id.clone()));
    }
    if !args.continue_session {
        return Ok(None);
    }
    crate::latest_session_id_for_workspace(workspace)?.map_or_else(
        || {
            bail!(
                "No saved sessions found for workspace {}. Use `codewhale sessions` to list sessions, or pass `codewhale exec --resume <SESSION_ID> ...`.",
                workspace.display()
            )
        },
        |id| Ok(Some(id)),
    )
}

pub(crate) struct CliAutoRoute {
    pub(crate) model: String,
    pub(crate) reasoning_effort: Option<crate::tui::app::ReasoningEffort>,
    pub(crate) auto_model: bool,
}

pub(crate) async fn resolve_cli_auto_route(
    config: &Config,
    model: &str,
    prompt: &str,
) -> CliAutoRoute {
    if model.trim().eq_ignore_ascii_case("auto") {
        let selection =
            model_routing::resolve_auto_route_with_flash(config, prompt, "", "auto", "auto").await;
        CliAutoRoute {
            model: selection.model,
            reasoning_effort: selection.reasoning_effort,
            auto_model: true,
        }
    } else {
        // When --model is not `auto`, fall back to the reasoning_effort
        // declared in the user's config.toml. The previous hard-coded `None`
        // silently dropped the user's setting on every non-auto-route exec
        // call, which (for example) prevented vllm + Qwen3 users from
        // disabling thinking via `reasoning_effort = "off"` and caused
        // 30+ second SSE idle timeouts on trivial prompts.
        CliAutoRoute {
            model: model.to_string(),
            reasoning_effort: config
                .reasoning_effort()
                .map(crate::tui::app::ReasoningEffort::from_setting),
            auto_model: false,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecStreamMeta {
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub session_id: String,
    pub resume_command: String,
    pub workspace: String,
    pub message_count: usize,
    pub status: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ExecStreamEvent {
    #[serde(rename = "content")]
    Content { content: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        name: String,
        id: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        id: String,
        output: String,
        status: String,
    },
    #[serde(rename = "session_capture")]
    SessionCapture { content: String },
    #[serde(rename = "metadata")]
    Metadata { meta: ExecStreamMeta },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { error: String },
    /// GEN_RELIABILITY_FIX. Lightweight liveness signal emitted while the model
    /// is streaming reasoning (`ThinkingDelta`) or text (`MessageDelta`) but has
    /// not yet produced a tool-call or file-write. The host app uses it to poke
    /// its idle watchdog so a long silent-thinking gap (DeepSeek V4 reasoning on
    /// large contexts) is not mistaken for a hang. Throttled (see exec loop);
    /// does NOT change what text is delivered to the final work.
    #[serde(rename = "heartbeat")]
    Heartbeat { phase: &'static str },
}

/// Where stream-json events go: the CLI prints one JSON line per event to
/// stdout (byte-identical to the pre-extraction `emit_exec_stream_event`),
/// while embedders receive the typed events over a channel.
enum ExecEventSink {
    Stdout,
    Channel(UnboundedSender<ExecStreamEvent>),
}

impl ExecEventSink {
    fn emit_exec_stream_event(&self, event: ExecStreamEvent) -> Result<()> {
        match self {
            Self::Stdout => {
                println!("{}", serde_json::to_string(&event)?);
                Ok(())
            }
            Self::Channel(tx) => {
                // A dropped receiver must not abort the run; cancellation is
                // signalled explicitly via the CancellationToken.
                let _ = tx.send(event);
                Ok(())
            }
        }
    }
}

pub(crate) fn exec_resume_command(session_id: &str) -> String {
    if session_id.trim().is_empty() {
        String::new()
    } else {
        format!("codewhale exec --resume {session_id}")
    }
}

fn persist_exec_session(
    messages: &[Message],
    model: &str,
    workspace: &Path,
    system_prompt: &Option<SystemPrompt>,
    session_id: Option<&str>,
    total_tokens: u64,
) -> Result<String> {
    let manager =
        SessionManager::default_location().context("could not open session manager for save")?;
    let saved = if let Some(id) = session_id.filter(|id| !id.trim().is_empty()) {
        match manager.load_session(id) {
            Ok(existing) => session_manager::update_session(
                existing,
                messages,
                total_tokens,
                system_prompt.as_ref(),
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                session_manager::create_saved_session_with_id_and_mode(
                    id.to_string(),
                    messages,
                    model,
                    workspace,
                    total_tokens,
                    system_prompt.as_ref(),
                    Some("exec"),
                )
            }
            Err(err) => return Err(err).context("could not load existing exec session"),
        }
    } else {
        session_manager::create_saved_session_with_mode(
            messages,
            model,
            workspace,
            total_tokens,
            system_prompt.as_ref(),
            Some("exec"),
        )
    };
    let id = saved.metadata.id.clone();
    manager
        .save_session(&saved)
        .context("could not save exec session")?;
    Ok(id)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecToolEntry {
    pub name: String,
    pub success: bool,
    pub output: String,
}

#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct ExecSummary {
    pub mode: String,
    pub model: String,
    pub prompt: String,
    pub output: String,
    pub tools: Vec<ExecToolEntry>,
    pub status: Option<String>,
    pub error: Option<String>,
}

/// Result of one exec run (library API).
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    /// Aggregated run summary (same shape the CLI prints with `--json`).
    pub summary: ExecSummary,
    /// Session id the run was persisted under, when session persistence
    /// applied (always the case for stream-json / library runs that
    /// completed a turn).
    pub session_id: Option<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Request for an embedded exec run. Mirrors the CLI surface the ThinkAI
/// desktop bridge passes to `codewhale exec --auto --output-format
/// stream-json` (see cw_tui.rs in ThinkAIDesktop), minus the process env:
/// `api_key` / `base_url` / `provider` replace the `DEEPSEEK_API_KEY` /
/// `DEEPSEEK_BASE_URL` / `DEEPSEEK_PROVIDER` env injection so the host
/// process env is never mutated.
#[derive(Debug, Clone, Default)]
pub struct ExecRequest {
    /// User prompt (the CLI's trailing PROMPT argument).
    pub prompt: String,
    /// Model override; `None` falls back to the configured default model.
    pub model: Option<String>,
    /// API base URL override. Interpreted like `DEEPSEEK_BASE_URL`: it is
    /// applied to the root `base_url`, which the deepseek / deepseek-cn
    /// providers read. Other providers configure `[providers.*]` instead.
    pub base_url: Option<String>,
    /// API key override. Interpreted like `DEEPSEEK_API_KEY` for the
    /// deepseek / deepseek-cn providers; wins over env and saved config.
    pub api_key: Option<String>,
    /// Provider override, e.g. `"deepseek"` (like `DEEPSEEK_PROVIDER`).
    pub provider: Option<String>,
    /// Workspace directory the agent operates in (the CLI's cwd).
    pub cwd: PathBuf,
    /// Resume a previous session by id (the CLI's `--resume`).
    pub resume_session_id: Option<String>,
    /// `--allowed-tools`: lowercase catalog names; all others denied.
    pub allowed_tools: Option<Vec<String>>,
    /// `--disallowed-tools`: deny wins over allow.
    pub disallowed_tools: Option<Vec<String>>,
    /// `--max-turns`; `None` uses the CLI default (100).
    pub max_turns: Option<u32>,
    /// `--append-system-prompt`: extra text appended to the system prompt.
    pub append_system_prompt: Option<String>,
    /// Max parallel subagents; `None` uses the configured default.
    pub max_subagents: Option<usize>,
    /// Redirect all CodeWhale state (sessions, …) into this directory
    /// instead of `~/.codewhale`, without touching env vars. Needed for the
    /// iOS sandbox. `None` keeps current behavior. Note: this installs a
    /// process-wide override (see `codewhale_config::set_codewhale_home_override`).
    pub state_dir: Option<PathBuf>,
    /// Images attached to the prompt (the CLI's repeatable `--image`).
    pub images: Vec<PathBuf>,
}

/// Run one exec-agent turn in-process.
///
/// Equivalent to `codewhale exec --auto --output-format stream-json` with
/// the stream-json lines delivered as typed [`ExecStreamEvent`] values on
/// `events` instead of stdout. Cancelling `cancel` is the in-process
/// replacement for killing the subprocess: the engine's in-flight work is
/// cancelled and shut down, and the run returns with status `"canceled"`.
pub async fn run_exec(
    req: ExecRequest,
    events: UnboundedSender<ExecStreamEvent>,
    cancel: CancellationToken,
) -> Result<ExecOutcome> {
    if let Some(dir) = req.state_dir.clone() {
        codewhale_config::set_codewhale_home_override(Some(dir));
    }

    let profile = std::env::var("DEEPSEEK_PROFILE").ok();
    let mut config =
        Config::load(None, profile.as_deref()).context("could not load CodeWhale config")?;
    if let Some(provider) = req.provider.clone() {
        config.provider = Some(provider);
    }
    if let Some(base_url) = req.base_url.clone() {
        config.base_url = Some(base_url);
    }
    if let Some(api_key) = req.api_key.clone() {
        config.api_key = Some(api_key);
    }
    crate::merge_user_workspace_config(&mut config, None, &req.cwd);

    let model = resolve_exec_model(&config, req.model.as_deref());
    let max_subagents = req.max_subagents.map_or_else(
        || config.max_subagents(),
        |value| value.clamp(1, MAX_SUBAGENTS),
    );
    // Same normalization the CLI dispatch applies to --allowed-tools /
    // --disallowed-tools.
    let allowed_tools = req.allowed_tools.as_ref().map(|v| {
        v.iter()
            .map(|s| s.to_ascii_lowercase().trim().to_string())
            .collect::<Vec<_>>()
    });
    let disallowed_tools = req.disallowed_tools.as_ref().map(|v| {
        v.iter()
            .map(|s| s.to_ascii_lowercase().trim().to_string())
            .collect::<Vec<_>>()
    });

    run_exec_agent_core(
        &config,
        &model,
        &req.prompt,
        req.cwd.clone(),
        max_subagents,
        true, // auto_approve (--auto)
        true, // trust_mode (--auto)
        false,
        req.resume_session_id.clone(),
        ExecOutputFormat::StreamJson,
        req.max_turns.unwrap_or(100),
        allowed_tools,
        disallowed_tools,
        req.append_system_prompt.clone(),
        req.images.clone(),
        &ExecEventSink::Channel(events),
        &cancel,
    )
    .await
}

/// Encode an attached image as a `data:` URL for an `image_url` part.
fn image_data_url(path: &std::path::Path) -> Result<String> {
    use base64::Engine as _;
    // Images ride along on every request of the agent loop, and providers
    // cap a single image at ~5 MB *encoded* (base64 grows data by 4/3), so
    // keep the raw file under 3.75 MB and fail early with a clear error.
    const MAX_IMAGE_BYTES: u64 = 3_750_000;
    let len = std::fs::metadata(path)
        .with_context(|| format!("could not read image {}", path.display()))?
        .len();
    if len > MAX_IMAGE_BYTES {
        bail!(
            "image {} is {} bytes; the limit is 3.75 MB (5 MB once encoded) — downscale it first",
            path.display(),
            len
        );
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("could not read image {}", path.display()))?;
    // Declare the type the bytes actually are (a JPEG saved as .png would be
    // rejected by providers that check the declared media type).
    let mime = if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF8") {
        "image/gif"
    } else if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        bail!("unsupported image format for {}: use png, jpeg, gif or webp", path.display());
    };
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// CLI entry point for `codewhale exec` in engine mode. Stream-json events
/// go to stdout; behavior is identical to the pre-extraction function.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_exec_agent(
    config: &Config,
    model: &str,
    prompt: &str,
    workspace: PathBuf,
    max_subagents: usize,
    auto_approve: bool,
    trust_mode: bool,
    json_output: bool,
    resume_session_id: Option<String>,
    output_format: ExecOutputFormat,
    max_turns: u32,
    allowed_tools: Option<Vec<String>>,
    disallowed_tools: Option<Vec<String>>,
    append_system_prompt: Option<String>,
    images: Vec<PathBuf>,
) -> Result<()> {
    let outcome = run_exec_agent_core(
        config,
        model,
        prompt,
        workspace,
        max_subagents,
        auto_approve,
        trust_mode,
        json_output,
        resume_session_id,
        output_format,
        max_turns,
        allowed_tools,
        disallowed_tools,
        append_system_prompt,
        images,
        &ExecEventSink::Stdout,
        &CancellationToken::new(),
    )
    .await?;
    let summary = outcome.summary;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }

    if let Some(error) = summary.error.as_ref()
        && !error.trim().is_empty()
    {
        bail!("exec turn failed: {error}");
    }

    if matches!(
        summary.status.as_deref(),
        Some("failed" | "canceled" | "interrupted")
    ) {
        let status = summary.status.as_deref().unwrap_or("unknown");
        bail!("exec turn ended with status {status}");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_exec_agent_core(
    config: &Config,
    model: &str,
    prompt: &str,
    workspace: PathBuf,
    max_subagents: usize,
    auto_approve: bool,
    trust_mode: bool,
    json_output: bool,
    resume_session_id: Option<String>,
    output_format: ExecOutputFormat,
    max_turns: u32,
    allowed_tools: Option<Vec<String>>,
    disallowed_tools: Option<Vec<String>>,
    append_system_prompt: Option<String>,
    images: Vec<PathBuf>,
    sink: &ExecEventSink,
    cancel: &CancellationToken,
) -> Result<ExecOutcome> {
    use crate::compaction::CompactionConfig;
    use crate::core::engine::{EngineConfig, spawn_engine};
    use crate::core::events::Event;
    use crate::core::ops::Op;
    use crate::models::{
        auto_compact_default_for_model, compaction_threshold_for_model_at_percent,
    };
    use crate::tools::plan::new_shared_plan_state;
    use crate::tools::todo::new_shared_todo_list;
    use crate::tui::app::AppMode;

    // Read attachments up front so a bad path fails before any model call.
    let image_urls = images
        .iter()
        .map(|p| image_data_url(p))
        .collect::<Result<Vec<_>>>()?;

    let route = resolve_cli_auto_route(config, model, prompt).await;
    let auto_model = route.auto_model;
    let effective_model = route.model;
    let effective_reasoning_effort = route
        .reasoning_effort
        .map(|effort| effort.as_setting().to_string());

    let settings = crate::settings::Settings::load().unwrap_or_default();
    let auto_compact_enabled = if crate::settings::Settings::auto_compact_explicitly_configured() {
        settings.auto_compact
    } else {
        auto_compact_default_for_model(&effective_model)
    };
    let compaction = CompactionConfig {
        enabled: auto_compact_enabled,
        model: effective_model.clone(),
        token_threshold: compaction_threshold_for_model_at_percent(
            &effective_model,
            settings.auto_compact_threshold_percent,
        ),
        ..Default::default()
    };

    let network_policy = config.network.clone().map(|toml_cfg| {
        crate::network_policy::NetworkPolicyDecider::with_default_audit(toml_cfg.into_runtime())
    });

    let lsp_config = config
        .lsp
        .clone()
        .map(crate::config::LspConfigToml::into_runtime);
    let engine_config = EngineConfig {
        model: effective_model.clone(),
        // ThinkAI item 4: opt-in fast mechanical turns via env (default off). ThinkAI's
        // subprocess sets CODEWHALE_FAST_MECHANICAL_TURNS=1 once verified on real games.
        fast_mechanical_turns: std::env::var("CODEWHALE_FAST_MECHANICAL_TURNS").ok().as_deref()
            == Some("1"),
        workspace: workspace.clone(),
        allow_shell: auto_approve || config.allow_shell(),
        trust_mode,
        notes_path: config.notes_path(),
        mcp_config_path: config.mcp_config_path(),
        skills_dir: config.skills_dir(),
        instructions: {
            let mut instrs: Vec<crate::prompts::InstructionSource> = config
                .instructions_paths()
                .into_iter()
                .map(Into::into)
                .collect();
            if let Some(ref extra) = append_system_prompt {
                instrs.push(crate::prompts::InstructionSource::Inline {
                    name: "cli:append-system-prompt".into(),
                    content: extra.clone(),
                });
            }
            instrs
        },
        project_context_pack_enabled: config.project_context_pack_enabled(),
        translation_enabled: false,
        show_thinking: settings.show_thinking,
        max_steps: max_turns,
        max_subagents,
        features: config.features(),
        compaction,
        capacity: crate::core::capacity::CapacityControllerConfig::from_app_config(config),
        todos: new_shared_todo_list(),
        plan_state: new_shared_plan_state(),
        goal_state: crate::tools::goal::new_shared_goal_state(),
        max_spawn_depth: crate::tools::subagent::DEFAULT_MAX_SPAWN_DEPTH,
        network_policy,
        snapshots_enabled: config.snapshots_config().enabled,
        snapshots_max_workspace_bytes: config
            .snapshots_config()
            .max_workspace_gb
            .saturating_mul(1024 * 1024 * 1024),
        lsp_config,
        runtime_services: crate::tools::spec::RuntimeToolServices::default(),
        subagent_model_overrides: config.subagent_model_overrides(),
        subagent_api_timeout: std::time::Duration::from_secs(config.subagent_api_timeout_secs()),
        stream_chunk_timeout: std::time::Duration::from_secs(config.stream_chunk_timeout_secs()),
        subagent_heartbeat_timeout: std::time::Duration::from_secs(
            config.subagent_heartbeat_timeout_secs(),
        ),
        prefer_bwrap: config.prefer_bwrap.unwrap_or(false),
        memory_enabled: config.memory_enabled(),
        memory_path: config.memory_path(),
        speech_output_dir: config.speech_output_dir(),
        vision_config: config.vision_model_config(),
        strict_tool_mode: config.strict_tool_mode.unwrap_or(false),
        goal_objective: None,
        allowed_tools: allowed_tools.clone(),
        disallowed_tools: disallowed_tools.clone(),
        hook_executor: None,
        locale_tag: crate::localization::resolve_locale(&settings.locale)
            .tag()
            .to_string(),
        workshop: config.workshop.clone(),
        search_provider: config.search_provider(),
        search_api_key: config.search.as_ref().and_then(|s| s.api_key.clone()),
        search_base_url: config.search.as_ref().and_then(|s| s.base_url.clone()),
        tools_always_load: config.tools_always_load(),
        tools: config.tools.clone(),
        verbosity: config.verbosity.clone(),
    };

    let engine_handle = spawn_engine(engine_config, config);
    let mode = if auto_approve {
        AppMode::Yolo
    } else {
        AppMode::Agent
    };

    let mut loaded_session_id = None;
    if let Some(session_id) = resume_session_id.as_deref() {
        let manager = SessionManager::default_location()
            .context("could not open session manager for exec resume")?;
        let saved = manager
            .load_session_by_prefix(session_id)
            .with_context(|| format!("could not load session '{session_id}'"))?;
        let saved_id = saved.metadata.id.clone();
        if saved.metadata.workspace != workspace && output_format == ExecOutputFormat::Text {
            eprintln!(
                "Warning: session {} was created in a different workspace ({}). Resuming anyway.",
                truncate_id(&saved_id),
                saved.metadata.workspace.display(),
            );
        }

        engine_handle
            .send(Op::SyncSession {
                session_id: Some(saved_id.clone()),
                messages: saved.messages,
                system_prompt: saved.system_prompt.map(SystemPrompt::Text),
                system_prompt_override: false,
                model: saved.metadata.model,
                workspace: saved.metadata.workspace,
            })
            .await?;
        loaded_session_id = Some(saved_id.clone());
        if output_format == ExecOutputFormat::Text && !json_output {
            eprintln!("resumed session: {saved_id}");
        }
    }

    engine_handle
        .send(Op::SendMessage {
            content: prompt.to_string(),
            mode,
            model: effective_model.clone(),
            goal_objective: None,
            allowed_tools: allowed_tools.clone(),
            hook_executor: None,
            reasoning_effort: effective_reasoning_effort,
            reasoning_effort_auto: auto_model,
            auto_model,
            allow_shell: auto_approve || config.allow_shell(),
            trust_mode,
            auto_approve,
            translation_enabled: false,
            show_thinking: settings.show_thinking,
            approval_mode: if auto_approve {
                crate::tui::approval::ApprovalMode::Auto
            } else {
                config
                    .approval_policy
                    .as_deref()
                    .and_then(crate::tui::approval::ApprovalMode::from_config_value)
                    .unwrap_or_default()
            },
            verbosity: config.verbosity.clone(),
            images: image_urls,
        })
        .await?;

    let mut summary = ExecSummary {
        mode: "agent".to_string(),
        model: effective_model.clone(),
        prompt: prompt.to_string(),
        ..ExecSummary::default()
    };

    let should_persist_session =
        resume_session_id.is_some() || output_format == ExecOutputFormat::StreamJson;
    let mut latest_session_id = loaded_session_id;
    let mut latest_messages: Vec<Message> = Vec::new();
    let mut latest_system_prompt: Option<SystemPrompt> = None;
    let mut latest_model = effective_model;
    let mut latest_workspace = workspace.clone();

    let mut outcome_session_id: Option<String> = None;
    let mut outcome_input_tokens: u32 = 0;
    let mut outcome_output_tokens: u32 = 0;

    let mut stdout = io::stdout();
    let mut ends_with_newline = false;
    // GEN_RELIABILITY_FIX. Throttle for the liveness `Heartbeat` events emitted
    // during silent thinking / text streaming. We coalesce so we emit at most
    // ~one per second no matter how many token deltas arrive — enough to keep
    // the host watchdog fed, without spamming one event per token.
    let mut last_heartbeat = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap_or_else(std::time::Instant::now);
    const HEARTBEAT_MIN_GAP: std::time::Duration = std::time::Duration::from_millis(1000);
    let maybe_heartbeat =
        |sink: &ExecEventSink, phase: &'static str, last: &mut std::time::Instant| {
            if output_format != ExecOutputFormat::StreamJson {
                return Ok(());
            }
            if last.elapsed() < HEARTBEAT_MIN_GAP {
                return Ok(());
            }
            *last = std::time::Instant::now();
            sink.emit_exec_stream_event(ExecStreamEvent::Heartbeat { phase })
        };
    loop {
        let event = {
            let mut rx = engine_handle.rx_event.write().await;
            tokio::select! {
                biased;
                () = cancel.cancelled() => None,
                event = rx.recv() => event,
            }
        };

        let Some(event) = event else {
            if cancel.is_cancelled() {
                // In-process replacement for killing the subprocess: abort
                // the in-flight turn, shut the engine down, and report a
                // canceled status to the caller.
                summary.status = Some("canceled".to_string());
                let _ = engine_handle.send(Op::CancelRequest).await;
                let _ = engine_handle.send(Op::Shutdown).await;
            }
            break;
        };

        match event {
            Event::MessageDelta { content, .. } => {
                summary.output.push_str(&content);
                if output_format == ExecOutputFormat::StreamJson {
                    // GEN_RELIABILITY_FIX: keep the host watchdog alive during
                    // text streaming too (not just on tools/files). The Content
                    // event itself already reaches the host, but emit a throttled
                    // heartbeat as a uniform liveness signal alongside it.
                    maybe_heartbeat(&sink, "text", &mut last_heartbeat)?;
                    sink.emit_exec_stream_event(ExecStreamEvent::Content { content })?;
                } else if !json_output {
                    print!("{content}");
                    stdout.flush()?;
                }
                ends_with_newline = summary.output.ends_with('\n');
            }
            Event::MessageComplete { .. }
                if output_format == ExecOutputFormat::Text
                    && !json_output
                    && !ends_with_newline =>
            {
                println!();
            }
            Event::ThinkingDelta { .. } => {
                // Exec stream-json intentionally omits reasoning deltas from the
                // work text; the TUI transcript retains its existing Activity
                // Detail surface. GEN_RELIABILITY_FIX: but we DO emit a throttled
                // liveness heartbeat so the host watchdog sees the model is alive
                // during a long silent-thinking gap (V4 reasoning) — otherwise a
                // tool-only watchdog fires 210s into a healthy thinking pause.
                maybe_heartbeat(&sink, "thinking", &mut last_heartbeat)?;
            }
            Event::ToolCallStarted { id, name, input } => {
                if output_format == ExecOutputFormat::StreamJson {
                    sink.emit_exec_stream_event(ExecStreamEvent::ToolUse { name, id, input })?;
                } else if !json_output {
                    let summary = summarize_tool_args(&input);
                    if let Some(summary) = summary {
                        eprintln!("tool: {name} ({summary})");
                    } else {
                        eprintln!("tool: {name}");
                    }
                }
            }
            Event::ToolCallProgress { id, output }
                if output_format == ExecOutputFormat::Text && !json_output =>
            {
                eprintln!("tool {id}: {}", summarize_tool_output(&output));
            }
            Event::ToolCallComplete {
                id, name, result, ..
            } => match result {
                Ok(output) => {
                    summary.tools.push(ExecToolEntry {
                        name: name.clone(),
                        success: output.success,
                        output: output.content.clone(),
                    });
                    if output_format == ExecOutputFormat::StreamJson {
                        sink.emit_exec_stream_event(ExecStreamEvent::ToolResult {
                            id,
                            output: output.content,
                            status: if output.success {
                                "success".to_string()
                            } else {
                                "error".to_string()
                            },
                        })?;
                    } else if !json_output {
                        if name == "exec_shell" && !output.content.trim().is_empty() {
                            eprintln!("tool {name} completed");
                            eprintln!(
                                "--- stdout/stderr ---\n{}\n---------------------",
                                output.content
                            );
                        } else {
                            eprintln!(
                                "tool {name} completed: {}",
                                summarize_tool_output(&output.content)
                            );
                        }
                    }
                }
                Err(err) => {
                    let error_text = err.to_string();
                    summary.tools.push(ExecToolEntry {
                        name: name.clone(),
                        success: false,
                        output: error_text.clone(),
                    });
                    if output_format == ExecOutputFormat::StreamJson {
                        sink.emit_exec_stream_event(ExecStreamEvent::ToolResult {
                            id,
                            output: error_text,
                            status: "error".to_string(),
                        })?;
                    } else if !json_output {
                        eprintln!("tool {name} failed: {err}");
                    }
                }
            },
            Event::AgentSpawned { id, prompt }
                if output_format == ExecOutputFormat::Text && !json_output =>
            {
                eprintln!("sub-agent {id} spawned: {}", summarize_tool_output(&prompt));
            }
            Event::AgentProgress { id, status }
                if output_format == ExecOutputFormat::Text && !json_output =>
            {
                eprintln!("sub-agent {id}: {status}");
            }
            Event::AgentComplete { id, result }
                if output_format == ExecOutputFormat::Text && !json_output =>
            {
                eprintln!(
                    "sub-agent {id} completed: {}",
                    summarize_tool_output(&result)
                );
            }
            Event::AgentSpawned { .. }
            | Event::AgentProgress { .. }
            | Event::AgentComplete { .. } => {}
            Event::ApprovalRequired { id, .. } => {
                if auto_approve {
                    let _ = engine_handle.approve_tool_call(id).await;
                } else {
                    let _ = engine_handle.deny_tool_call(id).await;
                }
            }
            Event::ElevationRequired {
                tool_id,
                tool_name,
                denial_reason,
                ..
            } => {
                if auto_approve {
                    if output_format == ExecOutputFormat::Text && !json_output {
                        eprintln!("sandbox denied {tool_name}: {denial_reason} (auto-elevating)");
                    }
                    let policy = crate::sandbox::SandboxPolicy::DangerFullAccess;
                    let _ = engine_handle.retry_tool_with_policy(tool_id, policy).await;
                } else {
                    if output_format == ExecOutputFormat::Text && !json_output {
                        eprintln!("sandbox denied {tool_name}: {denial_reason}");
                    }
                    let _ = engine_handle.deny_tool_call(tool_id).await;
                }
            }
            Event::Error {
                envelope,
                recoverable: _,
            } => {
                // GEN_RELIABILITY_FIX A3: prefix a stable `[CODE]` token so the
                // client can route user-facing copy precisely (idempotent —
                // no-ops if already tagged).
                let coded = crate::error_taxonomy::with_error_code(envelope.message);
                summary.error = Some(coded.clone());
                if output_format == ExecOutputFormat::StreamJson {
                    sink.emit_exec_stream_event(ExecStreamEvent::Error { error: coded })?;
                } else if !json_output {
                    eprintln!("error: {coded}");
                }
            }
            Event::TurnComplete {
                status,
                error,
                usage,
                ..
            } => {
                summary.status = Some(format!("{status:?}").to_lowercase());
                // GEN_RELIABILITY_FIX A3: tag the terminal turn error too. This
                // is the string the client parses off the final ExecOutcome.
                summary.error = error.map(crate::error_taxonomy::with_error_code);
                let saved_session_id = if should_persist_session && !latest_messages.is_empty() {
                    match persist_exec_session(
                        &latest_messages,
                        &latest_model,
                        &latest_workspace,
                        &latest_system_prompt,
                        latest_session_id.as_deref(),
                        u64::from(usage.input_tokens) + u64::from(usage.output_tokens),
                    ) {
                        Ok(id) => {
                            if output_format == ExecOutputFormat::Text && !json_output {
                                eprintln!("session: {id}");
                            }
                            Some(id)
                        }
                        Err(err) => {
                            if output_format == ExecOutputFormat::Text && !json_output {
                                eprintln!("warning: failed to save exec session: {err}");
                            }
                            latest_session_id.clone()
                        }
                    }
                } else {
                    latest_session_id.clone()
                };

                outcome_session_id = saved_session_id.clone();
                outcome_input_tokens = usage.input_tokens;
                outcome_output_tokens = usage.output_tokens;

                if output_format == ExecOutputFormat::StreamJson {
                    if let Some(id) = saved_session_id.as_ref() {
                        sink.emit_exec_stream_event(ExecStreamEvent::SessionCapture {
                            content: id.clone(),
                        })?;
                    }
                    sink.emit_exec_stream_event(ExecStreamEvent::Metadata {
                        meta: ExecStreamMeta {
                            model: latest_model.clone(),
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            resume_command: saved_session_id
                                .as_deref()
                                .map(exec_resume_command)
                                .unwrap_or_default(),
                            session_id: saved_session_id.unwrap_or_default(),
                            workspace: latest_workspace.display().to_string(),
                            message_count: latest_messages.len(),
                            status: summary.status.clone(),
                        },
                    })?;
                    sink.emit_exec_stream_event(ExecStreamEvent::Done)?;
                }
                let _ = engine_handle.send(Op::Shutdown).await;
                break;
            }
            Event::SessionUpdated {
                session_id,
                messages,
                system_prompt,
                model,
                workspace,
            } => {
                latest_session_id = Some(session_id);
                latest_messages = messages;
                latest_system_prompt = system_prompt;
                latest_model = model;
                latest_workspace = workspace;
            }
            // #3027: surface the engine's max-steps notice in text mode so a
            // --max-turns run that stops early says why instead of going quiet.
            Event::Status { message }
                if output_format == ExecOutputFormat::Text
                    && !json_output
                    && message.contains("Reached maximum steps") =>
            {
                eprintln!("{message}");
            }
            // GEN_RELIABILITY_FIX (mobile compaction stall): auto-compaction runs
            // a heavy local prune loop + a summarization LLM call that can burn
            // >150s emitting NO event the host watchdog recognizes — so the host
            // kills a healthy, busy engine mid-compaction. Map compaction lifecycle
            // + generic status events to a throttled liveness heartbeat (reusing the
            // shared HEARTBEAT_MIN_GAP helper) so the host's idle watchdog stays fed
            // through the whole silent compaction window. turn_loop.rs also emits a
            // periodic Status tick every ~20s during compaction, and each such tick
            // lands here → heartbeat.
            Event::CompactionStarted { .. }
            | Event::CompactionCompleted { .. }
            | Event::CompactionFailed { .. }
            | Event::Status { .. } => {
                maybe_heartbeat(&sink, "compaction", &mut last_heartbeat)?;
            }
            _ => {}
        }
    }

    let outcome_session_id = outcome_session_id.or(latest_session_id);

    Ok(ExecOutcome {
        summary,
        session_id: outcome_session_id,
        input_tokens: outcome_input_tokens,
        output_tokens: outcome_output_tokens,
    })
}

#[cfg(test)]
mod image_attachment_tests {
    use super::image_data_url;

    #[test]
    fn image_data_url_encodes_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hero.PNG");
        std::fs::write(&p, [0x89u8, b'P', b'N', b'G']).unwrap();
        assert_eq!(image_data_url(&p).unwrap(), "data:image/png;base64,iVBORw==");
        // 扩展名说 png, 内容其实是 jpeg → 按内容报 jpeg
        let lying = dir.path().join("photo.png");
        std::fs::write(&lying, [0xffu8, 0xd8, 0xff, 0xe0]).unwrap();
        assert!(image_data_url(&lying).unwrap().starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn image_data_url_rejects_unknown_format_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let bmp = dir.path().join("x.bmp");
        std::fs::write(&bmp, [0u8]).unwrap();
        assert!(image_data_url(&bmp).is_err());
        assert!(image_data_url(&dir.path().join("missing.png")).is_err());
        let big = dir.path().join("big.png");
        std::fs::write(&big, vec![0u8; 3_750_001]).unwrap();
        assert!(image_data_url(&big).is_err());
    }
}
