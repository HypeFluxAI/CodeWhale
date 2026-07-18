//! Shared error taxonomy across client, tools, runtime, and UI.
use std::fmt;

use crate::llm_client::LlmError;
use crate::tools::spec::ToolError;

/// Broad category for typed error handling and policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    Network,
    Authentication,
    Authorization,
    RateLimit,
    Timeout,
    InvalidInput,
    Parse,
    Tool,
    State,
    Internal,
}

/// Severity hint for UI and logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Unified envelope used when crossing subsystem boundaries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub recoverable: bool,
    pub code: String,
    pub message: String,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Network => "network",
            Self::Authentication => "authentication",
            Self::Authorization => "authorization",
            Self::RateLimit => "rate_limit",
            Self::Timeout => "timeout",
            Self::InvalidInput => "invalid_input",
            Self::Parse => "parse",
            Self::Tool => "tool",
            Self::State => "state",
            Self::Internal => "internal",
        };
        f.write_str(label)
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        };
        f.write_str(label)
    }
}

impl fmt::Display for ErrorEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.code, self.message)
    }
}

impl std::error::Error for ErrorEnvelope {}

impl ErrorEnvelope {
    #[must_use]
    pub fn new(
        category: ErrorCategory,
        severity: ErrorSeverity,
        recoverable: bool,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            recoverable,
            code: code.into(),
            message: message.into(),
        }
    }

    /// Recoverable internal error — stream stalls, transient retries, generic
    /// engine errors that the user can resolve by retrying. Severity is
    /// `Warning` so the UI surfaces it in amber rather than red.
    #[must_use]
    pub fn transient(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::Internal,
            ErrorSeverity::Warning,
            true,
            "transient",
            message,
        )
    }

    /// Non-recoverable internal error — missing client, spawn failure, etc.
    /// Flips the session into offline mode.
    #[must_use]
    pub fn fatal(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::Internal,
            ErrorSeverity::Error,
            false,
            "fatal",
            message,
        )
    }

    /// Authentication failure — fatal and blocks the session.
    #[must_use]
    pub fn fatal_auth(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::Authentication,
            ErrorSeverity::Critical,
            false,
            "auth_fatal",
            message,
        )
    }

    /// Context length / overflow — invalid input, recoverable via /compact.
    #[must_use]
    pub fn context_overflow(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::InvalidInput,
            ErrorSeverity::Error,
            true,
            "context_overflow",
            message,
        )
    }

    /// Recoverable network / transport hiccup.
    #[must_use]
    pub fn network(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::Network,
            ErrorSeverity::Warning,
            true,
            "network_transient",
            message,
        )
    }

    /// Tool execution failure.
    #[must_use]
    pub fn tool(message: impl Into<String>) -> Self {
        Self::new(
            ErrorCategory::Tool,
            ErrorSeverity::Error,
            true,
            "tool_failed",
            message,
        )
    }

    /// Build an envelope by classifying a raw error message string. Used at
    /// boundaries where the underlying error type was already stringified.
    #[must_use]
    pub fn classify(message: impl Into<String>, recoverable: bool) -> Self {
        let message = message.into();
        let category = classify_error_message(&message);
        let severity = match category {
            ErrorCategory::Authentication => ErrorSeverity::Critical,
            ErrorCategory::RateLimit | ErrorCategory::Timeout | ErrorCategory::Network => {
                ErrorSeverity::Warning
            }
            ErrorCategory::InvalidInput | ErrorCategory::Authorization | ErrorCategory::Parse => {
                ErrorSeverity::Error
            }
            ErrorCategory::Tool | ErrorCategory::State | ErrorCategory::Internal => {
                if recoverable {
                    ErrorSeverity::Warning
                } else {
                    ErrorSeverity::Error
                }
            }
        };
        Self::new(
            category,
            severity,
            recoverable,
            category.to_string(),
            message,
        )
    }
}

impl From<LlmError> for ErrorEnvelope {
    fn from(value: LlmError) -> Self {
        match value {
            LlmError::RateLimited { message, .. } => Self::new(
                ErrorCategory::RateLimit,
                ErrorSeverity::Warning,
                true,
                "llm_rate_limited",
                message,
            ),
            LlmError::ServerError { status, message } => Self::new(
                ErrorCategory::Internal,
                ErrorSeverity::Error,
                true,
                format!("llm_server_{status}"),
                message,
            ),
            LlmError::NetworkError(message) => Self::new(
                ErrorCategory::Network,
                ErrorSeverity::Error,
                true,
                "llm_network_error",
                message,
            ),
            LlmError::Timeout(duration) => Self::new(
                ErrorCategory::Timeout,
                ErrorSeverity::Warning,
                true,
                "llm_timeout",
                format!("Request timed out after {duration:?}"),
            ),
            LlmError::AuthenticationError(auth) => Self::new(
                ErrorCategory::Authentication,
                ErrorSeverity::Critical,
                false,
                "llm_auth_error",
                auth.to_user_message(),
            ),
            LlmError::AuthorizationError(message) => Self::new(
                ErrorCategory::Authorization,
                ErrorSeverity::Error,
                false,
                "llm_authorization_error",
                message,
            ),
            LlmError::InvalidRequest { message, .. } => Self::new(
                ErrorCategory::InvalidInput,
                ErrorSeverity::Error,
                false,
                "llm_invalid_request",
                message,
            ),
            LlmError::ModelError(message) => Self::new(
                ErrorCategory::InvalidInput,
                ErrorSeverity::Error,
                false,
                "llm_model_error",
                message,
            ),
            LlmError::ContentPolicyError(message) => Self::new(
                ErrorCategory::Authorization,
                ErrorSeverity::Error,
                false,
                "llm_content_policy",
                message,
            ),
            LlmError::ParseError(message) => Self::new(
                ErrorCategory::Parse,
                ErrorSeverity::Error,
                false,
                "llm_parse_error",
                message,
            ),
            LlmError::ContextLengthError(message) => Self::new(
                ErrorCategory::InvalidInput,
                ErrorSeverity::Error,
                false,
                "llm_context_length",
                message,
            ),
            LlmError::Other(message) => Self::new(
                ErrorCategory::Internal,
                ErrorSeverity::Error,
                true,
                "llm_other",
                message,
            ),
        }
    }
}

/// Classify an error message string into an ErrorCategory.
///
/// Uses heuristic keyword matching on the lowercased message.
/// This is a replacement for ad-hoc string matching in callers.
#[must_use]
pub fn classify_error_message(message: &str) -> ErrorCategory {
    let lower = message.to_lowercase();

    if lower.contains("maximum context length")
        || lower.contains("context length")
        || lower.contains("context_length")
        || lower.contains("prompt is too long")
        || (lower.contains("requested") && lower.contains("tokens") && lower.contains("maximum"))
        || lower.contains("context window")
    {
        return ErrorCategory::InvalidInput;
    }
    if lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("429")
        || lower.contains("quota")
    {
        return ErrorCategory::RateLimit;
    }
    if lower.contains("timeout") || lower.contains("timed out") {
        return ErrorCategory::Timeout;
    }
    if lower.contains("authentication")
        || lower.contains("auth failed")
        || lower.contains("auth error")
        || lower.contains("unauthorized")
        || lower.contains("api key")
        || lower.contains("invalid key")
        || lower.contains("invalid token")
        || lower.contains("bearer token")
    {
        return ErrorCategory::Authentication;
    }
    if lower.contains("authorization")
        || lower.contains("permission")
        || lower.contains("forbidden")
        || lower.contains("denied")
    {
        return ErrorCategory::Authorization;
    }
    if lower.contains("network")
        || lower.contains("connection")
        || lower.contains("dns")
        || lower.contains("stream read error")
        || lower.contains("error decoding response body")
        || lower.contains("chunk decode error")
        || lower.contains("body decode")
        || lower.contains("temporarily unavailable")
        || lower.contains(" 502 ")
        || lower.contains(" 503 ")
        || lower.contains(" 504 ")
        || lower.starts_with("502 ")
        || lower.starts_with("503 ")
        || lower.starts_with("504 ")
        || lower.ends_with(" 502")
        || lower.ends_with(" 503")
        || lower.ends_with(" 504")
        || lower == "502"
        || lower == "503"
        || lower == "504"
    {
        return ErrorCategory::Network;
    }
    if lower.contains("parse") || lower.contains("syntax") || lower.contains("malformed") {
        return ErrorCategory::Parse;
    }
    if lower.contains("not found")
        || lower.contains("unavailable")
        || lower.contains("not available")
    {
        return ErrorCategory::State;
    }
    if lower.contains("tool") {
        return ErrorCategory::Tool;
    }

    ErrorCategory::Internal
}

/// GEN_RELIABILITY_FIX A3: stable structured error-code token for the client.
///
/// The client (ThinkAI Kids desktop/mobile) parses the leading `[CODE]` token
/// off `run_exec`'s failure message to route the user-facing copy precisely —
/// e.g. `[HTTP_402]` becomes a quota/subscription prompt, not a generic
/// "network is slow". This is a low-coupling string contract; the original
/// English detail is preserved after the token.
///
/// Codes (see spec):
///   `[NET_STALL]`    idle / socket read timeout — connection alive-but-mute
///   `[NET_RESET]`    connection reset / dropped mid-stream
///   `[NET_CONNECT]`  could not connect / DNS resolution failed
///   `[TLS]`          TLS handshake failure
///   `[HTTP_401]`     unauthenticated (re-auth)
///   `[HTTP_402]`     payment required (quota / subscription)
///   `[HTTP_429]`     rate limited
///   `[HTTP_5xx]`     upstream server error
///   `[UPSTREAM_EMPTY]` upstream produced no content
///   `[PARSE]`        malformed / unparseable response
///   `[CANCELLED]`    user / caller cancelled
///
/// The message strings we classify here originate in `chat.rs`
/// (`"Stream read error: {reqwest_err}"`, `"SSE stream idle timeout ..."`,
/// `"SSE stream request failed: HTTP {status}: ..."`) and in `turn_loop.rs`
/// (`StreamError::Stall` / transparent-retry failures). reqwest's
/// `is_timeout()`/`is_connect()`/`status()` info is embedded in these strings
/// by the time it reaches the boundary, so we match on the message — the same
/// approach `classify_error_message` already uses. Because HTTP-status and
/// reset phrasings are checked before the generic timeout/connect keywords,
/// the more specific code always wins.
#[must_use]
pub fn run_exec_error_code(message: &str) -> &'static str {
    let lower = message.to_lowercase();

    // Cancellation first — never a network failure, must not be retried or
    // mis-surfaced as an error to the child.
    if lower.contains("cancel") || lower.contains("interrupted") || lower.contains("aborted") {
        return "[CANCELLED]";
    }

    // Explicit HTTP status codes (from `SSE stream request failed: HTTP NNN`
    // and provider error bodies). Check the exact billing/auth codes before
    // the broad 5xx bucket and before the generic keyword scans, because a
    // 402 must never be swallowed as "network slow".
    if http_status_present(&lower, 401) || lower.contains("unauthorized") {
        return "[HTTP_401]";
    }
    if http_status_present(&lower, 402)
        || lower.contains("payment required")
        || lower.contains("insufficient")
        || lower.contains("quota")
    {
        return "[HTTP_402]";
    }
    if http_status_present(&lower, 429)
        || lower.contains("rate limit")
        || lower.contains("too many requests")
    {
        return "[HTTP_429]";
    }
    if http_status_present(&lower, 500)
        || http_status_present(&lower, 502)
        || http_status_present(&lower, 503)
        || http_status_present(&lower, 504)
        || lower.contains("bad gateway")
        || lower.contains("service unavailable")
        || lower.contains("gateway timeout")
        || lower.contains("internal server error")
    {
        return "[HTTP_5xx]";
    }

    // TLS handshake failures — distinct from a plain connect failure so the
    // client can tell "clock/cert problem" from "no route".
    if lower.contains("tls")
        || lower.contains("handshake")
        || lower.contains("certificate")
        || lower.contains("ssl")
    {
        return "[TLS]";
    }

    // Connection reset / dropped mid-stream (the mobile root cause). Reqwest
    // surfaces these as "connection reset", "broken pipe", "stream read
    // error", or a body-decode failure after bytes were flowing.
    if lower.contains("reset")
        || lower.contains("broken pipe")
        || lower.contains("connection closed")
        || lower.contains("connection aborted")
        || lower.contains("stream read error")
        || lower.contains("error decoding response body")
        || lower.contains("chunk decode error")
        || lower.contains("body decode")
        || lower.contains("incomplete message")
        || lower.contains("unexpected eof")
    {
        return "[NET_RESET]";
    }

    // Idle / read stall — connection is nominally alive but no data arrives.
    if lower.contains("idle timeout")
        || lower.contains("stream stalled")
        || lower.contains("no data received")
        || lower.contains("read timed out")
        || lower.contains("timed out")
        || lower.contains("timeout")
    {
        return "[NET_STALL]";
    }

    // Could not establish the connection at all (DNS / route / refused).
    if lower.contains("dns")
        || lower.contains("failed to lookup")
        || lower.contains("name resolution")
        || lower.contains("connection refused")
        || lower.contains("could not connect")
        || lower.contains("connect error")
        || lower.contains("network is unreachable")
        || lower.contains("no route to host")
        || lower.contains("unreachable")
    {
        return "[NET_CONNECT]";
    }

    if lower.contains("parse") || lower.contains("malformed") || lower.contains("syntax") {
        return "[PARSE]";
    }

    // Empty upstream: stream ended without producing any content.
    if lower.contains("empty") || lower.contains("no content") || lower.contains("no output") {
        return "[UPSTREAM_EMPTY]";
    }

    // Fall back on the broad category classifier so at least a network-shaped
    // failure gets a network code rather than nothing.
    match classify_error_message(message) {
        ErrorCategory::Network => "[NET_RESET]",
        ErrorCategory::Timeout => "[NET_STALL]",
        ErrorCategory::RateLimit => "[HTTP_429]",
        ErrorCategory::Authentication => "[HTTP_401]",
        ErrorCategory::Parse => "[PARSE]",
        _ => "",
    }
}

/// Prefix `message` with its A3 error-code token when one applies and the
/// message isn't already tagged. Returns the message unchanged when no code
/// maps (so unclassified internal errors keep their original text) or when a
/// `[CODE]` prefix is already present (idempotent — safe to call at multiple
/// boundaries).
#[must_use]
pub fn with_error_code(message: impl Into<String>) -> String {
    let message = message.into();
    if message.starts_with('[') {
        // Already tagged (e.g. re-tagged at an inner boundary) — don't
        // double-prefix.
        return message;
    }
    let code = run_exec_error_code(&message);
    if code.is_empty() {
        message
    } else {
        format!("{code} {message}")
    }
}

/// True when `lower` contains `status` as a standalone HTTP status number
/// (bounded by non-digits) so "402" matches but "14025" does not.
fn http_status_present(lower: &str, status: u16) -> bool {
    let needle = status.to_string();
    let bytes = lower.as_bytes();
    let mut start = 0;
    while let Some(rel) = lower[start..].find(&needle) {
        let idx = start + rel;
        let before_ok = idx == 0 || !bytes[idx - 1].is_ascii_digit();
        let after = idx + needle.len();
        let after_ok = after >= bytes.len() || !bytes[after].is_ascii_digit();
        if before_ok && after_ok {
            return true;
        }
        start = idx + needle.len();
    }
    false
}

impl From<ToolError> for ErrorEnvelope {
    fn from(value: ToolError) -> Self {
        match value {
            ToolError::InvalidInput { message } => Self::new(
                ErrorCategory::InvalidInput,
                ErrorSeverity::Error,
                false,
                "tool_invalid_input",
                message,
            ),
            ToolError::MissingField { field } => Self::new(
                ErrorCategory::InvalidInput,
                ErrorSeverity::Error,
                false,
                "tool_missing_field",
                format!("Missing required field: {field}"),
            ),
            ToolError::PathEscape { path } => Self::new(
                ErrorCategory::Authorization,
                ErrorSeverity::Error,
                false,
                "tool_path_escape",
                format!("Path escapes workspace: {}", path.display()),
            ),
            ToolError::ExecutionFailed { message } => Self::new(
                ErrorCategory::Tool,
                ErrorSeverity::Error,
                true,
                "tool_execution_failed",
                message,
            ),
            ToolError::Timeout { seconds } => Self::new(
                ErrorCategory::Timeout,
                ErrorSeverity::Warning,
                true,
                "tool_timeout",
                format!("Tool timed out after {seconds}s"),
            ),
            ToolError::NotAvailable { message } => Self::new(
                ErrorCategory::State,
                ErrorSeverity::Error,
                false,
                "tool_not_available",
                message,
            ),
            ToolError::PermissionDenied { message } => Self::new(
                ErrorCategory::Authorization,
                ErrorSeverity::Error,
                false,
                "tool_permission_denied",
                message,
            ),
        }
    }
}

/// Stream‑level error discriminated by origin.
///
/// Each variant maps to an `ErrorCategory` so the UI can render
/// stream‑specific icons or formatting. Wired into engine.rs at the three
/// stream guard sites (chunk timeout, max-bytes overflow, max-duration).
#[derive(Debug, Clone)]
pub enum StreamError {
    /// Stream stalled — no chunk received within the idle timeout.
    Stall { timeout_secs: u64 },
    /// Stream exceeded content size limit.
    Overflow { limit_bytes: usize },
    /// Stream exceeded wall‑clock duration limit.
    DurationLimit { limit_secs: u64 },
}

impl StreamError {
    /// Convert directly into an `ErrorEnvelope` for emission on the engine
    /// event channel. Stalls are warning-severity and recoverable; size and
    /// duration limits are errors (the user must restart the turn).
    #[must_use]
    pub fn into_envelope(self) -> ErrorEnvelope {
        match self {
            Self::Stall { timeout_secs } => ErrorEnvelope::new(
                ErrorCategory::Timeout,
                ErrorSeverity::Warning,
                true,
                "stream_stall",
                format!("Stream stalled: no data received for {timeout_secs}s, closing stream"),
            ),
            Self::Overflow { limit_bytes } => ErrorEnvelope::new(
                ErrorCategory::Internal,
                ErrorSeverity::Error,
                true,
                "stream_overflow",
                format!("Stream exceeded maximum content size of {limit_bytes} bytes, closing"),
            ),
            Self::DurationLimit { limit_secs } => ErrorEnvelope::new(
                ErrorCategory::Timeout,
                ErrorSeverity::Error,
                true,
                "stream_duration_limit",
                format!("Stream exceeded maximum duration of {limit_secs}s, closing"),
            ),
        }
    }
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stall { timeout_secs } => {
                write!(f, "Stream stalled after {timeout_secs}s idle")
            }
            Self::Overflow { limit_bytes } => {
                write!(f, "Stream exceeded {limit_bytes} bytes limit")
            }
            Self::DurationLimit { limit_secs } => {
                write!(f, "Stream exceeded {limit_secs}s duration limit")
            }
        }
    }
}

impl std::error::Error for StreamError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(msg: &str) -> ErrorCategory {
        classify_error_message(msg)
    }

    #[test]
    fn invalid_input_catches_context_overflow_phrasings() {
        // Provider phrasing varies: DeepSeek/OpenAI/Anthropic/etc each
        // surface context-overflow as a slightly different string.
        // The classifier needs all of them on the same branch.
        for msg in [
            "This model's maximum context length is 1000000 tokens",
            "Error: context_length_exceeded",
            "Your prompt is too long for the current model",
            "You requested 100000 tokens but the maximum is 65536",
            "request exceeds context window",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::InvalidInput,
                "expected InvalidInput for `{msg}`",
            );
        }
    }

    #[test]
    fn rate_limit_catches_429_and_quota_phrasings() {
        for msg in [
            "Rate limit reached for gpt-4",
            "Too Many Requests",
            "HTTP 429 from upstream",
            "Your quota has been exceeded",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::RateLimit,
                "expected RateLimit for `{msg}`",
            );
        }
    }

    #[test]
    fn timeout_catches_both_spellings() {
        assert_eq!(classify("connection timeout"), ErrorCategory::Timeout);
        assert_eq!(
            classify("request timed out after 30s"),
            ErrorCategory::Timeout
        );
    }

    #[test]
    fn network_catches_stream_body_decode_failures() {
        for msg in [
            "Warn Stream read error: error decoding response body",
            "Stream read error: error decoding response body",
            "chunk decode error",
            "provider body decode failed mid-stream",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Network,
                "expected Network for `{msg}`",
            );
        }
    }

    #[test]
    fn authentication_beats_authorization_when_api_key_phrasing_is_used() {
        // "api key" landing on Authentication (not Authorization) keeps
        // the operator-facing message correct: the user needs to fix
        // their key, not their permissions.
        for msg in [
            "Invalid API key provided",
            "Authentication failed",
            "401 Unauthorized",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Authentication,
                "expected Authentication for `{msg}`",
            );
        }
    }

    #[test]
    fn llm_auth_error_envelope_renders_context_without_secret() {
        let api_key = "tp-secret-token-plan-value";
        let env = ErrorEnvelope::from(LlmError::from_http_response_with_request_context(
            401,
            &format!("Invalid API Key: {api_key}"),
            Some("Xiaomi MiMo"),
            Some("https://token-plan-sgp.xiaomimimo.com/v1"),
            Some("mimo-v2.5"),
            Some("env"),
            Some(api_key),
        ));

        assert_eq!(env.category, ErrorCategory::Authentication);
        assert_eq!(env.severity, ErrorSeverity::Critical);
        assert!(!env.recoverable);
        assert!(env.message.contains("provider: Xiaomi MiMo"));
        assert!(
            env.message
                .contains("base URL authority: token-plan-sgp.xiaomimimo.com")
        );
        assert!(env.message.contains("model: mimo-v2.5"));
        assert!(env.message.contains("key source: env"));
        assert!(env.message.contains("key fingerprint: tp-... (len=26)"));
        assert!(env.message.contains("key type: Xiaomi MiMo Token Plan key"));
        assert!(!env.message.contains(api_key));
        assert!(!env.message.contains("secret-token-plan-value"));
    }

    #[test]
    fn authorization_catches_forbidden_and_denied() {
        for msg in [
            "403 Forbidden",
            "Authorization failed: Arcee AI API returned Cloudflare Access Denied",
            "Permission denied for resource",
            "Tool 'edit_file' denied by user",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Authorization,
                "expected Authorization for `{msg}`",
            );
        }
    }

    #[test]
    fn network_catches_dns_connection_5xx() {
        for msg in [
            "Network is unreachable",
            "Connection reset by peer",
            "DNS resolution failed for api.deepseek.com",
            "503 Service Unavailable",
            "Upstream returned 502 Bad Gateway",
            "Service temporarily unavailable",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Network,
                "expected Network for `{msg}`",
            );
        }
        // Edge-case precedence: "504 Gateway Timeout" mentions both
        // a 504 status code AND the word "timeout". The classifier
        // picks Timeout, which is correct — the operator-actionable
        // category for a 504 is "wait and retry" (Timeout semantics)
        // rather than "DNS / connection broken" (Network semantics).
        assert_eq!(
            classify("504 Gateway Timeout"),
            ErrorCategory::Timeout,
            "504 with the literal word `timeout` resolves as Timeout, not Network"
        );
    }

    #[test]
    fn parse_catches_syntax_and_malformed_json() {
        for msg in [
            "Failed to parse response JSON",
            "Syntax error in tool arguments",
            "Malformed event from stream",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Parse,
                "expected Parse for `{msg}`",
            );
        }
    }

    #[test]
    fn state_catches_not_found_and_unavailable() {
        for msg in [
            "Session not found",
            "Model is unavailable for this provider",
            "Endpoint not available in this region",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::State,
                "expected State for `{msg}`",
            );
        }
    }

    #[test]
    fn tool_is_a_low_priority_catchall_for_tool_keyword() {
        // The Tool branch is the last keyword check before falling
        // through to Internal. Anything mentioning "tool" that didn't
        // match an earlier category should land here.
        assert_eq!(
            classify("Tool returned non-zero exit status"),
            ErrorCategory::Tool,
        );
    }

    #[test]
    fn unknown_messages_fall_through_to_internal() {
        for msg in [
            "Something exploded",
            "panic at the disco",
            "u-200 something happened",
            "",
        ] {
            assert_eq!(
                classify(msg),
                ErrorCategory::Internal,
                "expected Internal for `{msg}`",
            );
        }
    }

    #[test]
    fn classifier_is_case_insensitive() {
        // The function lowercases internally — every category must
        // match regardless of input casing.
        assert_eq!(classify("RATE LIMIT EXCEEDED"), ErrorCategory::RateLimit);
        assert_eq!(classify("TimeOut"), ErrorCategory::Timeout);
        assert_eq!(classify("UNAUTHORIZED"), ErrorCategory::Authentication);
    }

    #[test]
    fn precedence_invalid_input_beats_tool() {
        // A "context length" tool error should classify as
        // InvalidInput, not Tool — InvalidInput is the more actionable
        // category (the user needs to shorten their prompt; "tool
        // failure" wouldn't tell them that).
        assert_eq!(
            classify("tool returned: maximum context length is 1000000"),
            ErrorCategory::InvalidInput,
        );
    }

    #[test]
    fn precedence_timeout_beats_network() {
        // A timeout that mentions a network call should still classify
        // as Timeout — the retry policy for timeouts is gentler than
        // for outright network failures.
        assert_eq!(
            classify("network call timed out after 30s"),
            ErrorCategory::Timeout,
        );
    }

    #[test]
    fn precedence_rate_limit_beats_authentication() {
        // 429 messages sometimes mention "api" or "auth" tokens, but
        // RateLimit's retry semantics (back off + retry) are what the
        // operator actually wants.
        assert_eq!(
            classify("Rate limit on your API quota exceeded"),
            ErrorCategory::RateLimit,
        );
    }

    #[test]
    fn classifier_handles_unicode_safely() {
        // Unicode shouldn't trip the lowercase step or the keyword
        // scan — Chinese/Japanese error messages from
        // OpenAI-compatible providers go through the same path.
        assert_eq!(
            classify("\u{8d85}\u{51fa}\u{6700}\u{5927}\u{4e0a}\u{4e0b}\u{6587} context length"),
            ErrorCategory::InvalidInput,
        );
        // Pure-Chinese messages with no keyword match land on Internal.
        assert_eq!(
            classify("\u{4e0d}\u{77e5}\u{9053}\u{600e}\u{4e48}\u{56de}\u{4e8b}"),
            ErrorCategory::Internal,
        );
    }

    #[test]
    fn error_envelope_display_includes_severity_code_message() {
        let env = ErrorEnvelope::new(
            ErrorCategory::Network,
            ErrorSeverity::Warning,
            true,
            "net_transient",
            "DNS resolution failed",
        );
        assert_eq!(
            format!("{env}"),
            "[warning] net_transient: DNS resolution failed"
        );
    }

    // === GEN_RELIABILITY_FIX A3: structured error-code token tests ===

    #[test]
    fn a3_maps_transport_failures_to_net_codes() {
        // These are the exact phrasings chat.rs / turn_loop.rs produce.
        assert_eq!(
            run_exec_error_code("SSE stream idle timeout after 120s — no data received"),
            "[NET_STALL]"
        );
        assert_eq!(
            run_exec_error_code("Stream stalled: no data received for 120s, closing stream"),
            "[NET_STALL]"
        );
        assert_eq!(
            run_exec_error_code("Stream read error: connection reset by peer"),
            "[NET_RESET]"
        );
        assert_eq!(
            run_exec_error_code("Stream read error: error decoding response body"),
            "[NET_RESET]"
        );
        assert_eq!(
            run_exec_error_code("error sending request: connection refused"),
            "[NET_CONNECT]"
        );
        assert_eq!(
            run_exec_error_code("dns error: failed to lookup address information"),
            "[NET_CONNECT]"
        );
    }

    #[test]
    fn a3_http_status_codes_beat_generic_network() {
        // A 402 must NEVER be swallowed as a network error — the client
        // shows a quota/subscription prompt, not "network is slow".
        assert_eq!(
            run_exec_error_code("SSE stream request failed: HTTP 402: insufficient quota"),
            "[HTTP_402]"
        );
        assert_eq!(
            run_exec_error_code("SSE stream request failed: HTTP 401: invalid token"),
            "[HTTP_401]"
        );
        assert_eq!(
            run_exec_error_code("SSE stream request failed: HTTP 429: rate limit reached"),
            "[HTTP_429]"
        );
        assert_eq!(
            run_exec_error_code("SSE stream request failed: HTTP 503: service unavailable"),
            "[HTTP_5xx]"
        );
        assert_eq!(
            run_exec_error_code("SSE stream request failed: HTTP 500: internal server error"),
            "[HTTP_5xx]"
        );
    }

    #[test]
    fn a3_http_status_matcher_is_digit_bounded() {
        // "14025" must not be read as a 402 or 429.
        assert_eq!(run_exec_error_code("processed 14025 tokens then died"), "");
        // But a bare "402" bounded by non-digits does match.
        assert!(http_status_present("http 402 payment", 402));
        assert!(!http_status_present("token 4025 used", 402));
    }

    #[test]
    fn a3_tls_and_cancel_and_parse() {
        assert_eq!(
            run_exec_error_code("tls handshake failed: certificate expired"),
            "[TLS]"
        );
        assert_eq!(run_exec_error_code("Request cancelled by user"), "[CANCELLED]");
        assert_eq!(run_exec_error_code("turn interrupted"), "[CANCELLED]");
        assert_eq!(
            run_exec_error_code("failed to parse SSE data frame: malformed json"),
            "[PARSE]"
        );
    }

    #[test]
    fn a3_with_error_code_is_idempotent_and_preserves_detail() {
        let coded = with_error_code("Stream read error: connection reset by peer");
        assert_eq!(coded, "[NET_RESET] Stream read error: connection reset by peer");
        // Calling again must not double-prefix.
        assert_eq!(with_error_code(coded.clone()), coded);
    }

    #[test]
    fn a3_unclassified_internal_error_keeps_original_text() {
        // No code maps → message returned unchanged (don't mislabel).
        assert_eq!(with_error_code("something weird happened"), "something weird happened");
    }

    #[test]
    fn error_category_display_round_trips_via_snake_case() {
        // The snake_case labels are what crosses the wire / hits logs;
        // pin them so a future rename doesn't silently shift consumer
        // contracts.
        assert_eq!(format!("{}", ErrorCategory::Network), "network");
        assert_eq!(format!("{}", ErrorCategory::RateLimit), "rate_limit");
        assert_eq!(format!("{}", ErrorCategory::InvalidInput), "invalid_input");
        assert_eq!(format!("{}", ErrorSeverity::Critical), "critical");
    }
}
