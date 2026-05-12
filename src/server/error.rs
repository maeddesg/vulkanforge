//! Error representation for the v0.4 API server.
//!
//! Three layers:
//!
//! 1. [`ApiError`] — the internal enum produced by handlers and
//!    validators. Carries enough information to derive both an HTTP
//!    status code and the OpenAI-shaped JSON body.
//! 2. [`ErrorResponse`] / [`ErrorDetail`] — the wire-format structs
//!    that serialise to OpenAI's documented error shape.
//! 3. [`axum::response::IntoResponse`] impl on `ApiError` — bridges
//!    the two by emitting `(StatusCode, Json<ErrorResponse>)`.
//!
//! Spec references: §3.6 (wire shape) and §8.1 (mapping table).
//! Decision §2 (Merge-Sprint): context-length-exceeded is **400**,
//! not 422 — OpenAI compatibility wins over HTTP-semantic purity.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Internal error enum. Handlers return `Result<T, ApiError>`; axum
/// auto-converts to the proper HTTP response via [`IntoResponse`].
#[derive(Debug, Clone)]
pub enum ApiError {
    /// 400 — generic schema or semantic validation failure. The
    /// `code` argument propagates to the JSON body so clients can
    /// distinguish sub-cases (`invalid_body`, `invalid_role`,
    /// `unsupported_role`, `unsupported_content_type`, `unsupported_n`,
    /// `no_user_message`, `last_message_not_user`, …).
    InvalidRequest {
        message: String,
        code: &'static str,
        param: Option<String>,
    },

    /// 400 with `code: "context_length_exceeded"`. Held as its own
    /// variant because it's the single most-common error and
    /// callers usually compute the message from token counts.
    ContextLengthExceeded {
        prompt_tokens: u32,
        max_tokens: u32,
        context_window: u32,
    },

    /// 429 — concurrent request rejected by the request-permit
    /// semaphore (single-slot in v0.4).
    ServerBusy,

    /// 500 — internal failures (tokenizer, GPU, IO).
    InternalError {
        message: String,
        code: &'static str,
    },

    /// 503 — server is up but the model hasn't finished loading
    /// yet (async-load path planned for Sprint 4).
    ModelLoading,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>, code: &'static str) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            code,
            param: None,
        }
    }

    pub fn invalid_with_param(
        message: impl Into<String>,
        code: &'static str,
        param: impl Into<String>,
    ) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            code,
            param: Some(param.into()),
        }
    }

    pub fn internal(message: impl Into<String>, code: &'static str) -> Self {
        Self::InternalError {
            message: message.into(),
            code,
        }
    }

    /// HTTP status code per §8.1.
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::InvalidRequest { .. } | ApiError::ContextLengthExceeded { .. } => {
                StatusCode::BAD_REQUEST
            }
            ApiError::ServerBusy => StatusCode::TOO_MANY_REQUESTS,
            ApiError::InternalError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::ModelLoading => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// OpenAI `error.type` value per §3.6.
    pub fn error_type(&self) -> &'static str {
        match self {
            ApiError::InvalidRequest { .. } | ApiError::ContextLengthExceeded { .. } => {
                "invalid_request_error"
            }
            ApiError::ServerBusy => "rate_limit_exceeded",
            ApiError::InternalError { .. } => "server_error",
            ApiError::ModelLoading => "engine_unavailable",
        }
    }

    /// OpenAI `error.code` value per §8.1.
    pub fn code(&self) -> &'static str {
        match self {
            ApiError::InvalidRequest { code, .. } => code,
            ApiError::ContextLengthExceeded { .. } => "context_length_exceeded",
            ApiError::ServerBusy => "concurrent_limit",
            ApiError::InternalError { code, .. } => code,
            ApiError::ModelLoading => "model_loading",
        }
    }

    /// Human-readable message that goes into `error.message`.
    pub fn message(&self) -> String {
        match self {
            ApiError::InvalidRequest { message, .. } | ApiError::InternalError { message, .. } => {
                message.clone()
            }
            ApiError::ContextLengthExceeded {
                prompt_tokens,
                max_tokens,
                context_window,
            } => format!(
                "Prompt tokens ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds context length ({context_window})"
            ),
            ApiError::ServerBusy => {
                "Server busy: another request is in progress".into()
            }
            ApiError::ModelLoading => "Model is still loading".into(),
        }
    }

    /// Optional `error.param` (which request field caused the
    /// error, where known).
    pub fn param(&self) -> Option<String> {
        match self {
            ApiError::InvalidRequest { param, .. } => param.clone(),
            ApiError::ContextLengthExceeded { .. } => Some("messages".into()),
            _ => None,
        }
    }

    /// Build the wire-format body without consuming `self`.
    pub fn to_response_body(&self) -> ErrorResponse {
        ErrorResponse {
            error: ErrorDetail {
                message: self.message(),
                error_type: self.error_type(),
                param: self.param(),
                code: Some(self.code()),
            },
        }
    }
}

// =========================================================================
// Wire format (§3.6)
// =========================================================================

#[derive(Debug, Serialize, Clone)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, Clone)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}

// =========================================================================
// axum bridge
// =========================================================================

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let is_busy = matches!(self, ApiError::ServerBusy);
        let body = self.to_response_body();
        let mut resp = (status, Json(body)).into_response();
        // Open WebUI honours `Retry-After` and re-tries the request
        // after the given delay instead of surfacing an error to the
        // user. 1 s is enough to bridge the brief permit-release
        // window between a streamed response's `[DONE]` chunk and the
        // spawn_blocking task's drop chain.
        if is_busy {
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        resp
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message(), self.code())
    }
}

impl std::error::Error for ApiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_maps_to_400_with_type_and_code() {
        let e = ApiError::invalid("bad json", "invalid_body");
        assert_eq!(e.status(), StatusCode::BAD_REQUEST);
        assert_eq!(e.error_type(), "invalid_request_error");
        assert_eq!(e.code(), "invalid_body");
        assert!(e.param().is_none());
    }

    #[test]
    fn context_overflow_maps_to_400_not_422() {
        // Merge-Sprint decision #2: 400, not 422.
        let e = ApiError::ContextLengthExceeded {
            prompt_tokens: 9000,
            max_tokens: 200,
            context_window: 2048,
        };
        assert_eq!(e.status(), StatusCode::BAD_REQUEST);
        assert_eq!(e.code(), "context_length_exceeded");
        assert_eq!(e.error_type(), "invalid_request_error");
        assert_eq!(e.param().as_deref(), Some("messages"));
        assert!(
            e.message().contains("9000") && e.message().contains("2048"),
            "message should embed counts; got: {}",
            e.message()
        );
    }

    #[test]
    fn server_busy_maps_to_429_rate_limit() {
        let e = ApiError::ServerBusy;
        assert_eq!(e.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(e.error_type(), "rate_limit_exceeded");
        assert_eq!(e.code(), "concurrent_limit");
    }

    #[test]
    fn model_loading_maps_to_503_engine_unavailable() {
        let e = ApiError::ModelLoading;
        assert_eq!(e.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(e.error_type(), "engine_unavailable");
        assert_eq!(e.code(), "model_loading");
    }

    #[test]
    fn internal_error_maps_to_500_server_error() {
        let e = ApiError::internal("vk crash", "gpu_error");
        assert_eq!(e.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(e.error_type(), "server_error");
        assert_eq!(e.code(), "gpu_error");
    }

    #[test]
    fn wire_body_matches_openai_shape() {
        let e = ApiError::invalid_with_param("Bad role 'foo'", "invalid_role", "messages[0].role");
        let body = e.to_response_body();
        let v = serde_json::to_value(&body).unwrap();

        assert_eq!(v["error"]["message"], "Bad role 'foo'");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "messages[0].role");
        assert_eq!(v["error"]["code"], "invalid_role");
    }

    #[test]
    fn into_response_emits_correct_status_and_json_body() {
        // axum::IntoResponse smoke test — we don't actually fire
        // an HTTP request; we just verify the converted Response
        // carries the right status code.
        let e = ApiError::ServerBusy;
        let resp = e.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn server_busy_attaches_retry_after_header() {
        // Open WebUI compat: 429 must carry Retry-After so the
        // client backs off instead of surfacing a user-visible error.
        let resp = ApiError::ServerBusy.into_response();
        let v = resp
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .expect("Retry-After header should be set on 429");
        assert_eq!(v.to_str().unwrap(), "1");
    }

    #[test]
    fn non_busy_errors_have_no_retry_after_header() {
        // Other 4xx/5xx must NOT carry Retry-After — only the
        // concurrent-limit case is transient and retry-friendly.
        for e in [
            ApiError::invalid("x", "invalid_body"),
            ApiError::ContextLengthExceeded {
                prompt_tokens: 1,
                max_tokens: 1,
                context_window: 1,
            },
            ApiError::internal("x", "internal_error"),
            ApiError::ModelLoading,
        ] {
            let resp = e.into_response();
            assert!(
                resp.headers().get(axum::http::header::RETRY_AFTER).is_none(),
                "Retry-After must not appear on non-ServerBusy errors"
            );
        }
    }

    #[test]
    fn display_uses_message_and_code() {
        let e = ApiError::invalid("boom", "invalid_body");
        let s = e.to_string();
        assert!(s.contains("boom"));
        assert!(s.contains("invalid_body"));
    }
}
