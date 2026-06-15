//! Streamable-HTTP MCP client (non-streaming JSON-RPC subset).
//!
//! Implements the client side of the Model Context Protocol's Streamable HTTP
//! transport sufficient for stateless servers: every JSON-RPC request is sent as
//! a single HTTP `POST`, and the server replies with a single JSON-RPC response
//! object (`Content-Type: application/json`). Notifications receive HTTP `202`
//! with no body. A `Mcp-Session-Id` response header, if present, is echoed back
//! on subsequent requests, but is not required.

use std::collections::BTreeMap;
use std::io;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::mcp_client::McpRemoteTransport;
use crate::mcp_stdio::{
    JsonRpcId, JsonRpcRequest, JsonRpcResponse, McpInitializeParams, McpInitializeResult,
    McpListToolsParams, McpListToolsResult, McpToolCallParams, McpToolCallResult,
};

const MCP_SESSION_ID_HEADER: &str = "mcp-session-id";
const HTTP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Async Streamable-HTTP MCP client bound to a single server endpoint.
#[derive(Debug)]
pub struct McpHttpClient {
    client: reqwest::Client,
    url: String,
    headers: BTreeMap<String, String>,
    session_id: Option<String>,
}

impl McpHttpClient {
    /// Build a client from a remote transport config (`url` + configured headers).
    pub fn from_transport(transport: &McpRemoteTransport) -> io::Result<Self> {
        let mut builder = reqwest::Client::builder();
        // Opt-in (env-driven) TLS bypass for self-signed origin certs in trusted
        // local deployments; default off (secure).
        if insecure_tls_enabled() {
            builder = builder.danger_accept_invalid_certs(true);
        }
        let client = builder
            .build()
            .map_err(|error| io::Error::other(format!("failed to build HTTP client: {error}")))?;
        Ok(Self {
            client,
            url: transport.url.clone(),
            headers: transport.headers.clone(),
            session_id: None,
        })
    }

    fn request_headers(&self) -> io::Result<HeaderMap> {
        let mut map = HeaderMap::new();
        map.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        map.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        if let Some(session_id) = &self.session_id {
            if let (Ok(name), Ok(value)) = (
                HeaderName::from_bytes(MCP_SESSION_ID_HEADER.as_bytes()),
                HeaderValue::from_str(session_id),
            ) {
                map.insert(name, value);
            }
        }
        for (key, value) in &self.headers {
            let name = HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid MCP header name `{key}`: {error}"),
                )
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid MCP header value for `{key}`: {error}"),
                )
            })?;
            map.insert(name, header_value);
        }
        Ok(map)
    }

    /// Send a JSON-RPC request that expects a JSON-RPC response object.
    async fn send_request<TParams: Serialize, TResult: DeserializeOwned>(
        &mut self,
        request: &JsonRpcRequest<TParams>,
        method: &str,
    ) -> io::Result<JsonRpcResponse<TResult>> {
        let headers = self.request_headers()?;
        let response = self
            .client
            .post(&self.url)
            .headers(headers)
            .json(request)
            .send()
            .await
            .map_err(|error| io::Error::other(format!("HTTP request for {method} failed: {error}")))?;

        if let Some(session_id) = response
            .headers()
            .get(MCP_SESSION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            self.session_id = Some(session_id.to_string());
        }

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| io::Error::other(format!("HTTP body read for {method} failed: {error}")))?;

        if !status.is_success() {
            return Err(io::Error::other(format!(
                "HTTP {status} from MCP server for {method}: {body}"
            )));
        }

        let parsed: JsonRpcResponse<TResult> = serde_json::from_str(&body)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

        if parsed.jsonrpc != "2.0" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "MCP response for {method} used unsupported jsonrpc version `{}`",
                    parsed.jsonrpc
                ),
            ));
        }

        if parsed.id != request.id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "MCP response for {method} used mismatched id: expected {:?}, got {:?}",
                    request.id, parsed.id
                ),
            ));
        }

        Ok(parsed)
    }

    /// Send a JSON-RPC notification (no `id`); expects HTTP `202`/no body.
    async fn send_notification(&mut self, method: &str) -> io::Result<()> {
        let headers = self.request_headers()?;
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        let response = self
            .client
            .post(&self.url)
            .headers(headers)
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                io::Error::other(format!("HTTP notification {method} failed: {error}"))
            })?;

        if let Some(session_id) = response
            .headers()
            .get(MCP_SESSION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            self.session_id = Some(session_id.to_string());
        }

        // Notifications get 202 (Accepted) with no body, but tolerate any 2xx.
        if !response.status().is_success() {
            return Err(io::Error::other(format!(
                "HTTP {} from MCP server for notification {method}",
                response.status()
            )));
        }
        Ok(())
    }

    /// Perform the `initialize` handshake and send `notifications/initialized`.
    pub async fn initialize(
        &mut self,
        id: JsonRpcId,
        params: McpInitializeParams,
    ) -> io::Result<JsonRpcResponse<McpInitializeResult>> {
        let request = JsonRpcRequest::new(id, "initialize", Some(params));
        let response = self.send_request(&request, "initialize").await?;
        if response.error.is_none() {
            self.send_notification("notifications/initialized").await?;
        }
        Ok(response)
    }

    /// List tools exposed by the server.
    pub async fn list_tools(
        &mut self,
        id: JsonRpcId,
        params: Option<McpListToolsParams>,
    ) -> io::Result<JsonRpcResponse<McpListToolsResult>> {
        let request = JsonRpcRequest::new(id, "tools/list", params);
        self.send_request(&request, "tools/list").await
    }

    /// Invoke a tool by raw name.
    pub async fn call_tool(
        &mut self,
        id: JsonRpcId,
        params: McpToolCallParams,
    ) -> io::Result<JsonRpcResponse<McpToolCallResult>> {
        let request = JsonRpcRequest::new(id, "tools/call", Some(params));
        self.send_request(&request, "tools/call").await
    }
}

/// Whether `CLAW_MCP_INSECURE_TLS` opts into skipping TLS cert verification.
fn insecure_tls_enabled() -> bool {
    std::env::var("CLAW_MCP_INSECURE_TLS")
        .map(|value| {
            let value = value.trim();
            value == "1" || value.eq_ignore_ascii_case("true")
        })
        .unwrap_or(false)
}

/// Default `initialize` params used by the HTTP transport (protocol 2025-06-18).
pub fn default_http_initialize_params() -> McpInitializeParams {
    McpInitializeParams {
        protocol_version: HTTP_PROTOCOL_VERSION.to_string(),
        capabilities: JsonValue::Object(serde_json::Map::new()),
        client_info: crate::mcp_stdio::McpInitializeClientInfo {
            name: "clawcode".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::mcp_client::{McpClientAuth, McpRemoteTransport};
    use crate::mcp_stdio::{JsonRpcId, McpToolCallParams};

    use super::{default_http_initialize_params, McpHttpClient};

    /// Hermetic mock Streamable-HTTP MCP server.
    ///
    /// Reads one HTTP request, parses the JSON-RPC body, and answers
    /// initialize/tools-list/tools-call (202 for notifications). Records the
    /// `Cookie` header it last observed so the test can assert header passthrough.
    async fn spawn_mock_server(
        observed_cookie: Arc<Mutex<Option<String>>>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let url = format!("http://{addr}/mcp");

        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let cookie_sink = Arc::clone(&observed_cookie);
                tokio::spawn(async move {
                    loop {
                        // Read until end of headers, then read the body by Content-Length.
                        let mut buf = Vec::new();
                        let mut tmp = [0u8; 1024];
                        let header_end = loop {
                            match socket.read(&mut tmp).await {
                                Ok(0) => return,
                                Ok(n) => {
                                    buf.extend_from_slice(&tmp[..n]);
                                    if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                                        break pos;
                                    }
                                }
                                Err(_) => return,
                            }
                        };
                        let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
                        let mut content_length = 0usize;
                        for line in header_text.lines() {
                            let lower = line.to_ascii_lowercase();
                            if let Some(rest) = lower.strip_prefix("content-length:") {
                                content_length = rest.trim().parse().unwrap_or(0);
                            }
                            // reqwest normalizes header names to lowercase on the wire.
                            if lower.starts_with("cookie:") {
                                let value = line.splitn(2, ':').nth(1).unwrap_or("").trim();
                                *cookie_sink.lock().unwrap() = Some(value.to_string());
                            }
                        }
                        let mut body = buf[header_end + 4..].to_vec();
                        while body.len() < content_length {
                            match socket.read(&mut tmp).await {
                                Ok(0) => break,
                                Ok(n) => body.extend_from_slice(&tmp[..n]),
                                Err(_) => return,
                            }
                        }
                        let request: serde_json::Value =
                            serde_json::from_slice(&body).unwrap_or(json!({}));
                        let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");
                        let id = request.get("id").cloned();

                        let response_body = match method {
                            "initialize" => Some(json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "protocolVersion": "2025-06-18",
                                    "capabilities": {"tools": {}},
                                    "serverInfo": {"name": "mock-http", "version": "0.1.0"}
                                }
                            })),
                            "tools/list" => Some(json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "tools": [{
                                        "name": "ping",
                                        "description": "Ping the server",
                                        "inputSchema": {"type": "object"}
                                    }]
                                }
                            })),
                            "tools/call" => Some(json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "content": [{"type": "text", "text": "pong"}],
                                    "isError": false
                                }
                            })),
                            // notifications/initialized and others: 202, no body.
                            _ => None,
                        };

                        let raw = match response_body {
                            Some(value) => {
                                let payload = serde_json::to_vec(&value).unwrap();
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                                    payload.len()
                                )
                                .into_bytes()
                                .into_iter()
                                .chain(payload)
                                .collect::<Vec<u8>>()
                            }
                            None => b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n".to_vec(),
                        };
                        if socket.write_all(&raw).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });

        (url, handle)
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    #[tokio::test]
    async fn http_client_initializes_lists_and_calls_tools() {
        let observed_cookie = Arc::new(Mutex::new(None));
        let (url, _handle) = spawn_mock_server(Arc::clone(&observed_cookie)).await;

        let transport = McpRemoteTransport {
            url,
            headers: BTreeMap::from([(
                "Cookie".to_string(),
                "knowmax.session=abc123".to_string(),
            )]),
            headers_helper: None,
            auth: McpClientAuth::None,
        };

        let mut client = McpHttpClient::from_transport(&transport).expect("client");

        let init = client
            .initialize(JsonRpcId::Number(1), default_http_initialize_params())
            .await
            .expect("initialize");
        assert!(init.error.is_none());
        assert_eq!(
            init.result.expect("init result").server_info.name,
            "mock-http"
        );

        let tools = client
            .list_tools(JsonRpcId::Number(2), None)
            .await
            .expect("tools/list");
        let tools = tools.result.expect("tools result").tools;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "ping");

        let call = client
            .call_tool(
                JsonRpcId::Number(3),
                McpToolCallParams {
                    name: "ping".to_string(),
                    arguments: Some(json!({})),
                    meta: None,
                },
            )
            .await
            .expect("tools/call");
        let result = call.result.expect("call result");
        assert_eq!(result.content.len(), 1);
        assert_eq!(
            result.content[0]
                .data
                .get("text")
                .and_then(|v| v.as_str()),
            Some("pong")
        );

        // The configured Cookie header must be sent on every request.
        assert_eq!(
            observed_cookie.lock().unwrap().clone(),
            Some("knowmax.session=abc123".to_string())
        );
    }
}
