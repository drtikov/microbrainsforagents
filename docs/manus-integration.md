# Manus Custom MCP Integration

Connect **Aetheris Microbrains** to **Manus** (including **Manus 2.0**) through Manus's documented **Custom MCP Servers** connector over Streamable HTTP (`/mcp`) with TLS + Bearer authentication.

> **Important Security Boundary (Advisory Only):**
> Custom MCP tools exposed to Manus are **callable/advisory tools** (`microbrain_preflight_check`, `microbrain_scrub_and_scan`, `microbrain_scan_mcp_tools`, `microbrain_narrow_tools`, `microbrain_health`). They do **not** automatically intercept or block Manus's built-in file, shell, or browser operations unless the agent workflow explicitly calls these MCP tools.

---

## 1. Local Smoke Test

Install the MCP extra and verify the conservative remote profile locally:

```bash
pip install 'aetheris-microbrains[mcp]'
aetheris-mcp --check --profile manus
```

Run a loopback-only local smoke server (requires `--allow-unauthenticated-local` when starting `--profile manus` without a bearer token):

```bash
aetheris-mcp --transport streamable-http --profile manus --host 127.0.0.1 --port 8080 --allow-unauthenticated-local
curl http://127.0.0.1:8080/health
```

With Bearer authentication enabled locally:

```bash
export AETHERIS_MCP_BEARER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
aetheris-mcp --transport streamable-http --profile manus --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
# Expect 401 without Authorization header:
curl -i -X POST http://127.0.0.1:8080/mcp
```

---

## 2. Single-User Production Deployment (`deploy/manus/`)

For Manus Cloud, deploy one isolated instance per user behind an HTTPS ingress proxy (Caddy/Nginx/Cloudflare/ALB). **Never expose `0.0.0.0:8080` directly to the public internet without TLS and authentication, and never share one server/state directory across unrelated Manus users.**

The ready-to-use recipe in [`deploy/manus/`](../deploy/manus/) runs `aetheris-mcp --profile manus` on a private container port (`8080`) with in-process `BearerAuthMiddleware` enabled and fronts it with automatic HTTPS via Caddy:

```bash
cd deploy/manus
export MICROBRAINS_DOMAIN=microbrains.example.com
export AETHERIS_MCP_BEARER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up -d --build
```

Verify from outside:

```bash
curl https://$MICROBRAINS_DOMAIN/health
# Unauthenticated MCP call must return HTTP 401:
curl -i -X POST https://$MICROBRAINS_DOMAIN/mcp
```

---

## 3. Connect in Manus

1. Open Manus **Settings → Integrations → Custom MCP Servers**.
2. Add your HTTPS MCP endpoint:
   - **URL:** `https://microbrains.example.com/mcp`
   - **Authentication header:** `Authorization: Bearer <your AETHERIS_MCP_BEARER_TOKEN>`
3. Save and click **Test connection** so Manus completes the MCP `initialize` and `tools/list` handshake.

---

## 4. Tools Exposed in `--profile manus`

To keep remote deployments safe by default, `--profile manus` enables a strict 5-tool advisory allowlist:

| Tool | Purpose | Remote Safety Behavior |
| :--- | :--- | :--- |
| `microbrain_health` | Health & version probe | Omits host `state_dir`, `journal_path`, and daemon socket details |
| `microbrain_preflight_check` | Advisory pre-flight shell/diff/path check | Ignores caller `session_state_path` |
| `microbrain_scrub_and_scan` | Secret redaction + prompt-injection scan | Stateless string scan |
| `microbrain_scan_mcp_tools` | MCP tool manifest poisoning/shadowing audit | Stateless manifest check |
| `microbrain_narrow_tools` | Relevance-based Top-K tool schema narrowing | Stateless schema routing |

Local-state tools (`microbrain_audit_explain`, `microbrain_directives`, `microbrain_verify`) are **disabled by default** in `--profile manus`. If you explicitly opt in with `--enable-local-tools --allowed-project-root /workspace`, `microbrain_verify(project_dir=...)` fails closed on any path outside `allowed_project_root`.

---

## 5. Troubleshooting

- **`Refusing unauthenticated remote Streamable HTTP`: ** Set `AETHERIS_MCP_BEARER_TOKEN` in the environment (or pass `--auth-token-env <VAR>`). `--profile manus` fails closed without a token unless `--allow-unauthenticated-local` is used on `127.0.0.1`/`localhost`/`::1`.
- **`401 Unauthorized` from `/mcp`:** Ensure Manus sends `Authorization: Bearer <token>` with no extra whitespace or quotes. Pass `--protect-health` if you also want `/health` to require the same Bearer token.
- **`Missing MCP SDK`:** Install with `pip install 'aetheris-microbrains[mcp]'` (targets `mcp<2.0.0` or compatible FastMCP Streamable HTTP).
