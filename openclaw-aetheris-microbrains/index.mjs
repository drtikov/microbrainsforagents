import { randomBytes } from "node:crypto";

const _SECRET_PATTERNS = [
  /\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b/g,
  /\bsk-(?:ant|proj)-[0-9A-Za-z_-]{16,}\b/g,
  /\bsk-[0-9A-Za-z]{20,}\b/g,
  /\bgh[pousr]_[0-9A-Za-z]{20,}\b/g,
  /\bgithub_pat_[0-9A-Za-z_]{20,}\b/g,
  /\bAKIA[0-9A-Z]{16}\b/g,
  /\bxox[baprs]-[0-9A-Za-z-]{12,}\b/g,
  /\bAIza[0-9A-Za-z_-]{24,}\b/g,
  /\b(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis):\/\/[^:\s]+:[^@\s]+@[^\s]+/gi,
  new RegExp(
    "-----BEGIN [A-Z0-9 ]*PRI" + "VATE KEY-----[\\s\\S]*?-----END [A-Z0-9 ]*PRI" + "VATE KEY-----",
    "g",
  ),
];

export function scrubSecretsSync(value) {
  if (typeof value === "string") {
    let text = value;
    let count = 0;
    for (const pat of _SECRET_PATTERNS) {
      pat.lastIndex = 0;
      text = text.replace(pat, () => {
        count += 1;
        return "[REDACTED_BY_AETHERIS]";
      });
    }
    return { value: text, redacted: count > 0, count };
  }
  if (Array.isArray(value)) {
    let redacted = false;
    let count = 0;
    const out = value.map((item) => {
      const sub = scrubSecretsSync(item);
      if (sub.redacted) {
        redacted = true;
        count += sub.count;
      }
      return sub.value;
    });
    return { value: out, redacted, count };
  }
  if (value && typeof value === "object") {
    let redacted = false;
    let count = 0;
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      const sub = scrubSecretsSync(v);
      if (sub.redacted) {
        redacted = true;
        count += sub.count;
      }
      out[k] = sub.value;
    }
    return { value: out, redacted, count };
  }
  return { value, redacted: false, count: 0 };
}

function _resolveConfig(api, ctx) {
  const raw = (ctx && ctx.pluginConfig) || api?.pluginConfig || api?.config || {};
  const port = Number.isInteger(raw.daemonPort) ? raw.daemonPort : 8765;
  const daemonUrl = String(raw.daemonUrl || `http://127.0.0.1:${port}`).replace(/\/+$/, "");
  const timeoutMs = Math.min(Math.max(Number(raw.timeoutMs ?? 2500), 200), 3500);
  const matcher = Array.isArray(raw.matcher) ? raw.matcher.map((m) => String(m)) : undefined;
  return {
    profile: raw.profile === "strict" ? "strict" : "dev",
    shadow: Boolean(raw.shadow),
    daemonUrl,
    port,
    autoStartDaemon: raw.autoStartDaemon !== false,
    failClosed: raw.failClosed !== false,
    timeoutMs,
    sandboxWrap: String(raw.sandboxWrap || "none"),
    autoApproveEscalations: Boolean(raw.autoApproveEscalations),
    pythonBin: String(raw.pythonBin || "python"),
    daemonToken: raw.daemonToken ? String(raw.daemonToken) : null,
    matcher,
  };
}

function _buildPrePayload(event, ctx) {
  const payload = {
    toolName: event?.toolName || event?.tool || "",
    params: event?.params && typeof event.params === "object" ? event.params : {},
  };
  if (Array.isArray(event?.derivedPaths)) {
    payload.derivedPaths = event.derivedPaths;
  }
  if (event?.runId) payload.runId = event.runId;
  if (event?.toolCallId) payload.toolCallId = event.toolCallId;
  if (ctx && typeof ctx === "object") {
    payload.context = {
      agentId: ctx.agentId,
      sessionKey: ctx.sessionKey,
      sessionId: ctx.sessionId,
      runId: ctx.runId,
      trigger: ctx.trigger,
      requester: ctx.requester,
    };
  }
  return payload;
}

async function _postDaemon(url, token, body, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Aetheris-Host": "openclaw",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return await resp.json();
  } finally {
    clearTimeout(timer);
  }
}

async function _checkHealth(daemonUrl, timeoutMs = 800) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(`${daemonUrl}/health`, { method: "GET", signal: ctrl.signal });
    return resp.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

const plugin = {
  id: "aetheris-microbrains",
  name: "Aetheris Microbrains Security Gate",
  description:
    "Deterministic sub-2ms pre-flight policy gate, polyglot AST/deferred-exec guard, IFC taint tracker, and synchronous DLP scrubber for OpenClaw.",
  register(api) {
    const baseCfg = _resolveConfig(api, null);
    const sessionToken = baseCfg.daemonToken || randomBytes(24).toString("hex");
    let daemonOwned = false;

    const fullMode = !api?.registrationMode || api.registrationMode === "full";
    if (fullMode && typeof api?.registerService === "function") {
      api.registerService({
        id: "aetheris-hook-daemon",
        async start(ctx) {
          const cfg = _resolveConfig(api, ctx);
          if (!cfg.autoStartDaemon) return;
          if (await _checkHealth(cfg.daemonUrl, 600)) return;
          const runCmd = api?.runtime?.system?.runCommandWithTimeout;
          if (typeof runCmd !== "function") {
            api?.logger?.warn?.("Aetheris hook daemon unreachable and OpenClaw runtime.system.runCommandWithTimeout unavailable.");
            return;
          }
          const args = [
            "-m",
            "aetheris_microbrains.hook_daemon",
            "--http",
            "--host",
            "openclaw",
            "--port",
            String(cfg.port),
            "--token",
            sessionToken,
            "--profile",
            cfg.profile,
            "--session-state",
            "auto",
            "--log-decisions",
            "--detach",
          ];
          if (cfg.shadow) args.push("--shadow");
          if (cfg.autoApproveEscalations) args.push("--auto-approve-escalations");
          if (cfg.sandboxWrap && cfg.sandboxWrap !== "none") {
            args.push("--sandbox-wrap", cfg.sandboxWrap);
          }
          try {
            const res = await runCmd(cfg.pythonBin, args, { timeoutMs: 4000, killProcessTree: true });
            if (res && res.code === 0) {
              daemonOwned = true;
              for (let i = 0; i < 10; i += 1) {
                if (await _checkHealth(cfg.daemonUrl, 300)) break;
                await new Promise((r) => setTimeout(r, 150));
              }
            }
          } catch (err) {
            api?.logger?.warn?.(`Failed to start Aetheris hook daemon: ${String(err)}`);
          }
        },
        async stop() {
          if (!daemonOwned) return;
          daemonOwned = false;
          try {
            await _postDaemon(`${baseCfg.daemonUrl}/v1/shutdown`, sessionToken, {}, 1500);
          } catch {
            // Daemon already stopped
          }
        },
      });
    }

    const hookOpts = { priority: 100, timeoutMs: 5000 };
    if (baseCfg.matcher && baseCfg.matcher.length > 0) {
      hookOpts.matcher = baseCfg.matcher;
    }

    api.on(
      "before_tool_call",
      async (event, ctx) => {
        const cfg = _resolveConfig(api, ctx);
        const payload = _buildPrePayload(event, ctx);
        let verdict = null;
        let lastErr = null;
        try {
          verdict = await _postDaemon(`${cfg.daemonUrl}/v1/hook/pre`, sessionToken, payload, cfg.timeoutMs);
        } catch (err) {
          lastErr = err;
          const runCmd = api?.runtime?.system?.runCommandWithTimeout;
          if (typeof runCmd === "function") {
            try {
              const cliArgs = [
                "-m",
                "aetheris_microbrains.hook",
                "--host",
                "openclaw",
                "--profile",
                cfg.profile,
                "--session-state",
                "auto",
                "--log-decisions",
              ];
              if (cfg.shadow) cliArgs.push("--shadow");
              if (cfg.autoApproveEscalations) cliArgs.push("--auto-approve-escalations");
              if (cfg.sandboxWrap && cfg.sandboxWrap !== "none") {
                cliArgs.push("--sandbox-wrap", cfg.sandboxWrap);
              }
              const res = await runCmd(cfg.pythonBin, cliArgs, {
                timeoutMs: cfg.timeoutMs,
                input: JSON.stringify(payload),
                killProcessTree: true,
              });
              if (res && typeof res.stdout === "string" && res.stdout.trim()) {
                verdict = JSON.parse(res.stdout.trim());
                lastErr = null;
              }
            } catch (cliErr) {
              lastErr = cliErr;
            }
          }
        }

        if (lastErr && !verdict) {
          if (cfg.shadow || !cfg.failClosed) {
            return undefined;
          }
          return {
            block: true,
            blockReason: `Aetheris pre-flight error/timeout (fail-closed): ${String(lastErr?.message || lastErr)}`,
          };
        }

        if (cfg.shadow || !verdict || typeof verdict !== "object") {
          return undefined;
        }
        if (verdict.block === true) {
          return {
            block: true,
            blockReason: String(verdict.blockReason || verdict.reason || "Blocked by Aetheris pre-flight policy"),
          };
        }
        if (verdict.requireApproval) {
          return { requireApproval: verdict.requireApproval };
        }
        if (verdict.params && typeof verdict.params === "object") {
          return { params: verdict.params };
        }
        return undefined;
      },
      hookOpts,
    );

    api.on(
      "tool_result_persist",
      (event) => {
        if (!event || typeof event !== "object") return undefined;
        const targetField =
          "result" in event ? "result" : "content" in event ? "content" : "message" in event ? "message" : null;
        if (!targetField) return undefined;
        const scrubbed = scrubSecretsSync(event[targetField]);
        if (!scrubbed.redacted) return undefined;
        return { [targetField]: scrubbed.value };
      },
      { priority: 100 },
    );

    api.on("after_tool_call", async (event, ctx) => {
      const cfg = _resolveConfig(api, ctx);
      const payload = {
        toolName: event?.toolName || event?.tool || "",
        params: event?.params && typeof event.params === "object" ? event.params : {},
        output: event?.output ?? event?.result ?? "",
        runId: event?.runId,
        toolCallId: event?.toolCallId,
        context: ctx && typeof ctx === "object" ? { sessionKey: ctx.sessionKey, sessionId: ctx.sessionId, trigger: ctx.trigger } : {},
      };
      try {
        await _postDaemon(`${cfg.daemonUrl}/v1/hook/post`, sessionToken, payload, cfg.timeoutMs);
      } catch {
        // Best-effort post observation; transcript secrets already scrubbed synchronously
      }
    });
  },
};

export default plugin;
