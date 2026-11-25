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

function _resolveEnvConfig(overrides = {}) {
  const env = typeof process !== "undefined" && process?.env ? process.env : {};
  const daemonUrl = String(overrides.daemonUrl || env.AETHERIS_DAEMON_URL || "http://127.0.0.1:8765").replace(/\/+$/, "");
  const timeoutRaw = Number(overrides.timeoutMs ?? env.AETHERIS_TIMEOUT_MS ?? 2500);
  const timeoutMs = Math.min(Math.max(Number.isFinite(timeoutRaw) ? timeoutRaw : 2500, 200), 4500);
  return {
    daemonUrl,
    token: String(overrides.token ?? env.AETHERIS_DAEMON_TOKEN ?? ""),
    shadow: Boolean(overrides.shadow ?? env.AETHERIS_HOOK_SHADOW === "1"),
    failClosed: overrides.failClosed ?? env.AETHERIS_FAIL_CLOSED !== "0",
    blockOnEscalate: overrides.blockOnEscalate ?? env.AETHERIS_BLOCK_ON_ESCALATE !== "0",
    timeoutMs,
  };
}

async function _postDaemon(url, token, body, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const headers = {
    "Content-Type": "application/json",
    "X-Aetheris-Host": "pi",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers,
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

async function _checkDaemonHealth(daemonUrl, token, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), Math.min(timeoutMs, 1000));
  const headers = { "X-Aetheris-Host": "pi" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const resp = await fetch(`${daemonUrl}/health`, {
      method: "GET",
      headers,
      signal: ctrl.signal,
    });
    return resp.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export function createAetherisPiExtension(overrides = {}) {
  return function aetherisMicrobrainsExtension(pi) {
    const cfg = _resolveEnvConfig(overrides);
    const stats = { checked: 0, blocked: 0, approved: 0, redacted: 0 };

    pi.on("tool_call", async (event, ctx) => {
      stats.checked += 1;
      const toolName = String(event?.toolName || "");
      const input = event?.input && typeof event.input === "object" && event.input !== null ? event.input : {};
      const cwd = ctx?.cwd || (typeof process !== "undefined" && typeof process.cwd === "function" ? process.cwd() : undefined);
      const payload = {
        toolName,
        params: input,
        cwd,
        callID: event?.toolCallId,
        context: {
          sessionId: ctx?.sessionId,
        },
      };

      let verdict = null;
      let err = null;
      try {
        verdict = await _postDaemon(`${cfg.daemonUrl}/v1/hook/pre`, cfg.token, payload, cfg.timeoutMs);
      } catch (postErr) {
        err = postErr;
      }

      if (err && !verdict) {
        if (!cfg.shadow && cfg.failClosed) {
          stats.blocked += 1;
          return {
            block: true,
            reason: `Aetheris pre-flight unavailable (fail-closed): ${String(err?.message || err)}`,
          };
        }
        return undefined;
      }

      if (cfg.shadow || !verdict || typeof verdict !== "object") {
        return undefined;
      }

      if (verdict.block === true) {
        stats.blocked += 1;
        return {
          block: true,
          reason: String(verdict.blockReason || verdict.reason || "Blocked by Aetheris pre-flight policy"),
        };
      }

      if (verdict.escalate === true) {
        const reason = String(verdict.reason || "Unvetted action");
        if (typeof ctx?.ui?.confirm === "function") {
          const ok = await ctx.ui.confirm("Aetheris Microbrains Escalation", `${toolName}: ${reason}\nAllow execution?`);
          if (ok === true) {
            stats.approved += 1;
          } else {
            stats.blocked += 1;
            return {
              block: true,
              reason: `Aetheris escalation declined: ${reason}`,
            };
          }
        } else if (cfg.blockOnEscalate) {
          stats.blocked += 1;
          return {
            block: true,
            reason: `Aetheris escalation (requires review): ${reason}`,
          };
        }
      }

      if (verdict.args && typeof verdict.args === "object" && event && typeof event.input === "object" && event.input !== null) {
        Object.assign(event.input, verdict.args);
      }
      return undefined;
    });

    pi.on("tool_result", async (event, ctx) => {
      if (event && typeof event === "object") {
        if ("content" in event) {
          const scrubbedContent = scrubSecretsSync(event.content);
          if (scrubbedContent.redacted) {
            event.content = scrubbedContent.value;
            stats.redacted += scrubbedContent.count;
          }
        }
        if ("details" in event) {
          const scrubbedDetails = scrubSecretsSync(event.details);
          if (scrubbedDetails.redacted) {
            event.details = scrubbedDetails.value;
            stats.redacted += scrubbedDetails.count;
          }
        }
      }

      const payload = {
        toolName: String(event?.toolName || ""),
        params: {},
        output: event?.content ?? "",
        context: {
          sessionId: ctx?.sessionId,
        },
      };
      try {
        await _postDaemon(`${cfg.daemonUrl}/v1/hook/post`, cfg.token, payload, Math.min(cfg.timeoutMs, 1000));
      } catch {
        // Post-tool daemon logging is best-effort; sync scrubbing above already executed.
      }
    });

    pi.registerCommand("aetheris", {
      description: "Show Aetheris Microbrains pre-flight status",
      handler: async (_args, ctx) => {
        const up = await _checkDaemonHealth(cfg.daemonUrl, cfg.token, cfg.timeoutMs);
        const msg =
          `Aetheris Microbrains: daemon=${up ? "UP" : "DOWN"} shadow=${cfg.shadow} ` +
          `failClosed=${cfg.failClosed} checked=${stats.checked} blocked=${stats.blocked} ` +
          `approved=${stats.approved} redacted=${stats.redacted}`;
        ctx?.ui?.notify?.(msg, up ? "info" : "warning");
      },
    });
  };
}

export default function aetherisMicrobrainsExtension(pi) {
  return createAetherisPiExtension()(pi);
}
