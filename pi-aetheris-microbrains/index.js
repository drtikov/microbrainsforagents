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

function _getSessionId(ctx) {
  return ctx?.sessionManager?.getSessionId?.() ?? ctx?.sessionId;
}

function _resolveEffectiveConfig(baseOverrides = {}, ctx = undefined) {
  const env = typeof process !== "undefined" && process?.env ? process.env : {};
  const rawSettings = ctx?.settingsManager?.get?.("aetheris");
  const settings = rawSettings && typeof rawSettings === "object" ? rawSettings : {};
  const daemonUrl = String(
    baseOverrides.daemonUrl ?? settings.daemonUrl ?? env.AETHERIS_DAEMON_URL ?? "http://127.0.0.1:8765",
  ).replace(/\/+$/, "");
  const timeoutRaw = Number(baseOverrides.timeoutMs ?? settings.timeoutMs ?? env.AETHERIS_TIMEOUT_MS ?? 2500);
  const timeoutMs = Math.min(Math.max(Number.isFinite(timeoutRaw) ? timeoutRaw : 2500, 200), 4500);
  return {
    daemonUrl,
    token: String(baseOverrides.token ?? settings.token ?? env.AETHERIS_DAEMON_TOKEN ?? ""),
    shadow: Boolean(baseOverrides.shadow ?? settings.shadow ?? (env.AETHERIS_HOOK_SHADOW === "1")),
    failClosed: Boolean(baseOverrides.failClosed ?? settings.failClosed ?? (env.AETHERIS_FAIL_CLOSED !== "0")),
    blockOnEscalate: Boolean(baseOverrides.blockOnEscalate ?? settings.blockOnEscalate ?? (env.AETHERIS_BLOCK_ON_ESCALATE !== "0")),
    timeoutMs,
  };
}

async function _postDaemon(url, token, body, timeoutMs, activeControllers = undefined) {
  const ctrl = new AbortController();
  activeControllers?.add(ctrl);
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
    activeControllers?.delete(ctrl);
  }
}

async function _checkDaemonHealth(daemonUrl, token, timeoutMs, activeControllers = undefined) {
  const ctrl = new AbortController();
  activeControllers?.add(ctrl);
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
    activeControllers?.delete(ctrl);
  }
}

export function createAetherisPiExtension(overrides = {}) {
  return function aetherisMicrobrainsExtension(pi) {
    const stats = { checked: 0, blocked: 0, approved: 0, redacted: 0 };
    const activeControllers = new Set();

    pi.on("session_start", async (_event, ctx) => {
      const entries = ctx?.sessionManager?.getEntries?.();
      if (Array.isArray(entries)) {
        for (let i = entries.length - 1; i >= 0; i -= 1) {
          const entry = entries[i];
          if (entry?.customType === "aetheris-stats" && entry?.data && typeof entry.data === "object") {
            for (const key of ["checked", "blocked", "approved", "redacted"]) {
              const val = Number(entry.data[key]);
              if (Number.isFinite(val) && val >= 0) {
                stats[key] = val;
              }
            }
            break;
          }
        }
      }

      const cfg = _resolveEffectiveConfig(overrides, ctx);
      const healthy = await _checkDaemonHealth(cfg.daemonUrl, cfg.token, cfg.timeoutMs, activeControllers);
      if (!healthy && !cfg.shadow && cfg.failClosed) {
        ctx?.ui?.notify?.(
          `Aetheris Microbrains daemon unreachable at ${cfg.daemonUrl} (fail-closed active)`,
          "warning",
        );
      }
    });

    pi.on("session_shutdown", async () => {
      for (const ctrl of activeControllers) {
        try {
          ctrl.abort();
        } catch {
          // ignore abort errors during shutdown
        }
      }
      activeControllers.clear();
      pi.appendEntry?.("aetheris-stats", { ...stats });
    });

    pi.on("tool_call", async (event, ctx) => {
      const cfg = _resolveEffectiveConfig(overrides, ctx);
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
          sessionId: _getSessionId(ctx),
        },
      };

      let verdict = null;
      let err = null;
      try {
        verdict = await _postDaemon(`${cfg.daemonUrl}/v1/hook/pre`, cfg.token, payload, cfg.timeoutMs, activeControllers);
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
      const cfg = _resolveEffectiveConfig(overrides, ctx);
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
          sessionId: _getSessionId(ctx),
        },
      };
      try {
        await _postDaemon(
          `${cfg.daemonUrl}/v1/hook/post`,
          cfg.token,
          payload,
          Math.min(cfg.timeoutMs, 1000),
          activeControllers,
        );
      } catch {
        // Post-tool daemon logging is best-effort; sync scrubbing above already executed.
      }
    });

    pi.registerCommand("aetheris", {
      description: "Show Aetheris Microbrains pre-flight status",
      handler: async (_args, ctx) => {
        const cfg = _resolveEffectiveConfig(overrides, ctx);
        const up = await _checkDaemonHealth(cfg.daemonUrl, cfg.token, cfg.timeoutMs, activeControllers);
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
