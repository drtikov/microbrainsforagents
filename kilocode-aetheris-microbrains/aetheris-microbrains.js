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
    "X-Aetheris-Host": "kilocode",
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

export async function AetherisMicrobrainsPlugin(ctx = {}) {
  const cfg = _resolveEnvConfig(ctx?.config || {});
  const cwd = ctx?.directory || ctx?.worktree || undefined;

  return {
    "tool.execute.before": async (input, output) => {
      const args = output && typeof output.args === "object" && output.args !== null ? output.args : {};
      const payload = {
        toolName: String(input?.tool || input?.toolName || ""),
        params: args,
        cwd,
        callID: input?.callID,
        context: {
          sessionId: input?.sessionID,
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
          throw new Error(`Aetheris pre-flight unavailable (fail-closed): ${String(err?.message || err)}`);
        }
        return;
      }

      if (cfg.shadow || !verdict || typeof verdict !== "object") {
        return;
      }
      if (verdict.block === true) {
        throw new Error(String(verdict.blockReason || verdict.reason || "Blocked by Aetheris pre-flight policy"));
      }
      if (verdict.escalate === true && cfg.blockOnEscalate) {
        throw new Error(`Aetheris escalation (requires review): ${String(verdict.reason || "Unvetted action")}`);
      }
      if (verdict.args && typeof verdict.args === "object" && output && typeof output.args === "object") {
        Object.assign(output.args, verdict.args);
      }
    },

    "tool.execute.after": async (input, output) => {
      if (output && typeof output === "object" && "output" in output) {
        const scrubbed = scrubSecretsSync(output.output);
        if (scrubbed.redacted) {
          output.output = scrubbed.value;
        }
      }
      const payload = {
        toolName: String(input?.tool || input?.toolName || ""),
        params: output?.args && typeof output.args === "object" ? output.args : {},
        output: output?.output ?? "",
        context: {
          sessionId: input?.sessionID,
        },
      };
      try {
        await _postDaemon(`${cfg.daemonUrl}/v1/hook/post`, cfg.token, payload, cfg.timeoutMs);
      } catch {
        // Best-effort IFC taint update; secrets already scrubbed synchronously
      }
    },
  };
}

export default { id: "aetheris-microbrains", server: AetherisMicrobrainsPlugin };
