export type PretoolDecision = "allow" | "deny" | "escalate";

export interface PretoolEvalResult {
  decision: PretoolDecision;
  reason_code: string;
  label: string;
  layer: number;
  host: string;
  profile: string;
}

export interface ScrubResult {
  secrets_found: boolean;
  redacted_text: string;
  injection_detected: boolean;
}

export interface AetherisWasmGate {
  /**
   * Evaluate a PreToolUse payload in-process via the Rust WASM Layer 0/1 engine.
   */
  evalPretool(
    payload: string | Record<string, unknown>,
    host?: string,
    profile?: string,
  ): PretoolEvalResult;

  /**
   * Scan and redact secrets and detect prompt injection in `text`.
   */
  scrub(text: string): ScrubResult;
}

/**
 * Resolve the default `aetheris_gate.wasm` path (or `AETHERIS_WASM_PATH` env override).
 */
export function resolveDefaultWasmPath(): string;

/**
 * Load and instantiate `aetheris_gate.wasm` using Node's `WebAssembly.instantiate`.
 */
export function loadAetherisWasm(
  wasmPath?: string | URL | Uint8Array | ArrayBuffer,
): Promise<AetherisWasmGate>;

declare const _default: {
  loadAetherisWasm: typeof loadAetherisWasm;
  resolveDefaultWasmPath: typeof resolveDefaultWasmPath;
};

export default _default;
