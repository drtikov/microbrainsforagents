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

export type TeleopCbfStatus =
  | "safe"
  | "simplex_braking_engaged"
  | "unsafe_no_solution"
  | "infeasible_box"
  | "invalid_input"
  | "dim_mismatch"
  | "malformed_json";

export interface TeleopCbfRequest {
  u_nominal: ArrayLike<number>;
  A?: ArrayLike<ArrayLike<number>>;
  b?: ArrayLike<number>;
  u_min?: ArrayLike<number> | null;
  u_max?: ArrayLike<number> | null;
  q_dot?: ArrayLike<number> | null;
  kd?: number;
  max_iters?: number;
  tol?: number;
}

export interface TeleopCbfResult {
  status: TeleopCbfStatus;
  verified_safe: boolean;
  u_filtered: number[];
  min_barrier_slack: number;
  iterations: number;
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

  /**
   * Clamp a nominal teleoperation command against affine CBF halfspaces `A u >= b`
   * and actuator bounds `[u_min, u_max]`, optionally engaging Simplex damping (`-kd * q_dot`).
   */
  filterTeleopCbf(request: string | TeleopCbfRequest): TeleopCbfResult;

  /**
   * Compute the deterministic dead-man watchdog authority scale in `[0.0, 1.0]`.
   */
  watchdogScale(ageMs: number, ttlMs: number, rampMs: number): number;
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
