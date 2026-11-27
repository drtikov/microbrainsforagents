import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const _ENCODER = new TextEncoder();
const _DECODER = new TextDecoder("utf-8");

const _REQUIRED_EXPORTS = [
  "memory",
  "aetheris_wasm_alloc",
  "aetheris_wasm_dealloc",
  "aetheris_wasm_eval_pretool",
  "aetheris_wasm_scrub",
];

/**
 * Resolve the default path to the compiled `aetheris_gate.wasm` artifact.
 * @returns {string}
 */
export function resolveDefaultWasmPath() {
  if (typeof process !== "undefined" && process?.env?.AETHERIS_WASM_PATH) {
    return process.env.AETHERIS_WASM_PATH;
  }
  return fileURLToPath(
    new URL(
      "../rust/aetheris-gate/target/wasm32-unknown-unknown/release/aetheris_gate.wasm",
      import.meta.url,
    ),
  );
}

async function _loadWasmBytes(wasmSource) {
  if (wasmSource instanceof Uint8Array) {
    return wasmSource;
  }
  if (wasmSource instanceof ArrayBuffer) {
    return new Uint8Array(wasmSource);
  }
  if (wasmSource instanceof URL) {
    return await readFile(wasmSource);
  }
  const targetPath = typeof wasmSource === "string" && wasmSource.length > 0
    ? wasmSource
    : resolveDefaultWasmPath();
  return await readFile(targetPath);
}

/**
 * Instantiate the Aetheris Gate WASM module and return a leak-free JS wrapper.
 *
 * @param {string | URL | Uint8Array | ArrayBuffer} [wasmPath]
 * @returns {Promise<{ evalPretool: (payload: string | Record<string, unknown>, host?: string, profile?: string) => any, scrub: (text: string) => any }>}
 */
export async function loadAetherisWasm(wasmPath = undefined) {
  const wasmBytes = await _loadWasmBytes(wasmPath);
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exports = instance.exports;

  for (const sym of _REQUIRED_EXPORTS) {
    if (!(sym in exports)) {
      throw new Error(`Aetheris WASM module is missing required export: ${sym}`);
    }
  }

  const memory = exports.memory;
  const alloc = exports.aetheris_wasm_alloc;
  const dealloc = exports.aetheris_wasm_dealloc;
  const wasmEvalPretool = exports.aetheris_wasm_eval_pretool;
  const wasmScrub = exports.aetheris_wasm_scrub;

  /**
   * Allocate and copy a UTF-8 string into WASM linear memory.
   * Returns `{ ptr, len }` where `ptr === 0` if `len === 0`.
   */
  function writeUtf8(str) {
    const bytes = _ENCODER.encode(str);
    if (bytes.length === 0) {
      return { ptr: 0, len: 0 };
    }
    const ptr = alloc(bytes.length) >>> 0;
    if (ptr === 0) {
      throw new Error(`aetheris_wasm_alloc failed for ${bytes.length} bytes`);
    }
    new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  }

  function freeBuf(buf) {
    if (buf && buf.ptr !== 0) {
      dealloc(buf.ptr, buf.len);
    }
  }

  function allocU32Slot() {
    const ptr = alloc(4) >>> 0;
    if (ptr === 0) {
      throw new Error("aetheris_wasm_alloc failed for 4-byte out_len slot");
    }
    new DataView(memory.buffer).setUint32(ptr, 0, true);
    return { ptr, len: 4 };
  }

  function readAndFreeResult(outPtrRaw, outLenPtr) {
    const outPtr = outPtrRaw >>> 0;
    const outLen = new DataView(memory.buffer).getUint32(outLenPtr, true);
    try {
      if (outPtr === 0 || outLen === 0) {
        return "";
      }
      const slice = new Uint8Array(memory.buffer, outPtr, outLen);
      return _DECODER.decode(slice);
    } finally {
      if (outPtr !== 0) {
        dealloc(outPtr, outLen);
      }
    }
  }

  function evalPretool(payload, host = "claude", profile = "dev") {
    const inputJson =
      typeof payload === "string"
        ? payload
        : JSON.stringify(payload ?? {});
    const hostStr = host == null ? "claude" : String(host);
    const profileStr = profile == null ? "dev" : String(profile);

    let inputBuf = null;
    let hostBuf = null;
    let profileBuf = null;
    let outLenSlot = null;

    try {
      inputBuf = writeUtf8(inputJson);
      hostBuf = writeUtf8(hostStr);
      profileBuf = writeUtf8(profileStr);
      outLenSlot = allocU32Slot();

      const outPtr = wasmEvalPretool(
        inputBuf.ptr,
        inputBuf.len,
        hostBuf.ptr,
        hostBuf.len,
        profileBuf.ptr,
        profileBuf.len,
        outLenSlot.ptr,
      );
      const rawJson = readAndFreeResult(outPtr, outLenSlot.ptr);
      return JSON.parse(rawJson);
    } finally {
      freeBuf(outLenSlot);
      freeBuf(profileBuf);
      freeBuf(hostBuf);
      freeBuf(inputBuf);
    }
  }

  function scrub(text) {
    const textStr = text == null ? "" : String(text);
    let textBuf = null;
    let outLenSlot = null;

    try {
      textBuf = writeUtf8(textStr);
      outLenSlot = allocU32Slot();

      const outPtr = wasmScrub(textBuf.ptr, textBuf.len, outLenSlot.ptr);
      const rawJson = readAndFreeResult(outPtr, outLenSlot.ptr);
      return JSON.parse(rawJson);
    } finally {
      freeBuf(outLenSlot);
      freeBuf(textBuf);
    }
  }

  return {
    evalPretool,
    scrub,
  };
}

export default {
  loadAetherisWasm,
  resolveDefaultWasmPath,
};
