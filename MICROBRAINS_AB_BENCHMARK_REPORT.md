# Microbrains on/off A/B benchmark report

## Scope

- Installed DSH bundle: `@local/aetheris-microbrains-mcp`
- MCP endpoint: `http://127.0.0.1:8766/mcp`
- Deterministic 26-step coding session (14 benign + 12 hazard steps)
- Three live runs through the DSH-connected MCP endpoint
- Zero LLM calls in these measurements. DSH-internal client bridge overhead is excluded.
- No destructive commands were executed.

## 1. Safety and recovery comparison

| Metric | Without Microbrains | With Microbrains | Delta |
|---|---:|---:|---:|
| Hazard intercepts | 0/12 | 12/12 | **100% interception** |
| Runtime syntax failures | 4 | 0 | **100% prevented** |
| Wasted recovery turns | 8 | 0 | **8 avoided (100%)** |
| Security incidents executed | 8 | 0 | **100% blocked** |
| Benign calls blocked | — | 0/14 | **0% false positives** |
| Error-feedback tokens | 146 | 37 in-process / 12 MCP labels | **74.7% / 91.8% fewer** |

## 2. Live MCP latency over three runs

| Run | In-process mean | MCP mean | MCP p95 | Total for 26 checks | Transport share of MCP total |
|---|---:|---:|---:|---:|---:|
| 1 | 0.73 ms | 51.91 ms | 100.86 ms | 1,349.75 ms | 98.6% |
| 2 | 0.79 ms | 18.47 ms | 33.36 ms | 480.35 ms | 95.7% |
| 3 | 0.57 ms | 16.64 ms | 29.79 ms | 432.72 ms | 96.6% |
| **Median** | **0.73 ms** | **18.47 ms** | **33.36 ms** | **480.35 ms** | **96.6%** |

- MCP mean latency range: **16.64–51.91 ms/check**.
- MCP p95 latency range: **29.79–100.86 ms/check**.
- Median in-process mean: **0.73 ms/check**.
- Median MCP/in-process mean-latency ratio: **23.38x**.
- Transport is the dominant cost, not the policy engine.
- DSH-internal client bridge overhead is not included.

## 3. Tool-schema token economics (38 external tasks + 2 OOD)

| Top-K | Recall@K | Needed-tool drop rate | Net token saved after retries |
|---:|---:|---:|---:|
| 2 | 92.1% | 7.9% (3/38) | **35.1%** |
| 4 | 97.4% | 2.6% (1/38) | **39.7%** |
| 6 | 97.4% | 2.6% (1/38) | **36.6%** |
| 8 | 97.4% | 2.6% (1/38) | **33.6%** |

Full 32-tool schema size: **2,589 tokens/step**.

### Architecture comparison at Top-4

| Architecture | Recall@4 | Needed-tool drop rate | Net saved after retries |
|---|---:|---:|---:|
| Naive keyword | 76.3% | 23.7% | **52.1%** |
| Naive keyword + frozen escalation | 84.2% | 15.8% | **45.5%** |
| Embedding + ridge | 73.7% | 26.3% | **49.0%** |
| Microbrains hybrid | **97.4%** | **2.6%** | **39.7%** |

Microbrains has the best recall and lowest drop rate, but not the highest raw token reduction because it escalates to the full catalog when confidence is insufficient. That is an intentional fail-safe to avoid starving the agent of necessary tools.

## 4. Modeled token/cost savings (not a live LLM invoice)

- Input tokens saved per 26-step session: **~37,055**
- Estimated input-cost saved per session: **$0.1112**
- Estimated input-cost saved per 100 sessions: **$11.12**
- Model: 4,000 input tokens/recovery turn + 1,200 tokens/poisoned test output + traceback re-reads, Sonnet-class list pricing of $3/M input tokens.

These token and dollar figures are modeled estimates, not measured provider charges.

## Bottom line

Microbrains provides **100% hazard interception and zero false positives** in this scenario, while avoiding **8 recovery turns** and reducing schema tokens by **39.7% at Top-4**. The policy engine itself is sub-millisecond on this host; MCP transport adds the majority of latency, at a **median of 18.47 ms/check** across the three live runs.
