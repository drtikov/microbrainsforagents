"""Example 06: External Tool-Drop Rate, Retry Economics & Live-LLM A/B Benchmark.

Addresses the key metric for MCP-Zero Top-K tool narrowing:
  1. Real serialized OpenAI JSON tool schemas (32 tools) rather than a flat constant.
  2. Needed-Tool Drop Rate (1 - Recall@K) across K in {2, 4, 6, 8} on 40 diverse
     SWE-bench / BFCL-style tasks not copied from detector rules.
  3. Net Token Savings *after* Retry Penalty (re-prompting with the full 32-tool
     catalog + context whenever Top-K drops the required tool).
  4. Optional live-LLM A/B mode (`--live`) using the `openai` SDK against any
     OpenAI-compatible endpoint (OpenAI, NVIDIA NIM, vLLM, or local Ollama) to
     measure real API prompt_tokens, tool accuracy, retries, and wall-clock time.

Usage:
    python examples/06_real_llm_ab_benchmark.py
    python examples/06_real_llm_ab_benchmark.py --live --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aetheris_microbrains as mb


# ---------------------------------------------------------------------------
# 1. Real 32-Tool OpenAI Function-Calling Schema Registry
# ---------------------------------------------------------------------------
def build_32_tool_schemas() -> List[Dict[str, Any]]:
    """Return 32 realistic OpenAI-format JSON tool definitions."""
    specs: List[Tuple[str, str, Dict[str, Any]]] = [
        ("repo_read", "Read a source file from the repository with optional line range.", {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}),
        ("repo_write", "Write or overwrite a file in the repository.", {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("edit_file", "Apply a targeted diff or search-and-replace edit to an existing file.", {"path": {"type": "string"}, "diff": {"type": "string"}}),
        ("repo_list", "List files and directories inside a repository path.", {"path": {"type": "string"}, "recursive": {"type": "boolean"}}),
        ("grep_search", "Search repository files using a regular expression pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
        ("find_files", "Find files matching a glob pattern in the workspace.", {"glob": {"type": "string"}}),
        ("git_status", "Show working tree status, staged changes, and current branch.", {}),
        ("git_diff", "Show git diff for staged or unstaged changes.", {"target": {"type": "string"}}),
        ("git_commit", "Stage and commit repository changes with a commit message.", {"message": {"type": "string"}}),
        ("git_log", "Inspect recent git commit history.", {"limit": {"type": "integer"}}),
        ("run_shell", "Execute a shell command in the workspace.", {"cmd": {"type": "string"}}),
        ("run_tests", "Execute pytest or project test suite.", {"target": {"type": "string"}, "verbose": {"type": "boolean"}}),
        ("lint_check", "Run static linter / type checker on modified files.", {"path": {"type": "string"}}),
        ("format_code", "Run code formatter on specified files.", {"path": {"type": "string"}}),
        ("memory_recall", "Search long-term episodic or semantic memory store.", {"query": {"type": "string"}, "top_k": {"type": "integer"}}),
        ("memory_store", "Persist a structured note or fact into long-term memory.", {"key": {"type": "string"}, "value": {"type": "string"}}),
        ("web_search", "Search the web for documentation, CVEs, or API references.", {"query": {"type": "string"}}),
        ("web_fetch", "Fetch and extract readable text from a URL.", {"url": {"type": "string"}}),
        ("browser_open", "Open a URL in a headless browser and inspect rendered DOM.", {"url": {"type": "string"}}),
        ("db_query", "Execute a read-only SQL query against the analytics database.", {"sql": {"type": "string"}}),
        ("db_migrate", "Apply or inspect database schema migrations.", {"revision": {"type": "string"}}),
        ("http_request", "Send an HTTP REST request to an internal or external endpoint.", {"method": {"type": "string"}, "url": {"type": "string"}}),
        ("docker_ps", "List running containers and health status.", {}),
        ("docker_logs", "Fetch recent logs from a container.", {"container": {"type": "string"}, "tail": {"type": "integer"}}),
        ("k8s_pods", "List Kubernetes pods in a namespace.", {"namespace": {"type": "string"}}),
        ("metrics_query", "Query Prometheus/telemetry time-series metrics.", {"promql": {"type": "string"}}),
        ("trace_lookup", "Inspect distributed trace spans by trace_id.", {"trace_id": {"type": "string"}}),
        ("package_audit", "Scan dependencies for known vulnerabilities.", {"manifest": {"type": "string"}}),
        ("ast_outline", "Extract classes, functions, and symbols from a source file.", {"path": {"type": "string"}}),
        ("env_check", "Inspect sanitized runtime environment and Python version.", {}),
        ("archive_extract", "Inspect or extract a local tar/zip archive.", {"archive_path": {"type": "string"}}),
        ("finish_turn", "Return final verified answer to the user.", {"summary": {"type": "string"}}),
    ]
    assert len(specs) == 32
    schemas: List[Dict[str, Any]] = []
    for name, desc, props in specs:
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props.keys())[:1],
                },
            },
        })
    return schemas


def count_schema_tokens(schemas: Sequence[Dict[str, Any]]) -> int:
    """Count exact tokens for JSON schemas using tiktoken if present, else calibrated BPE estimate."""
    payload = json.dumps(list(schemas), separators=(",", ":"), ensure_ascii=False)
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(payload))
    except Exception:
        # Calibrated JSON-schema BPE ratio (~3.7 chars/token on compact OpenAI function JSON)
        subwords = re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z0-9_]", payload)
        return max(1, int(round(0.62 * len(subwords) + 0.12 * len(payload))))


# ---------------------------------------------------------------------------
# 2. 40 Diverse External-Style Multi-Tool Tasks (SWE-bench / BFCL Style)
# ---------------------------------------------------------------------------
def build_external_40_tasks() -> List[Dict[str, str]]:
    """40 natural tasks spanning direct, paraphrased, bilingual, and OOD queries."""
    return [
        {"query": "open src/auth/jwt_handler.py and inspect token expiry logic", "expected_tool": "repo_read"},
        {"query": "view the contents of pyproject.toml around dependencies", "expected_tool": "repo_read"},
        {"query": "покажи содержимое файла core/router.py", "expected_tool": "repo_read"},
        {"query": "create a new helper module src/utils/retry.py with backoff", "expected_tool": "repo_write"},
        {"query": "patch the off-by-one bug in pagination.py line 42", "expected_tool": "edit_file"},
        {"query": "update the existing timeout constant in client.py", "expected_tool": "edit_file"},
        {"query": "отредактируй обработчик ошибок в файле api.py", "expected_tool": "edit_file"},
        {"query": "list all files inside the tests/integration directory", "expected_tool": "repo_list"},
        {"query": "find where class ConnectionPool is defined across the codebase", "expected_tool": "grep_search"},
        {"query": "locate all references to deprecated_v1_endpoint", "expected_tool": "grep_search"},
        {"query": "find all yaml configuration files in the repo", "expected_tool": "find_files"},
        {"query": "check current git branch and modified uncommitted files", "expected_tool": "git_status"},
        {"query": "покажи текущую ветку git и статус", "expected_tool": "git_status"},
        {"query": "show the unstaged git diff before I commit", "expected_tool": "git_diff"},
        {"query": "commit the staged bugfix with a descriptive message", "expected_tool": "git_commit"},
        {"query": "check the last 10 commits in repository history", "expected_tool": "git_log"},
        {"query": "run pytest on tests/test_auth.py to verify the fix", "expected_tool": "run_tests"},
        {"query": "execute the unit test suite for billing", "expected_tool": "run_tests"},
        {"query": "запусти тесты для проверки регрессий", "expected_tool": "run_tests"},
        {"query": "run ruff or flake8 lint checks on modified python files", "expected_tool": "lint_check"},
        {"query": "format the codebase with black/clang-format", "expected_tool": "format_code"},
        {"query": "execute shell command df -h to check disk space", "expected_tool": "run_shell"},
        {"query": "recall what architecture decision we saved yesterday", "expected_tool": "memory_recall"},
        {"query": "найди в памяти прошлые заметки о конфигурации", "expected_tool": "memory_recall"},
        {"query": "save this postmortem summary to long-term memory", "expected_tool": "memory_store"},
        {"query": "search online documentation for Pydantic v2 model_validator", "expected_tool": "web_search"},
        {"query": "fetch the changelog text from https://example.com/releases", "expected_tool": "web_fetch"},
        {"query": "open localhost:3000 in browser to inspect rendered login form", "expected_tool": "browser_open"},
        {"query": "run a read-only SELECT count(*) query on orders table", "expected_tool": "db_query"},
        {"query": "check pending alembic database migrations", "expected_tool": "db_migrate"},
        {"query": "send a GET healthcheck request to http://localhost:8080/health", "expected_tool": "http_request"},
        {"query": "list running docker containers on this host", "expected_tool": "docker_ps"},
        {"query": "tail the last 200 lines of api-gateway container logs", "expected_tool": "docker_logs"},
        {"query": "list kubernetes pods in the staging namespace", "expected_tool": "k8s_pods"},
        {"query": "query prometheus p99 request latency for checkout service", "expected_tool": "metrics_query"},
        {"query": "scan requirements.txt for known CVE vulnerabilities", "expected_tool": "package_audit"},
        {"query": "inspect classes and function signatures in parser.py", "expected_tool": "ast_outline"},
        {"query": "check python version and sanitized runtime environment", "expected_tool": "env_check"},
        {"query": "закажи такси до дома", "expected_tool": "OOD_NO_TOOL"},
        {"query": "write a haiku about autumn leaves", "expected_tool": "OOD_NO_TOOL"},
    ]


# ---------------------------------------------------------------------------
# 3. Drop-Rate & Retry-Adjusted Net Token Economics Evaluation
# ---------------------------------------------------------------------------
def evaluate_drop_rate_and_net_savings(
    top_k_values: Sequence[int] = (2, 4, 6, 8),
    context_tokens_per_retry: int = 1200,
) -> Dict[str, Any]:
    """Measure Needed-Tool Drop Rate (1 - Recall@K) and Net Token Savings after retries."""
    schemas = build_32_tool_schemas()
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    all_tool_names = list(schema_by_name.keys())
    full_schema_tokens = count_schema_tokens(schemas)
    tasks = build_external_40_tasks()
    in_domain_tasks = [t for t in tasks if t["expected_tool"] != "OOD_NO_TOOL"]
    ood_tasks = [t for t in tasks if t["expected_tool"] == "OOD_NO_TOOL"]

    # Warm router
    mb.narrow_tool_surface("read file", available_tools=all_tool_names, top_k=4)

    rows: List[Dict[str, Any]] = []
    for k in top_k_values:
        dropped = 0
        escalated_full = 0
        ood_escalated = 0
        gross_tokens = 0
        retry_tokens = 0
        latencies_ms: List[float] = []

        for task in in_domain_tasks:
            t0 = time.perf_counter_ns()
            res = mb.narrow_tool_surface(task["query"], available_tools=all_tool_names, top_k=k)
            latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

            cands = res["candidate_tools"]
            cand_schemas = [schema_by_name[c] for c in cands if c in schema_by_name]
            step_tokens = count_schema_tokens(cand_schemas)
            gross_tokens += step_tokens

            if res["mode"] == "escalate_full_catalog":
                escalated_full += 1

            if task["expected_tool"] not in cands:
                dropped += 1
                # Retry penalty: agent falls back to full 32-tool catalog + re-sends conversation context
                retry_tokens += full_schema_tokens + context_tokens_per_retry

        for task in ood_tasks:
            res = mb.narrow_tool_surface(task["query"], available_tools=all_tool_names, top_k=k)
            if not res["conformal_passed"]:
                ood_escalated += 1

        n_in = len(in_domain_tasks)
        unfiltered_total = n_in * full_schema_tokens
        net_total = gross_tokens + retry_tokens
        gross_saved_pct = (1.0 - (gross_tokens / float(unfiltered_total))) * 100.0
        net_saved_pct = (1.0 - (net_total / float(unfiltered_total))) * 100.0
        drop_rate_pct = (dropped / float(n_in)) * 100.0
        latencies_ms.sort()
        p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]

        rows.append({
            "top_k": k,
            "tasks": n_in,
            "dropped": dropped,
            "drop_rate_pct": round(drop_rate_pct, 1),
            "recall_at_k_pct": round(100.0 - drop_rate_pct, 1),
            "escalated_full_count": escalated_full,
            "ood_escalated": f"{ood_escalated}/{len(ood_tasks)}",
            "full_schema_tokens_per_step": full_schema_tokens,
            "gross_saved_pct": round(gross_saved_pct, 1),
            "retry_penalty_tokens": retry_tokens,
            "net_saved_pct": round(net_saved_pct, 1),
            "p95_ms": round(p95_ms, 3),
        })

    return {
        "full_32_tool_schema_tokens": full_schema_tokens,
        "context_tokens_per_retry": context_tokens_per_retry,
        "in_domain_tasks": len(in_domain_tasks),
        "ood_tasks": len(ood_tasks),
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# 4. Optional Live-LLM A/B Benchmark (--live)
# ---------------------------------------------------------------------------
def run_live_llm_ab(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    limit: int = 12,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Compare real LLM tool-calling with Full 32 Tools vs. Microbrains Top-K + Retry."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
    schemas = build_32_tool_schemas()
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    all_names = list(schema_by_name.keys())
    tasks = [t for t in build_external_40_tasks() if t["expected_tool"] != "OOD_NO_TOOL"][:limit]

    def _eval_mode(narrowed: bool) -> Dict[str, Any]:
        prompt_toks = 0
        comp_toks = 0
        correct = 0
        retries = 0
        t_start = time.perf_counter()

        for task in tasks:
            active_schemas = schemas
            if narrowed:
                res = mb.narrow_tool_surface(task["query"], available_tools=all_names, top_k=top_k)
                active_schemas = [schema_by_name[c] for c in res["candidate_tools"] if c in schema_by_name]

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Select the single best tool for the user task."},
                    {"role": "user", "content": task["query"]},
                ],
                tools=active_schemas,
                tool_choice="auto",
                temperature=0.0,
            )
            if resp.usage:
                prompt_toks += int(resp.usage.prompt_tokens or 0)
                comp_toks += int(resp.usage.completion_tokens or 0)

            tool_calls = resp.choices[0].message.tool_calls or []
            chosen = tool_calls[0].function.name if tool_calls else ""

            if narrowed and chosen != task["expected_tool"] and task["expected_tool"] not in [s["function"]["name"] for s in active_schemas]:
                # Retry with full 32-tool schema when Top-K dropped the needed tool
                retries += 1
                resp2 = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Select the single best tool for the user task."},
                        {"role": "user", "content": task["query"]},
                    ],
                    tools=schemas,
                    tool_choice="auto",
                    temperature=0.0,
                )
                if resp2.usage:
                    prompt_toks += int(resp2.usage.prompt_tokens or 0)
                    comp_toks += int(resp2.usage.completion_tokens or 0)
                tc2 = resp2.choices[0].message.tool_calls or []
                chosen = tc2[0].function.name if tc2 else ""

            if chosen == task["expected_tool"]:
                correct += 1

        elapsed_s = time.perf_counter() - t_start
        return {
            "tasks": len(tasks),
            "accuracy_pct": round((correct / float(len(tasks))) * 100.0, 1),
            "retries": retries,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "wall_clock_s": round(elapsed_s, 2),
        }

    baseline = _eval_mode(narrowed=False)
    microbrains = _eval_mode(narrowed=True)
    net_prompt_saved_pct = round(
        (1.0 - (microbrains["prompt_tokens"] / float(max(1, baseline["prompt_tokens"])))) * 100.0,
        1,
    )
    return {
        "model": model,
        "top_k": top_k,
        "baseline_full_32": baseline,
        "microbrains_narrowed": microbrains,
        "net_api_prompt_token_saved_pct": net_prompt_saved_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External Tool-Drop Rate & Live-LLM A/B Benchmark")
    parser.add_argument("--live", action="store_true", help="Run real LLM A/B test via OpenAI-compatible API")
    parser.add_argument("--model", default=os.environ.get("MICROBRAINS_BENCH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_drop_rate_and_net_savings()
    print("=== Aetheris Microbrains External Tool-Drop Rate & Retry-Adjusted Economics ===")
    print(f"Real 32-tool OpenAI JSON schema size: {report['full_32_tool_schema_tokens']:,} tokens/step "
          f"(retry context assumption: {report['context_tokens_per_retry']:,} tokens)\n")
    print("| Top-K | Recall@K | Needed-Tool Drop Rate | Escalated to Full | OOD Escalated | Gross Saved | Net Saved (After Retries) | Local p95 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["sweep"]:
        print(
            f"| {row['top_k']} | {row['recall_at_k_pct']}% | {row['drop_rate_pct']}% ({row['dropped']}/{row['tasks']}) | "
            f"{row['escalated_full_count']}/{row['tasks']} | {row['ood_escalated']} | "
            f"{row['gross_saved_pct']}% | {row['net_saved_pct']}% | {row['p95_ms']:.3f} ms |"
        )

    if args.live:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not api_key and not args.base_url:
            print("\n[--live skipped] Set OPENAI_API_KEY (or --base-url for Ollama/vLLM) to run live LLM A/B.")
            return
        print("\n=== Live LLM A/B Comparison ===")
        ab = run_live_llm_ab(model=args.model, base_url=args.base_url, api_key=api_key, limit=args.limit)
        print(json.dumps(ab, indent=2))


if __name__ == "__main__":
    main()
