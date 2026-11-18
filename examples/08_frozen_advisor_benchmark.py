"""Example 08: Frozen Multi-Label Tool Advisor & Session Prompt-Cache Benchmark.

Implements the controlled comparison suggested by independent engineering review:
- 120 frozen multi-label coding tasks across 6 external repository archetypes
  (web backend, CLI utility, ML trainer, SQL migration, Docker/CI ops, release tooling),
  frozen BEFORE running any comparator.
- Multiple acceptable tools per task (`acceptable_tools`) rather than a single arbitrary target.
- Four comparators evaluated over 3-turn sessions (modeling realistic LLM prompt caching where
  unchanged prefix schema tokens cost 0.1x of uncached input tokens vs 1.0x when invalidated):
    1. Full 32-Tool Catalog (Cached Prefix baseline)
    2. Simple Name+Description Keyword Ranker (Pre-Frozen Escalation Rule)
    3. Microbrains Top-4 Per-Turn Advisor (`narrow_tool_surface`)
    4. Microbrains Session-Sticky Top-4 Advisor (`SessionToolSurface`)
- Reports multi-label Recall@4, Drop Rate + 95% Clopper-Pearson upper bound, Escalation Rate,
  Agreed-Narrowed Error Rate (drops even when both Keyword & Microbrains narrow), Uncached vs
  Cache-Adjusted Effective Tokens, and Local p95 Latency.
- Also runs a decoupled `diagnose_patch()` check over clean and mutated Python diffs.

Usage:
    python examples/08_frozen_advisor_benchmark.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aetheris_microbrains as mb


def _load_ex06():
    p = _REPO_ROOT / "examples" / "06_real_llm_ab_benchmark.py"
    spec = importlib.util.spec_from_file_location("ex06_helpers", str(p))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_frozen_120_multilabel_tasks() -> List[Dict[str, Any]]:
    """Build 120 deterministic frozen multi-label coding tasks across 6 repo archetypes."""
    templates: List[Tuple[str, Tuple[str, ...]]] = [
        ("inspect file {mod} and examine implementation lines", ("repo_read", "grep_search")),
        ("search occurrences of symbol {sym} across codebase", ("grep_search", "repo_read")),
        ("apply bugfix diff to {mod} preserving public signature", ("edit_file", "repo_write")),
        ("write updated helper module inside {mod}", ("repo_write", "edit_file")),
        ("run pytest regression test suite for {pkg}", ("run_tests", "run_shell")),
        ("check git branch and working tree status before PR", ("git_status", "git_diff")),
        ("view uncommitted diff changes in {mod}", ("git_diff", "git_status")),
        ("inspect recent git commit history for {pkg}", ("git_log", "git_diff")),
        ("stage modified files and commit release patch", ("git_commit", "git_diff")),
        ("check ruff linting issues in {mod}", ("lint_check", "ast_outline")),
        ("verify static type annotations via mypy on {pkg}", ("lint_check", "ast_outline")),
        ("format code style with black in {mod}", ("format_code", "lint_check")),
        ("execute read-only SQL count query on {tbl} table", ("db_query", "db_migrate")),
        ("inspect database migrations and schema of {tbl}", ("db_migrate", "db_query")),
        ("list running docker containers and healthchecks", ("docker_ps", "docker_logs")),
        ("tail service error logs from docker container {pkg}", ("docker_logs", "docker_ps")),
        ("verify remote HTTP API health endpoint status", ("http_request", "web_fetch")),
        ("inspect environment packages inside virtualenv", ("env_check", "package_audit")),
        ("list files and subdirectories inside {pkg}", ("repo_list", "grep_search")),
        ("validate JSON config syntax in {cfg}", ("repo_read", "lint_check")),
    ]

    archetypes = [
        {"repo": "fastapi_service", "pkg": "services/auth", "mod": "services/auth/jwt_guard.py", "sym": "verify_token", "tbl": "sessions", "cfg": "config/auth.json"},
        {"repo": "cli_packager", "pkg": "cli/packer", "mod": "cli/packer/manifest.py", "sym": "build_wheel", "tbl": "artifacts", "cfg": "config/build.json"},
        {"repo": "ml_trainer", "pkg": "train/loop", "mod": "train/loop/scheduler.py", "sym": "step_cosine", "tbl": "checkpoints", "cfg": "config/train.json"},
        {"repo": "sql_migrator", "pkg": "db/migrate", "mod": "db/migrate/runner.py", "sym": "apply_rev", "tbl": "migrations", "cfg": "config/db.json"},
        {"repo": "ci_ops_runner", "pkg": "ops/ci", "mod": "ops/ci/container.py", "sym": "wait_healthy", "tbl": "jobs", "cfg": "config/pipeline.json"},
        {"repo": "release_bot", "pkg": "rel/tagger", "mod": "rel/tagger/changelog.py", "sym": "format_notes", "tbl": "releases", "cfg": "config/release.json"},
    ]

    tasks: List[Dict[str, Any]] = []
    for arch in archetypes:
        for idx, (tpl, acceptable) in enumerate(templates):
            tasks.append({
                "id": f"{arch['repo']}_{idx + 1:02d}",
                "repo": arch["repo"],
                "query": tpl.format(**arch),
                "acceptable_tools": list(acceptable),
                "followup_queries": [
                    f"continue verifying changes for {arch['sym']} in {arch['mod']}",
                    f"run tests or inspect status after updating {arch['mod']}",
                ],
            })
    return tasks


def evaluate_frozen_advisor_benchmark(
    top_k: int = 4,
    turns_per_session: int = 3,
    cached_discount: float = 0.10,
    context_tokens_per_retry: int = 1200,
) -> Dict[str, Any]:
    """Run the 120-task multi-label advisor & session prompt-cache benchmark."""
    ex06 = _load_ex06()
    schemas = ex06.build_32_tool_schemas()
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    all_names = list(schema_by_name.keys())
    descriptions = {s["function"]["name"]: s["function"]["description"] for s in schemas}
    full_tokens = ex06.count_schema_tokens(schemas)
    tasks = build_frozen_120_multilabel_tasks()
    all_set = set(all_names)
    assert all(set(t["acceptable_tools"]).issubset(all_set) for t in tasks)
    moe = mb.ToolRoutingMoE(d_model=256)

    def _kw_scored(q: str) -> Dict[str, Any]:
        q_toks = set(re.findall(r"(?u)\b\w+\b", q.lower()))
        scored = []
        for n in all_names:
            d_toks = set(re.findall(r"(?u)\b\w+\b", f"{n.replace('_', ' ')} {descriptions[n]}".lower()))
            scored.append((len(q_toks & d_toks), n))
        scored.sort(key=lambda kv: (-kv[0], kv[1]))
        top_sc = scored[0][0] if scored else 0
        k_ref = scored[min(top_k, len(scored) - 1)][0] if scored else 0
        margin = top_sc - k_ref
        nat_esc = (top_sc < 2) or (margin < 1)
        return {
            "mode": "escalate_full_catalog" if nat_esc else "narrowed",
            "candidate_tools": list(all_names) if nat_esc else [n for _, n in scored[:top_k]],
            "top_k_tools": [n for _, n in scored[:top_k]],
            "score": float(top_sc) + 0.5 * float(margin),
            "naturally_escalated": nat_esc,
        }

    def _kw_frozen_select(q: str) -> Dict[str, Any]:
        r = _kw_scored(q)
        return {"mode": r["mode"], "candidate_tools": r["candidate_tools"]}

    def _mb_per_turn_select(q: str) -> Dict[str, Any]:
        return moe.narrow_tool_surface(q, available_tools=all_names, top_k=top_k, tool_descriptions=descriptions)

    # First collect Keyword & Microbrains per-turn picks to measure agreed-narrowed error rate
    kw_cands: List[Dict[str, Any]] = [_kw_frozen_select(t["query"]) for t in tasks]
    mb_cands: List[Dict[str, Any]] = [_mb_per_turn_select(t["query"]) for t in tasks]

    agreed_narrowed_total = 0
    agreed_narrowed_drops = 0
    for t, kw_r, mb_r in zip(tasks, kw_cands, mb_cands):
        if kw_r["mode"] == "narrowed" and mb_r["mode"] == "narrowed":
            agreed_narrowed_total += 1
            acc = set(t["acceptable_tools"])
            if not (acc & set(mb_r["candidate_tools"])) or not (acc & set(kw_r["candidate_tools"])):
                agreed_narrowed_drops += 1

    def _eval_strategy(name: str, mode: str) -> Dict[str, Any]:
        dropped = 0
        escalated = 0
        narrowed = 0
        narrowed_hits = 0
        uncached_tokens = 0
        effective_cached_tokens = 0
        lat_ms: List[float] = []

        for t in tasks:
            acc = set(t["acceptable_tools"])
            turn_queries = [t["query"], *t["followup_queries"]][: max(1, int(turns_per_session))]

            sticky = mb.SessionToolSurface(all_names, top_k=top_k, tool_descriptions=descriptions, moe=moe) if mode == "sticky" else None
            prev_cands: List[str] = []

            for turn_idx, q in enumerate(turn_queries):
                t0 = time.perf_counter_ns()
                if mode == "full":
                    cands = list(all_names)
                    turn_mode = "escalate_full_catalog"
                elif mode == "keyword":
                    res = _kw_frozen_select(q)
                    cands = list(res["candidate_tools"])
                    turn_mode = res["mode"]
                elif mode == "per_turn":
                    res = _mb_per_turn_select(q)
                    cands = list(res["candidate_tools"])
                    turn_mode = res["mode"]
                else:
                    assert sticky is not None
                    res = sticky.select_for_turn(q)
                    cands = list(res["candidate_tools"])
                    turn_mode = "escalate_full_catalog" if len(cands) >= len(all_names) else "narrowed"
                lat_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

                toks_this_turn = ex06.count_schema_tokens([schema_by_name[c] for c in cands if c in schema_by_name])
                uncached_tokens += toks_this_turn
                if turn_idx == 0 or cands != prev_cands:
                    # Prefix changed or cold start: pay 1.0x input cost on schema prefix
                    effective_cached_tokens += toks_this_turn
                else:
                    # Prefix preserved across turn: prompt cache hit pays `cached_discount`
                    effective_cached_tokens += int(round(toks_this_turn * cached_discount))
                prev_cands = list(cands)

                if turn_idx == 0:
                    hit = bool(acc & set(cands))
                    if not hit:
                        dropped += 1
                        uncached_tokens += full_tokens + context_tokens_per_retry
                        effective_cached_tokens += full_tokens + context_tokens_per_retry
                        if sticky is not None:
                            cands = sticky.expand_to_full()
                            prev_cands = list(cands)
                    if turn_mode == "narrowed":
                        narrowed += 1
                        if hit:
                            narrowed_hits += 1
                    else:
                        escalated += 1

        n = len(tasks)
        full_uncached_total = n * turns_per_session * full_tokens
        full_cached_baseline = n * (full_tokens + (turns_per_session - 1) * int(round(full_tokens * cached_discount)))
        uncached_saved_pct = (1.0 - (uncached_tokens / float(full_uncached_total))) * 100.0
        cached_saved_pct = (1.0 - (effective_cached_tokens / float(full_cached_baseline))) * 100.0
        drop_pct = (dropped / float(n)) * 100.0
        cp95 = mb.AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(dropped, n, alpha=0.05) * 100.0
        lat_ms.sort()
        return {
            "strategy": name,
            "recall_at_k_pct": round(100.0 - drop_pct, 1),
            "narrowed_recall_pct": round((narrowed_hits / float(narrowed)) * 100.0, 1) if narrowed else 100.0,
            "escalated": f"{escalated}/{n}",
            "narrowed": f"{narrowed}/{n}",
            "dropped": f"{dropped}/{n}",
            "cp95_upper_drop_pct": round(cp95, 1),
            "uncached_net_saved_pct": round(uncached_saved_pct, 1),
            "cache_adjusted_net_saved_pct": round(cached_saved_pct, 1),
            "p95_ms": round(lat_ms[int(len(lat_ms) * 0.95)], 3) if lat_ms else 0.0,
        }

    rows = [
        _eval_strategy("Full 32-Tool Catalog (Baseline)", "full"),
        _eval_strategy("Simple Name+Desc Keyword (Frozen Rule)", "keyword"),
        _eval_strategy("Microbrains Top-4 (Per-Turn)", "per_turn"),
        _eval_strategy("Microbrains Top-4 (Session-Sticky)", "sticky"),
    ]

    # Separate holdout Top-K evaluation on the MoE instance
    holdout_report = moe.evaluate_topk_holdout(
        tasks[:40],
        top_k=top_k,
        available_tools=all_names,
        tool_descriptions=descriptions,
        sample_id="frozen_advisor_40",
    )

    # Equal-escalation-rate & Risk-Coverage AURC comparison at matched escalation budget
    kw_scored = []
    mb_scored = []
    for t in tasks:
        kw_r = _kw_scored(t["query"])
        kw_scored.append({
            "id": t["id"],
            "acceptable_tools": set(t["acceptable_tools"]),
            "top_k_tools": list(kw_r["top_k_tools"]),
            "score": float(kw_r["score"]),
            "naturally_escalated": bool(kw_r["naturally_escalated"]),
        })
        mb_r = moe.rank_tools(
            t["query"],
            available_tools=all_names,
            top_k=top_k,
            tool_descriptions=descriptions,
        )
        mb_scored.append({
            "id": t["id"],
            "acceptable_tools": set(t["acceptable_tools"]),
            "top_k_tools": list(mb_r["recommended_top_k"][:top_k]),
            "score": float(mb_r["confidence"]),
            "naturally_escalated": bool(mb_r["should_escalate_full_catalog"]),
        })

    target_escalations = sum(1 for x in mb_scored if x["naturally_escalated"])

    def _eval_at_equal_esc(sel_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_items = len(items)
        nat_esc = sum(1 for x in items if x["naturally_escalated"])
        eff_esc = min(n_items, max(target_escalations, nat_esc))
        ordered = sorted(
            items,
            key=lambda x: (0 if x["naturally_escalated"] else 1, float(x["score"]), x["id"]),
        )
        esc_ids = {x["id"] for x in ordered[:eff_esc]}

        eq_dropped = 0
        eq_narrowed = 0
        eq_narrowed_hits = 0
        for x in items:
            if x["id"] in esc_ids:
                continue
            eq_narrowed += 1
            if x["acceptable_tools"] & set(x["top_k_tools"]):
                eq_narrowed_hits += 1
            else:
                eq_dropped += 1

        aurc_ordered = sorted(
            items,
            key=lambda x: (1 if x["naturally_escalated"] else 0, -float(x["score"]), x["id"]),
        )
        cum_drops = 0
        risk_sum = 0.0
        for k_cov, x in enumerate(aurc_ordered, start=1):
            cands_k = all_names if x["naturally_escalated"] else x["top_k_tools"]
            if not (x["acceptable_tools"] & set(cands_k)):
                cum_drops += 1
            risk_sum += cum_drops / float(k_cov)
        aurc_drop_risk = round(risk_sum / float(n_items), 4) if n_items else 0.0

        eq_drop_pct = (eq_dropped / float(n_items)) * 100.0 if n_items else 0.0
        eq_cp95 = mb.AnalyticalMicroTransformer.compute_clopper_pearson_upper_bound(eq_dropped, n_items, alpha=0.05) * 100.0
        return {
            "selector": sel_name,
            "target_escalations": target_escalations,
            "natural_escalations": nat_esc,
            "effective_escalations": eff_esc,
            "narrowed": f"{eq_narrowed}/{n_items}",
            "dropped": f"{eq_dropped}/{n_items}",
            "recall_at_k_pct": round(100.0 - eq_drop_pct, 1),
            "narrowed_recall_pct": round((eq_narrowed_hits / float(eq_narrowed)) * 100.0, 1) if eq_narrowed else 100.0,
            "cp95_upper_drop_pct": round(eq_cp95, 1),
            "aurc_drop_risk": aurc_drop_risk,
        }

    equal_esc_comparison = {
        "target_escalations": target_escalations,
        "rows": [
            _eval_at_equal_esc("Simple Name+Desc Keyword", kw_scored),
            _eval_at_equal_esc("Microbrains Top-4", mb_scored),
        ],
    }

    return {
        "n_tasks": len(tasks),
        "turns_per_session": turns_per_session,
        "cached_discount": cached_discount,
        "full_32_tool_schema_tokens": full_tokens,
        "agreed_narrowed_total": agreed_narrowed_total,
        "agreed_narrowed_drops": agreed_narrowed_drops,
        "rows": rows,
        "equal_escalation_comparison": equal_esc_comparison,
        "moe_topk_holdout_report": holdout_report,
    }


def run_patch_advisor_check() -> Dict[str, Any]:
    """Evaluate `diagnose_patch()` across clean patches, secret leaks, obfuscated sinks, and partial hunks."""
    cases = [
        ("clean", "src/math_helper.py", "def add(a: int, b: int) -> int:\n    return a + b\n", None, "clean"),
        ("secret", "src/cfg.py", 'KEY = "sk-proj-1234567890abcdef1234567890abcdef"\n', None, "findings_detected"),
        ("eval_sink", "src/run.py", "eval(user_expr)\n", None, "findings_detected"),
        ("protected", ".ssh/id_rsa", "PRIVATE KEY MATERIAL\n", None, "policy_blocked_path"),
        ("partial_hunk", "src/app.py", "@@ -4,2 +4,3 @@\n+    return x + 1\n", None, "clean"),
    ]
    checked = []
    exact = 0
    for tag, path, diff, orig, expected in cases:
        diag = mb.diagnose_patch(path, diff, original_source=orig)
        assert diag["authorizes_execution"] is False
        ok = diag["status"] == expected
        if tag == "partial_hunk":
            ok = ok and diag["hunk_only_checked"] is True
        if ok:
            exact += 1
        checked.append({"case": tag, "status": diag["status"], "findings": diag["findings"], "hunk_only": diag["hunk_only_checked"]})
    return {"cases": len(cases), "expected_status_matched": exact, "details": checked}


def main() -> None:
    rep = evaluate_frozen_advisor_benchmark()
    print("=== Frozen 120-Task Multi-Label Tool Advisor & Prompt-Cache Benchmark ===")
    print(f"Schema size: {rep['full_32_tool_schema_tokens']:,} tok | Turns/session: {rep['turns_per_session']} | Cached token rate: {rep['cached_discount']}x")
    print(f"Agreed narrowed pairs (Keyword & Microbrains): {rep['agreed_narrowed_total']} | Dropped in agreed pairs: {rep['agreed_narrowed_drops']}\n")
    print("| Strategy | Recall@4 | Narrowed Recall | Narrowed | Dropped | 95% CP Upper Drop | Uncached Net Saved | Cache-Adjusted Net Saved | Local p95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rep["rows"]:
        print(
            f"| {r['strategy']} | {r['recall_at_k_pct']}% | {r['narrowed_recall_pct']}% | "
            f"{r['narrowed']} | {r['dropped']} | <= {r['cp95_upper_drop_pct']}% | "
            f"{r['uncached_net_saved_pct']}% | {r['cache_adjusted_net_saved_pct']}% | {r['p95_ms']:.3f} ms |"
        )
    eq = rep["equal_escalation_comparison"]
    print(f"\n=== Equal-Escalation-Rate Comparison (Matched Budget: {eq['target_escalations']}/120 escalated) ===")
    print("| Selector | Escalated | Narrowed | Dropped | Recall@4 | Narrowed Recall | 95% CP Upper Drop | AURC Drop Risk |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in eq["rows"]:
        print(
            f"| {r['selector']} | {r['effective_escalations']}/120 (nat {r['natural_escalations']}) | "
            f"{r['narrowed']} | {r['dropped']} | {r['recall_at_k_pct']}% | "
            f"{r['narrowed_recall_pct']}% | <= {r['cp95_upper_drop_pct']}% | {r['aurc_drop_risk']:.4f} |"
        )
    print("\n=== Decoupled Patch Diagnostic Advisor Check ===")
    patch_rep = run_patch_advisor_check()
    print(f"Matched {patch_rep['expected_status_matched']}/{patch_rep['cases']} diagnostic cases (authorizes_execution=False).")


if __name__ == "__main__":
    main()
