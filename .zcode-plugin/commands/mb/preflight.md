# /mb:preflight — Aetheris Microbrains Dry-Run Pre-Flight Check

Evaluate `$ARGUMENTS` strictly through the Aetheris Microbrains pre-flight gate **without executing it**. The verdict is the deliverable — **never execute the proposed command or patch**.

1. Construct a JSON hook payload for `Bash` (or `Edit` if `$ARGUMENTS` looks like a file path / patch) and pipe it into:
   `python -m aetheris_microbrains.hook --host zcode --session-state auto`
2. Report:
   - Verdict (`allow`, `ask`, or `deny`) and exit code (`0` or `2`)
   - Exact `reason_code` / label from stdout/stderr
   - Safe alternative if `deny`.
