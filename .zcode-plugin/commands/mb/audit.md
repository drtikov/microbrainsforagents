# /mb:audit — Aetheris Microbrains Decision Journal Forensics

Inspect and explain Aetheris Microbrains pre-flight hook decisions using the arguments: `$ARGUMENTS`.

1. Run the appropriate audit command (do **not** modify repository files):
   - Default (empty args): `python -m aetheris_microbrains.audit --last --explain`
   - If `$ARGUMENTS` matches `--tail`, `--summary`, `--daily`, `--savings`, `--find <sha>`, or `--propose-rules`, pass `$ARGUMENTS` directly to `python -m aetheris_microbrains.audit`.
2. Summarize:
   - What tool/command triggered the decision
   - The `reason_code` and `label`
   - One concrete safe fix or next step.
