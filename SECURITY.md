# Security Policy, Threat Model & Rule Governance

This document describes how security boundaries are defined, audited, and updated in **Aetheris Microbrains**, addressing the threat model of autonomous AI agent execution.

---

## 🎯 Threat Model & Architecture Boundary

When autonomous AI agents (Claude Code, SWE-agent, Codex, or custom LLM bots) execute commands or apply code edits on your machine, they operate with broad system privileges. They face several primary threat vectors:

1. **Indirect Prompt Injection & Poisoned Context:** Untrusted third-party inputs (web pages, user issues, pull request descriptions, or MCP tools) containing adversarial instructions (`<system>`, `[OVERRIDE]`, Unicode tag smuggling) attempting to hijack the agent.
2. **Credential Exfiltration & In-Band Leaks:** Attempts to read `.env`, `.ssh/id_rsa`, `~/.aws/credentials`, or environment variables and transmit them via network sinks (`curl`, `requests.post`, `socket`).
3. **Destructive Shell Chains & File System Mutations:** Destructive commands (`rm -rf`, `dd of=/dev/`, `DROP DATABASE`, modifying git hooks or CI/CD pipelines).
4. **Syntax Corruption & Malformed Diffs:** Generating code with syntax errors or breaking API signatures that trap the agent in expensive trial-and-error debugging loops.
5. **MCP Supply-Chain Rug-Pulls:** Malicious or compromised MCP tool servers dynamically altering schemas after registration.

---

## 🛡️ Beyond Naive Regex: Normalization & AST Parsing

A common weakness of naive security filters is relying purely on pattern-matching (e.g. searching for the string `"rm -rf"`). Microbrains prevents evasion by operating at the **structural syntax and tokenizer layers**:

### 1. Shell Normalization & Wrapper Unwrapping (`Layer 1`)
Attackers or complex scripts often wrap dangerous commands behind wrappers or environment variables. Before evaluating shell rules, Microbrains:
* **Unwraps Execution Wrappers:** Recursively strips and analyzes commands wrapped in `xargs`, `find -exec`, `env`, `sudo`, `timeout`, `nohup`.
* **Inspects Inline Environment Overrides:** Parses inline variables like `LD_PRELOAD=...`, `PYTHONPATH=...` for dangerous hijack attempts.
* **Handles Newline & Subshell Injection:** Catches chained newline-separated commands (`echo ok\nrm -rf /`) and command substitutions (`` `cmd` `` and `$(cmd)`).
* **Enforces Allowlist-First Execution:** Unrecognized shell commands and uninspected tools do not auto-pass—they escalate to the supervisor (`"ask"`).

### 2. True Python Abstract Syntax Tree (AST) Parsing (`Layer 1`)
Diffs are applied in memory and inspected using Python's native `ast.parse`:
* **Call Graph Analysis:** Detects `subprocess.Popen(..., shell=True)` and `shutil.rmtree()` regardless of argument formatting or whitespace.
* **Obfuscated Dynamic Calls:** Detects dynamic imports (`importlib.import_module`), eval chains (`eval`, `exec`), and encoded payload execution (`base64.b64decode` into execution).
* **Secret Exfiltration Sinks (`ast_secret_exfil_pattern`):** Traverses the AST for sensitive environment variable lookups (`os.getenv("AWS_SECRET_ACCESS_KEY")`) flowing directly into network sinks (`urllib`, `requests`, `httpx`, `socket`).

### 3. Session Taint Gate (`SessionSecurityState`)
Tracks taint across multi-step agent interactions:
* If an agent ingests untrusted external data (e.g., fetches an external web page) and subsequently accesses sensitive credentials, any subsequent mutation or outbound network request triggers an immediate hard denial (`decision: "deny"`).

---

## ⚙️ Rule Governance & Custom Policy Configuration

Organizations can configure and update rules dynamically without touching engine code:

### Loading Policies from File
Policies can be loaded from JSON or YAML files via `PolicyConfig.from_file()`:

```python
from aetheris_microbrains import PolicyConfig, decide

# Load custom corporate security policy
policy = PolicyConfig.from_file("my_policy.yaml")

# Run pre-flight check with custom rules
verdict = decide(payload, policy=policy)
```

Example `my_policy.yaml`:
```yaml
# Require human confirmation for any external network calls
require_approval_for_external: true

# Require human confirmation for irreversible operations (git push, delete_file)
require_approval_for_irreversible: true

# Strict Python AST import allowlist
strict_ast_allowlist: true
allowed_ast_imports:
  - math
  - json
  - typing
  - dataclasses

# Additional protected project files
protected_patterns:
  - ".env*"
  - ".git/hooks/"
  - "deploy/production.yaml"
  - "secrets/"
```

---

## 🔍 Forensic Auditability & The Decision Journal

Every decision leaves an auditable, cryptographic trace:

### What Is Logged (`memory/decision_journal.jsonl`):
* `decision`: `"allow"`, `"deny"`, or `"escalate"`
* `layer`: Triggering layer (0 = Policy, 1 = AST/Shell, 2 = Conformal MoE, 3 = Supervisor)
* `reason_code`: Machine-readable reason (e.g. `layer1_ast_sink:subprocess_shell_true`, `layer2_conformal_abstention_ood`)
* `label`: Human-readable explanation of why the rule fired
* `input_sha256`: SHA-256 fingerprint of the exact command or diff evaluated
* `payload_preview`: Redacted context preview snippet of the command/code that triggered the rule
* `upper_error_bound_95`: Statistical conformal bound (when ML routing is used)
* `latency_ms`: Execution time in milliseconds

### Inspecting Audits via CLI
Humans can immediately inspect why an agent was stopped or verify recent decisions using the audit CLI:

```bash
# View the last decision event with payload preview and reason
python -m aetheris_microbrains.audit --last

# View the last 5 decision events
python -m aetheris_microbrains.audit --tail 5

# View aggregate summary and latency statistics
python -m aetheris_microbrains.audit --summary
```
