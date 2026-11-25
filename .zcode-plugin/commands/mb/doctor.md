# /mb:doctor — Aetheris Microbrains Environment & Gate Self-Test

Verify that Aetheris Microbrains is properly wired in this ZCode session (extra flags: `$ARGUMENTS`).

1. Check engine import path (`python -c "import aetheris_microbrains; print(aetheris_microbrains.__file__)"`) and compare against `ZCODE_PLUGIN_ROOT` / `CLAUDE_PLUGIN_ROOT` if set.
2. Run:
   - `python -m aetheris_microbrains.install_zcode --self-test`
   - `python -m aetheris_microbrains.doctor`
3. Report engine location, Python version, red-team `deny` (`rc=2`) / safe `allow` (`rc=0`) probe status, and decision journal status in a compact table.
