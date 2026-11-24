@echo off
cd /d "%~dp0"
python -m aetheris_microbrains.mcp_server --http --port 8080 %*
