@echo off
title Multi-Agent System
color 0A
echo.
echo  ══════════════════════════════════════════════
echo   Multi-Agent System — Starting...
echo  ══════════════════════════════════════════════
echo.
set PORT=8001

echo  Open your browser at:  http://localhost:%PORT%
echo  Press Ctrl+C to stop.
echo.

:: Change to script's directory
cd /d "%~dp0"

:: Check Ollama
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Ollama does not appear to be running at localhost:11434
    echo         Start it with: ollama serve
    echo.
)

uvicorn main:app --host 0.0.0.0 --port %PORT% --reload
pause
