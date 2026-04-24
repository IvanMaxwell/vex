@echo off
title Multi-Agent System — Setup
color 0B
echo.
echo  ══════════════════════════════════════════════
echo   Multi-Agent System — Windows 10 Setup
echo  ══════════════════════════════════════════════
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.11+ from python.org
    pause & exit /b 1
)
echo  [OK] Python found.

:: Upgrade pip
echo.
echo  [1/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install PyTorch (CPU-only, lighter)
echo  [2/5] Installing PyTorch (CPU-only)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

:: Install core deps
echo  [3/5] Installing core dependencies...
pip install fastapi uvicorn[standard] langchain langchain-community langchain-ollama langchain-core langgraph --quiet

:: Install tool deps
echo  [4/5] Installing tool dependencies...
pip install psutil pywin32 wmi chromadb sentence-transformers pyautogui pyperclip aiofiles python-multipart jinja2 --quiet

:: Pre-download MiniLM model
echo  [5/5] Pre-downloading all-MiniLM-L2-v2 embedding model...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L2-v2'); print('  Model ready.')" 2>nul || echo  [WARN] Could not pre-download model (will download on first run).

:: Create dirs
if not exist "agent_workspace" mkdir agent_workspace
if not exist "data" mkdir data
if not exist "data\chroma_db" mkdir data\chroma_db

echo.
echo  ══════════════════════════════════════════════
echo   Setup complete!
echo   Make sure Ollama is running:
echo     ollama serve
echo     ollama pull phi4-mini
echo.
echo   Then run:  run.bat
echo  ══════════════════════════════════════════════
echo.
pause
