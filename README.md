# 🤖 Multi-Agent System

A fully-featured, self-contained multi-agent AI app running on **Windows 10** using
**LangGraph**, **LangChain**, and **Ollama** (Phi4-mini by default).

---

## What it does

Four AI agents take turns using the **same Ollama model** with different system prompts:

| Agent | Role |
|---|---|
| 📋 **Planner** | Reads your task and creates a numbered step-by-step plan |
| ⚙️ **Executor** | Executes each plan step using tools |
| 🔍 **Researcher** | Searches the web (DuckDuckGo) and summarises findings |
| 🖥️ UI Controller | Controls your desktop — opens apps, types text, clicks |

---

## Quick Start

### 1. Install Ollama

Download from https://ollama.com — then run:

```bat
ollama serve
ollama pull phi4-mini
```

### 2. Set up the app

```bat
setup.bat
```

### 3. Run

```bat
run.bat
```

Open **http://localhost:8001** in your browser.

---

## Changing the model

Edit `config.py` — change the first line:

```python
MODEL_NAME = "phi4-mini:latest"   # ← change to "llama3.2", "mistral", etc.
```

---

## Memory Layers

| Layer | Storage | Purpose |
|---|---|---|
| **Short-term** | In-memory dict | Current session results & preferences |
| **Long-term** | `data/long_term_memory.json` | User preferences, task history, learned facts |
| **Vector** | ChromaDB + all-MiniLM-L2-v2 | Semantic similarity search over past tasks |

---

## Tools

| Tool | What it does |
|---|---|
| `file_ops` | Read, write, copy, move, delete, find files |
| `web_search` | DuckDuckGo search — no API key needed |
| `terminal_exec` | Run any shell / PowerShell command |
| `system_tool` | CPU, RAM, GPU (nvidia-smi), disk, top processes |
| `ui_control` | Open apps, type text, hotkeys, click, screenshot, WhatsApp |

**Every tool requires your permission before it runs.** A dialog will pop up.

---

## Permission System

Before any tool executes, you see a modal showing:
- Which tool wants to run
- Exactly what it will do
- The parameters it will use

Click **Allow** or **Deny**. You have 180 seconds before it times out (auto-denied).

---

## Workspace (Safe Sandbox)

Agents write to `agent_workspace/` inside the app folder.
Your original files are **never modified** unless you explicitly allow a move operation.

---

## Folder Structure

```
multi_agent_system/
├── main.py              FastAPI app + WebSocket server
├── graph.py             LangGraph StateGraph
├── agents.py            4 agent node functions
├── tools.py             All tools with permission gates
├── memory_system.py     3-layer memory
├── runtime.py           Streamer + PermissionManager
├── state.py             Shared AgentState TypedDict
├── config.py            ← Change MODEL_NAME here
├── templates/
│   └── index.html       Web UI
├── agent_workspace/     Safe sandbox for agent file ops
├── data/
│   ├── long_term_memory.json
│   └── chroma_db/       Vector embeddings
├── setup.bat            One-time install
└── run.bat              Start the server
```

---

## Example Tasks

- `check if my CPU GPU RAM usage are normal`
- `check Apple stock price`
- `find all PDF files in Downloads and put them in a new folder called PDF`
- `open WhatsApp and type hi to mom`
- `plan a simple PC cleanup without touching my project folders`
- `what is my disk usage?`

---

## Requirements

- Windows 10
- Python 3.11+
- Ollama running locally
- ~4 GB RAM for the model + embeddings
