"""
agents.py — Four agent nodes for LangGraph.

All four use the SAME Ollama LLM but with distinct system prompts.
Streaming tokens go to the frontend via runtime.streamer.
"""
import json
import re
from urllib import error as urllib_error
from urllib import request as urllib_request

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    MODEL_NAME,
    MODEL_FALLBACKS,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
)
from state import AgentState
from runtime import streamer
from tools import dispatch_tool


# ══════════════════════════════════════════════════════════════
#  LLM SINGLETON  ← change MODEL_NAME in config.py to swap
# ══════════════════════════════════════════════════════════════
_llm_cache: dict[str, ChatOllama] = {}
_active_model = MODEL_NAME
_announced_model: str | None = None


def _build_llm(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
        num_ctx=OLLAMA_NUM_CTX,
    )


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
TOOL_NAMES = "file_ops | web_search | terminal_exec | system_tool | ui_control"
EXECUTOR_TOOL_NAMES = "file_ops | terminal_exec | system_tool"
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "about",
    "have", "what", "when", "where", "which", "would", "could", "should", "please",
    "task", "then", "than", "them", "they", "their", "user", "need", "show", "tell",
}


def _get_installed_models() -> list[str]:
    try:
        with urllib_request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", [])
        models.sort(key=lambda item: item.get("size", 0))
        return [item.get("name", "") for item in models if item.get("name")]
    except (OSError, ValueError, urllib_error.URLError):
        return []


def _candidate_models() -> list[str]:
    ordered: list[str] = []
    for model_name in [_active_model, MODEL_NAME, *MODEL_FALLBACKS, *_get_installed_models()]:
        if model_name and model_name not in ordered:
            ordered.append(model_name)
    return ordered


def _get_llm(model_name: str) -> ChatOllama:
    llm = _llm_cache.get(model_name)
    if llm is None:
        llm = _build_llm(model_name)
        _llm_cache[model_name] = llm
    return llm


def _is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "model requires more system memory",
        "not found",
        "no such file",
        "connection refused",
        "failed to connect",
        "404",
        "500",
    )
    return any(marker in text for marker in markers)


def _is_tool_failure(result: str) -> bool:
    normalized = result.strip().lower()
    return (
        normalized.startswith("error")
        or normalized.startswith("unknown tool")
        or normalized.startswith("tool parameter error")
        or "permission denied" in normalized
        or normalized == "launched:"          # open_app with empty target
        or normalized.startswith("launched:\n")
    )


def _normalize_tool_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    aliases = {
        "search":         "web_search",
        "websearch":      "web_search",
        "browser_search": "web_search",
        "terminal":       "terminal_exec",
        "shell":          "terminal_exec",
        "system":         "system_tool",
        "ui":             "ui_control",
        "whatsapp":       "ui_control",
    }
    return aliases.get(normalized, normalized)


def _clean_llm_output(text: str) -> str:
    """Strip LLM reasoning noise and pure XML garbage, but KEEP tool-call XML."""
    cleaned = text or ""
    # Remove <think>...</think> reasoning blocks (Qwen3, DeepSeek, R1, etc.)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Remove PURE NOISE tags that never contain tool calls:
    # output, memory, observation, reflection — NOT tool/action/use/step/plan
    _noise_tags = r"output|memory|observation|reflection|context|reasoning|thought"
    cleaned = re.sub(
        fr"<({_noise_tags})[^>]*>.*?</\1>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip any lone leftover noise tags (unclosed / self-closing)
    cleaned = re.sub(
        fr"</?(?:{_noise_tags})[^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip Ollama / HuggingFace special tokens  (<|...|>)
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
    cleaned = cleaned.replace("</s>", "").replace("<s>", "")
    return cleaned.strip()


def _stream_llm(system: str, human: str, agent_name: str) -> str:
    """Call LLM and stream each chunk to the frontend. Returns full text.
    Stops early once a complete TOOL_CALL block is detected to prevent
    the model from repeating the same tool call dozens of times.
    """
    global _active_model, _announced_model

    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    last_error: Exception | None = None

    # Stop tokens: stop streaming as soon as we have a complete tool call
    _STOP_MARKERS = ("END_TOOL_CALL", "</tool_call>", "<|im_end|>", "<|endoftext|>")

    for model_name in _candidate_models():
        try:
            if _announced_model != model_name:
                streamer.send({
                    "type": "model_info",
                    "agent": "System",
                    "content": model_name,
                    "num_ctx": OLLAMA_NUM_CTX,
                })
                streamer.info(
                    f"Using Ollama model '{model_name}' with num_ctx={OLLAMA_NUM_CTX}.",
                    "System",
                )
                _announced_model = model_name

            full = ""
            stopped_early = False
            _xml_spam_count = 0   # track repeating noise-tag lines
            for chunk in _get_llm(model_name).stream(messages):
                token = chunk.content or ""
                if isinstance(token, list):
                    token = " ".join(str(t) for t in token)
                else:
                    token = str(token)
                full += token
                if token:
                    streamer.thought(token, agent_name)
                # Stop as soon as a complete tool call block is present
                cleaned_so_far = _clean_llm_output(full)
                if any(marker in cleaned_so_far for marker in _STOP_MARKERS):
                    stopped_early = True
                    break
                # Spam-detection: abort if the raw output is mostly XML noise tags
                # (Qwen3 sometimes enters a loop emitting <output>…<output>…<memory>…)
                if len(full) > 300:
                    noise_matches = len(re.findall(
                        r"<(?:output|memory|search|result|observation|reflection)[^>]*>",
                        full, re.IGNORECASE
                    ))
                    if noise_matches >= 4:
                        streamer.error(
                            "⚠️ XML spam detected — stopping generation early.",
                            agent_name,
                        )
                        stopped_early = True
                        break

            if not stopped_early and not _clean_llm_output(full):
                fallback = _get_llm(model_name).invoke(messages)
                full = getattr(fallback, "content", "") or full
            _active_model = model_name
            return _clean_llm_output(full)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_llm_error(exc):
                raise
            streamer.error(
                f"Ollama model '{model_name}' failed: {exc}",
                "System",
            )

    raise RuntimeError(
        f"Unable to get a response from Ollama. Last error: {last_error}. "
        f"Try lowering OLLAMA_NUM_CTX or installing a smaller model."
    )


def _parse_tool_calls(text: str) -> list[dict]:
    """
    Parse one or more TOOL_CALL blocks from LLM output.
    Handles:
      1. Qwen XML:  <tool_call><action>name</action><params>{...}</params></tool_call>
      2. JSON-body XML: <tool_call>{"name":"...","params":{...}}</tool_call>
      3. Block format: TOOL_CALL: name / PARAMS: {...} / END_TOOL_CALL
      4. Lenient block: any ordering / extra text
      5. Last-resort: JSON near TOOL_CALL keyword
    Returns only the FIRST complete tool call to prevent duplicates.
    """
    calls: list[dict] = []
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    # ── Format 0: <tool_use> XML (Qwen3-Instruct native format) ──────────
    # <tool_use><tool_name>file_ops</tool_name><arguments>{...}</arguments></tool_use>
    tool_use_pattern = re.compile(
        r"<tool_use>\s*"
        r"<tool_name>\s*([\w_]+)\s*</tool_name>\s*"
        r"<arguments>\s*(\{.*?\})\s*</arguments>\s*"
        r"</tool_use>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in tool_use_pattern.finditer(text):
        tool = _normalize_tool_name(m.group(1).strip())
        try:
            params = json.loads(m.group(2).strip())
            if tool and isinstance(params, dict):
                calls.append({"tool": tool, "params": params})
                break
        except Exception:
            pass
    if calls:
        return calls[:1]

    # ── Format 1a: Qwen native XML with sub-tags ──────────────────────
    # <tool_call><action>ui_control</action><params>{...}</params></tool_call>
    qwen_pattern = re.compile(
        r"<tool_call>\s*"
        r"<action>\s*([\w_]+)\s*</action>\s*"
        r"<params>\s*(\{.*?\})\s*</params>\s*"
        r"</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in qwen_pattern.finditer(text):
        tool = _normalize_tool_name(m.group(1).strip())
        try:
            params = json.loads(m.group(2).strip())
            if tool and isinstance(params, dict):
                calls.append({"tool": tool, "params": params})
                break  # only first
        except Exception:
            pass
    if calls:
        return calls[:1]

    # ── Format 1b: JSON-body XML ──────────────────────────────────────
    # <tool_call>{"name": "...", "params": {...}}</tool_call>
    tag_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in tag_pattern.finditer(text):
        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            continue
        tool = _normalize_tool_name(payload.get("name", ""))
        params = payload.get("arguments", payload.get("params", {}))
        if tool and isinstance(params, dict):
            calls.append({"tool": tool, "params": params})
            break
    if calls:
        return calls[:1]

    # ── Format 2: TOOL_CALL: name \n PARAMS: {...} \n END_TOOL_CALL ──
    # Matches even when whitespace / tabs vary between the three lines.
    pattern_strict = re.compile(
        r"TOOL_CALL\s*:\s*(\w[\w_]*)\s*\n"
        r"\s*PARAMS\s*:\s*(\{.*?\})\s*\n"
        r"\s*END_TOOL_CALL",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_strict.finditer(text):
        tool = _normalize_tool_name(m.group(1).strip())
        try:
            params = json.loads(m.group(2).strip())
            if tool and isinstance(params, dict):
                calls.append({"tool": tool, "params": params})
        except Exception:
            pass
    if calls:
        return calls

    # ── Format 3: Very lenient — any ordering / extra text between keywords
    pattern_lenient = re.compile(
        r"TOOL_CALL\s*:?\s*(\w[\w_]*).*?PARAMS\s*:?\s*(\{.*?\}).*?END_TOOL_CALL",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_lenient.finditer(text):
        tool = _normalize_tool_name(m.group(1).strip())
        try:
            params = json.loads(m.group(2).strip())
            if tool and isinstance(params, dict):
                calls.append({"tool": tool, "params": params})
        except Exception:
            pass
    if calls:
        return calls

    # ── Format 4: JSON blocks near TOOL_CALL keywords (last resort) ───
    tool_kw = re.compile(
        r"(?:TOOL_CALL|tool_call|call)\s*[:\s]+(\w[\w_]*)",
        re.IGNORECASE,
    )
    # Support nested braces one level deep
    json_blocks = list(
        re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    )
    for tool_match in tool_kw.finditer(text):
        tool_name = _normalize_tool_name(tool_match.group(1).strip())
        tool_pos  = tool_match.end()
        for jm in json_blocks:
            if jm.start() > tool_pos and jm.start() - tool_pos < 300:
                try:
                    params = json.loads(jm.group(0))
                    if isinstance(params, dict):
                        calls.append({"tool": tool_name, "params": params})
                        break
                except Exception:
                    pass

    return calls


def _heuristic_tool_calls(task: str, step: str, task_type: str = "") -> list[dict]:
    """Last-resort tool selection when the LLM produces no parseable TOOL_CALL."""
    step_l = step.lower()
    task_l = task.lower().strip()

    # ── FILE TASKS: always use file_ops, never ui_control ─────────────
    # If the planner classified this as a file task, force file_ops regardless of step text.
    if task_type == "file" or any(
        term in task_l
        for term in ("create file", "write file", "make file", "create a file",
                     "create a text file", "create folder", "make folder", "make a folder")
    ):
        # Extract filename from task — check for desktop path requests
        desktop_path = "C:/Users/Home/Desktop"
        if "desktop" in task_l:
            # Try to extract a filename
            name_match = re.search(
                r'named?\s+["\']?([\w.-]+)["\']?',
                task, re.IGNORECASE
            )
            fname = (name_match.group(1).strip() if name_match else "output")
            if not fname.endswith(".txt"):
                fname += ".txt"
            return [{"tool": "file_ops", "params": {
                "action": "write",
                "path": f"{desktop_path}/{fname}",
                "content": fname.replace(".txt", ""),
            }}]
        # Generic file write in workspace
        name_match = re.search(
            r'named?\s+["\']?([\w.-]+)["\']?',
            task, re.IGNORECASE
        )
        fname = (name_match.group(1).strip() if name_match else "output")
        if not fname.endswith(".txt"):
            fname += ".txt"
        return [{"tool": "file_ops", "params": {"action": "write", "path": fname, "content": ""}}]

    # system_tool
    if "system_tool" in step_l or any(
        kw in step_l for kw in ("cpu", "ram", "disk", "gpu", "memory usage", "process")
    ):
        action = "full_info"
        for candidate in ("cpu_info", "ram_info", "disk_info", "gpu_info", "processes"):
            if candidate.replace("_", " ").split()[0] in step_l or candidate in step_l:
                action = candidate
                break
        return [{"tool": "system_tool", "params": {"action": action}}]

    # file_ops — detect read/write/list steps
    if any(
        term in step_l
        for term in ("read file", "write file", "create file", "create folder", "list files",
                     "find file", "file_ops", "save to", "load from", "file_ops write",
                     "file_ops read", "file_ops mkdir")
    ):
        path_match = re.search(r'["\']([^"\']+\.[a-zA-Z]{2,5})["\']', step)
        path   = path_match.group(1) if path_match else "output.txt"
        action = (
            "read"  if "read"  in step_l else
            "list"  if "list"  in step_l else
            "mkdir" if "folder" in step_l or "mkdir" in step_l else
            "write"
        )
        return [{"tool": "file_ops", "params": {"action": action, "path": path}}]

    # terminal_exec — detect shell/command steps
    if any(
        term in step_l
        for term in ("run command", "execute", "terminal", "shell",
                     "powershell", "terminal_exec", "run script")
    ):
        cmd_match = re.search(r'["\`]([^"\`]+)["\`]', step)
        cmd = cmd_match.group(1) if cmd_match else "dir"
        return [{"tool": "terminal_exec", "params": {"command": cmd}}]

    # ui_control — only for explicit UI/desktop/WhatsApp actions
    # NOTE: "open " alone is NOT enough — it also matches "open a text editor" which is wrong.
    combined_l = step_l + " " + task_l
    if any(
        term in combined_l
        for term in ("ui_control", "whatsapp", "desktop app", "screen",
                     "click", "hotkey", "open app", "focus window",
                     "open whatsapp", "send message")
    ):
        # WhatsApp — decide between open-only vs send-message
        if "whatsapp" in combined_l:
            contact = _extract_contact_name(step + " " + task)
            msg     = _extract_message_text(step + " " + task)
            if contact or msg:
                return [{"tool": "ui_control", "params": {
                    "action": "open_whatsapp_send",
                    "target": contact,
                    "text":   msg,
                }}]
            return [{"tool": "ui_control", "params": {
                "action": "open_app",
                "target": "whatsapp:",
            }}]
        open_match = re.search(
            r"open\s+([\w.]+(?:\s+[\w.]+)?)",
            combined_l,
            re.IGNORECASE,
        )
        _app_map = {
            "notepad":    "notepad.exe",
            "calculator": "calc.exe",
            "calc":       "calc.exe",
            "paint":      "mspaint.exe",
            "explorer":   "explorer.exe",
            "chrome":     "chrome.exe",
            "edge":       "msedge.exe",
            "word":       "winword.exe",
            "excel":      "excel.exe",
            "settings":   "ms-settings:",
            "store":      "ms-windows-store:",
            "calendar":   "outlookcal:",
            "mail":       "outlookmail:",
            "teams":      "msteams:",
            "slack":      "slack:",
            "spotify":    "spotify:",
        }
        if open_match:
            app_name = open_match.group(1).strip()
            mapped = _app_map.get(app_name.lower())
            if mapped:
                return [{"tool": "ui_control", "params": {"action": "open_app", "target": mapped}}]
        for kw, tgt in _app_map.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", combined_l):
                return [{"tool": "ui_control", "params": {"action": "open_app", "target": tgt}}]
        return [{"tool": "ui_control", "params": {"action": "list_windows"}}]

    # web_search
    if "web_search" in step_l or any(
        term in task_l
        for term in ("price", "quote", "latest", "today", "news", "weather",
                     "search", "find information", "look up", "research")
    ):
        return [{"tool": "web_search", "params": {"query": task, "max_results": 5}}]

    return []


def _extract_contact_name(text: str) -> str:
    patterns = (
        r'contact named\s+"([^"]+)"',
        r"contact named\s+'([^']+)'",
        r'search\s+"([^"]+)"',
        r"search\s+'([^']+)'",
        r'to\s+"([^"]+)"',
        r"to\s+'([^']+)'",
        # Stop before stop-words like 'on', 'via', 'using', 'in', 'whatsapp'
        r"\bto\s+([A-Za-z][\w.-]{1,30})(?:\s+(?:on|via|using|in|through|whatsapp|telegram|slack)|$)",
        r"\bto\s+([A-Za-z][\w.-]{1,20})",  # shorter fallback
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,'\"")
            # Strip trailing stop-words that may have crept in
            name = re.sub(r"\s+(on|via|using|in|through|whatsapp|telegram)$", "", name, flags=re.IGNORECASE)
            return name.strip()
    return ""


def _extract_message_text(text: str) -> str:
    quoted_patterns = (
        r'type\s+"([^"]+)"',
        r"type\s+'([^']+)'",
        r'send\s+"([^"]+)"',
        r"send\s+'([^']+)'",
    )
    for pattern in quoted_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    inline_patterns = (
        r"type\s+(.+?)\s+to\s+[A-Za-z][\w .-]{1,40}",
        r"send\s+(.+?)\s+to\s+[A-Za-z][\w .-]{1,40}",
        r"type\s+(.+)$",
        r"send\s+(.+)$",
    )
    for pattern in inline_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            message = match.group(1).strip(" .,'\"")
            if message and "whatsapp" not in message.lower():
                return message
    return ""


def _ui_fallback_tool_calls(task: str, ui_task: str) -> list[dict]:
    combined = " ".join(part for part in (ui_task, task) if part).strip()
    lowered = combined.lower()

    if "whatsapp" in lowered:
        target = _extract_contact_name(combined)
        text   = _extract_message_text(combined)
        if target or text:
            # Specific contact/message — send a message
            return [{"tool": "ui_control", "params": {
                "action": "open_whatsapp_send",
                "target": target,
                "text":   text,
            }}]
        # No contact/message extracted — just open the app
        return [{"tool": "ui_control", "params": {"action": "open_app", "target": "whatsapp:"}}]

    # Default fallback: list open windows so the agent knows what's running
    return [{"tool": "ui_control", "params": {"action": "list_windows"}}]


def _handoff_for_tool_call(tool_call: dict) -> tuple[str, str] | None:
    tool = tool_call.get("tool", "")
    params = tool_call.get("params", {}) or {}

    if tool == "web_search":
        query = params.get("query") or params.get("topic") or "current step research"
        return ("research", str(query))

    if tool == "ui_control":
        action = params.get("action", "")
        target = params.get("target", "")
        text = params.get("text", "")
        desc = ", ".join(part for part in (action, target, text) if part) or "desktop interaction"
        return ("ui", desc)

    return None


def _results_summary(results: list[str]) -> str:
    """Last 3 results joined for context."""
    return "\n---\n".join(results[-3:]) if results else "No previous results."


def _task_keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]{3,}", text.lower())
        if word not in _STOPWORDS
    }


def _trim_memory_context(task: str, memory_context: str, max_chars: int = 700) -> str:
    """Keep only the most relevant memory lines so prompts stay focused."""
    if not memory_context or memory_context == "No previous context.":
        return "None"

    keywords = _task_keywords(task)
    lines = [line.strip() for line in memory_context.splitlines() if line.strip()]
    scored: list[tuple[int, int, str]] = []

    for idx, line in enumerate(lines):
        lowered = line.lower()
        line_keywords = set(re.findall(r"[a-z0-9]{3,}", lowered))
        score = len(keywords & line_keywords)

        if line.startswith("["):
            score += 5
        if "preference" in lowered:
            score += 4
        if "known facts" in lowered:
            score += 3
        if "recent results" in lowered:
            score += 2
        if "past successful tasks" in lowered and score == 0:
            score -= 2

        if score > 0:
            scored.append((score, idx, line))

    if not scored:
        scored = [(1, idx, line) for idx, line in enumerate(lines[:4])]

    selected: list[str] = []
    total = 0
    for _score, _idx, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        if line in selected:
            continue
        needed = len(line) + (1 if selected else 0)
        if total + needed > max_chars:
            continue
        selected.append(line)
        total += needed

    if not selected:
        clipped = memory_context[:max_chars].strip()
        return clipped + ("..." if len(memory_context) > max_chars else "")

    return "\n".join(selected)


def _infer_task_type(task: str, plan: list[str] | None = None) -> str:
    combined = " ".join(
        part for part in [task, *([] if plan is None else plan)] if part
    ).lower()

    # System tasks
    system_terms = (
        "cpu", "ram", "memory", "gpu", "disk", "storage", "process", "usage",
        "system info",
    )
    if any(term in combined for term in system_terms):
        return "system"

    # File tasks
    file_terms = (
        "file", "folder", "directory", "pdf", "move ", "copy ", "rename ",
        "delete ", "save ", "write ", "read ", "list files", "mkdir",
    )
    if any(term in combined for term in file_terms):
        return "file"

    # UI tasks (made less aggressive, moved below file/system)
    ui_terms = (
        "whatsapp", "desktop", "screen", "click", "press ",
        "hotkey", "focus window", "open app", "open whatsapp", "contact", "chat",
        "send message", "ui_control",
    )
    if any(term in combined for term in ui_terms):
        return "ui"

    # Research tasks
    research_terms = (
        "latest", "today", "price", "quote", "news", "weather", "look up",
        "research", "find information", "who is", "what is",
    )
    if any(term in combined for term in research_terms):
        return "research"

    return "general"


def _is_simple_lookup_task(task: str) -> bool:
    lowered = task.lower().strip()
    blocking_terms = (
        "write ", "create ", "build ", "refactor", "fix ", "install ", "delete ",
        "move ", "copy ", "open ", "launch ", "click ", "type ", "edit ", "code ",
    )
    if any(term in lowered for term in blocking_terms):
        return False

    lookup_terms = (
        "price", "quote", "weather", "time", "date", "latest", "current", "today",
        "news", "who is", "what is", "market cap", "stock",
    )
    return len(lowered.split()) <= 16 and any(term in lowered for term in lookup_terms)


def _direct_lookup_plan(task: str) -> list[str]:
    lowered = task.lower()
    if "stock" in lowered and any(term in lowered for term in ("price", "quote", "market cap", "share")):
        return [
            "Use web_search to get the latest quote from a reputable finance source, and cross-check with one more reputable source only if the first result looks stale or unclear.",
            "Return one concise answer with the price, currency, timestamp or date, and source names, then finish.",
        ]

    if _is_simple_lookup_task(task):
        return [
            "Use the most direct non-UI tool to get the answer in one pass.",
            "Return a concise final answer to the user and finish.",
        ]

    return []


def _normalize_plan(task: str, plan: list[str]) -> list[str]:
    cleaned: list[str] = []
    for step in plan:
        normalized = re.sub(r"\s+", " ", step).strip(" -")
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)

    # Only replace with fast plan when the LLM produced a LONGER plan for
    # what is genuinely a simple lookup — never discard a complex plan.
    fast_plan = _direct_lookup_plan(task)
    if fast_plan and len(cleaned) <= len(fast_plan):
        return fast_plan

    return cleaned if cleaned else fast_plan or [task]


def _system_inspection_plan(task: str) -> list[str]:
    lowered = task.lower()
    system_keywords = ("cpu", "ram", "memory", "gpu", "disk", "storage", "process", "usage")
    if not any(keyword in lowered for keyword in system_keywords):
        return []

    return [
        "Use system_tool to collect the requested system information directly.",
        "Summarize the findings for the user and then finish the task.",
    ]


# ══════════════════════════════════════════════════════════════
#  AGENT 1 — PLANNER
# ══════════════════════════════════════════════════════════════
PLANNER_SYSTEM = """You are the Planner Agent. Your ONLY job is to output a TASK_TYPE line and a numbered plan. Nothing else.

=== MANDATORY OUTPUT FORMAT ===
TASK_TYPE: <general|research|ui|file|system>
1. First step
2. Second step

=== TASK TYPE RULES (choose the MOST specific match) ===
  file     → create/read/write/delete/move/copy/list files or folders on disk
  system   → CPU, RAM, GPU, disk, process info
  research → fetch current/online information from the web
  ui       → control the desktop via WhatsApp, app clicks, screen interaction
  general  → everything else

=== CRITICAL PLANNING RULES ===
- To CREATE or WRITE a file → use file_ops in ONE step. NEVER plan to open Notepad or any editor.
- To READ a file → use file_ops in ONE step.
- To get system info → use system_tool in ONE step.
- To search the web → use web_search in ONE step.
- File/system tasks: 1-2 steps maximum.
- Research tasks: 2 steps maximum (search + summarise).
- NEVER plan steps like "open Notepad", "open browser", "click search bar"
- NEVER produce more steps than the task actually requires.
- Final step: return the result to the user.

=== DO NOT ===
- DO NOT explain your reasoning
- DO NOT add comments or prose
- DO NOT output XML tags like <output>, <memory>, <step>, <think>
- DO NOT output anything except TASK_TYPE and the numbered list
- DO NOT plan to open a text editor to create a file — file_ops does it directly

=== EXAMPLES ===
User: create a text file named hello on the desktop
TASK_TYPE: file
1. Use file_ops write to create C:/Users/Home/Desktop/hello.txt with content "hello".
2. Report to the user that the file was created.

User: what is the bitcoin price today?
TASK_TYPE: research
1. Use web_search to find the current bitcoin price.
2. Return the price with source and timestamp.

User: open WhatsApp and send hi to John
TASK_TYPE: ui
1. Use ui_control open_whatsapp_send with target=John, text=hi.

User: how much RAM is my PC using?
TASK_TYPE: system
1. Use system_tool ram_info to get memory usage.
2. Report the results to the user.

=== AVAILABLE TOOLS ===
  file_ops      → read | write | mkdir | copy | move | delete | list | find
  web_search    → search the internet
  terminal_exec → run PowerShell/cmd commands
  system_tool   → cpu_info | ram_info | disk_info | gpu_info | processes | full_info
  ui_control    → open apps, click, hotkeys, WhatsApp send
"""


def planner_node(state: AgentState) -> AgentState:
    agent = "Planner"
    streamer.status("📋 Creating execution plan…", agent)
    planner_memory = _trim_memory_context(state["task"], state["memory_context"], max_chars=450)

    human = f"""Task: {state['task']}

Memory context:
{planner_memory}

Previous errors (replan if any):
{chr(10).join(state['errors'][-3:]) if state['errors'] else 'None'}

Classify the task and produce a step-by-step plan:"""

    # Check for pre-built fast plans
    pre_system = _system_inspection_plan(state["task"])
    pre_lookup  = _direct_lookup_plan(state["task"])
    pre_plan    = pre_system or pre_lookup

    if pre_plan:
        response  = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(pre_plan))
        task_type = "system" if pre_system else "research"
        plan      = pre_plan
        streamer.info("Using the built-in fast planner for this task.", agent)
    else:
        response = _stream_llm(PLANNER_SYSTEM, human, agent)

        # ── Extract task type ──────────────────────────────────────────
        task_type = _infer_task_type(state["task"])
        type_match = re.search(r"TASK_TYPE\s*:\s*(\w+)", response, re.IGNORECASE)
        if type_match:
            raw_type = type_match.group(1).strip().lower()
            if raw_type in ("general", "research", "ui", "file", "system"):
                task_type = raw_type
        elif task_type != "general":
            streamer.info(
                f"LLM omitted TASK_TYPE; using heuristic classification: {task_type}",
                agent,
            )
        streamer.info(f"Task classified as: {task_type}", agent)

        # ── Extract numbered plan steps ────────────────────────────────
        plan = []
        for line in response.strip().splitlines():
            line = line.strip()
            if re.match(r"^\d+[\.)\-]\s+", line):
                plan.append(re.sub(r"^\d+[\.)\-]\s+", "", line))

        if not plan:
            plan = [response.strip()]  # fallback: treat whole response as single step

    plan = _normalize_plan(state["task"], plan)
    if task_type == "general":
        heuristic_type = _infer_task_type(state["task"], plan)
        if heuristic_type != "general":
            task_type = heuristic_type
            streamer.info(
                f"Adjusted task classification from plan/task heuristics: {task_type}",
                agent,
            )

    # ── UI tasks: collapse to a single atomic step ─────────────────
    # The UI Controller handles all sub-steps internally, so multi-step
    # plans for UI tasks only cause planner→UI→executor→UI bounce loops.
    if task_type == "ui" and len(plan) > 1:
        plan = [state["task"]]  # entire task as one step
        streamer.info("UI task: collapsed to single-step plan.", agent)

    # ── Determine routing from task type ──────────────────────────────
    research_needed = task_type == "research"
    ui_needed       = task_type == "ui"

    streamer.status(f"\u2705 Plan ready — {len(plan)} steps (type: {task_type})", agent)
    streamer.send({"type": "plan", "agent": agent, "steps": plan})

    return {
        **state,
        "plan":            plan,
        "task_type":       task_type,
        "current_step":    0,
        "current_agent":   agent,
        "iteration_count": state["iteration_count"] + 1,
        "retry_count":     state.get("retry_count", 0) + (1 if state.get("errors") else 0),
        "errors":          [],
        "research_needed": research_needed,
        "ui_needed":       ui_needed,
        "completed":       False,
        "messages":        state["messages"] + [{"role": "planner", "content": response}],
    }


# ══════════════════════════════════════════════════════════════
#  AGENT 2 — EXECUTOR
# ══════════════════════════════════════════════════════════════
EXECUTOR_SYSTEM = """You are the Executor Agent. Execute ONE plan step per response using a tool call.

=== MANDATORY OUTPUT FORMAT ===
You MUST output ONLY a tool call block. Nothing else. No explanation, no prose, no markdown.

TOOL_CALL: <tool_name>
PARAMS: {{"key": "value"}}
END_TOOL_CALL

=== WIRE FORMAT RULES (MUST follow exactly) ===
- Line 1: TOOL_CALL: <name>   (note the space after the colon)
- Line 2: PARAMS: <valid JSON>  (all strings quoted, no trailing comma)
- Line 3: END_TOOL_CALL
- Nothing before or after these 3 lines

=== YOUR TOOLS (you may call these directly) ===
  file_ops      → manage files and folders on disk
    PARAMS: {{"action": "write", "path": "path/file.txt", "content": "text"}}
    PARAMS: {{"action": "read",  "path": "path/file.txt"}}
    PARAMS: {{"action": "mkdir", "path": "path/new_folder"}}
    PARAMS: {{"action": "list",  "path": "."}}
    PARAMS: {{"action": "delete","path": "path/file.txt"}}
    PARAMS: {{"action": "copy",  "path": "src.txt", "destination": "dst.txt"}}
    PARAMS: {{"action": "move",  "path": "src.txt", "destination": "dst.txt"}}
    PARAMS: {{"action": "find",  "path": "C:/Users", "destination": "*.txt"}}

  terminal_exec → run shell/PowerShell commands
    PARAMS: {{"command": "dir C:\\Users", "cwd": "C:\\Users"}}

  system_tool   → hardware and process info (no UI needed)
    PARAMS: {{"action": "full_info"}}   or cpu_info | ram_info | disk_info | gpu_info | processes

=== HANDOFF SIGNALS (write one of these if step needs a specialist) ===
  NEED_RESEARCH: <topic>    → hand off to the Research Agent for web search
  NEED_UI: <description>    → hand off to the UI Controller for desktop actions

=== DO NOT ===
- DO NOT call web_search or ui_control yourself — use NEED_RESEARCH / NEED_UI instead
- DO NOT open Notepad or any app to create a file — use file_ops write
- DO NOT output prose, explanations, markdown, or XML tags
- DO NOT use backticks, code blocks, or HTML
- DO NOT invent facts — only use tool outputs from prior results
- DO NOT repeat the plan or summarise — just run the current step

=== EXAMPLES ===
# Create a file:
TOOL_CALL: file_ops
PARAMS: {{"action": "write", "path": "C:/Users/Home/Desktop/success.txt", "content": "success"}}
END_TOOL_CALL

# Create a folder:
TOOL_CALL: file_ops
PARAMS: {{"action": "mkdir", "path": "my_new_folder"}}
END_TOOL_CALL

# Run a command:
TOOL_CALL: terminal_exec
PARAMS: {{"command": "ipconfig"}}
END_TOOL_CALL

# Get system info:
TOOL_CALL: system_tool
PARAMS: {{"action": "ram_info"}}
END_TOOL_CALL

# Need web info:
NEED_RESEARCH: latest bitcoin price USD

Owned tools: {tools}
""".format(tools=EXECUTOR_TOOL_NAMES)


def executor_node(state: AgentState) -> AgentState:
    agent = "Executor"

    # Check if plan is done
    if state["current_step"] >= len(state["plan"]):
        streamer.status("✅ All steps executed.", agent)
        return {**state, "completed": True, "current_agent": agent}

    step = state["plan"][state["current_step"]]
    streamer.status(f"▶ Step {state['current_step'] + 1}/{len(state['plan'])}: {step}", agent)
    executor_memory = _trim_memory_context(state["task"], state["memory_context"], max_chars=550)

    human = f"""Current plan step ({state['current_step'] + 1} of {len(state['plan'])}):
{step}

Full plan:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(state['plan']))}

Results so far:
{_results_summary(state['results'])}

Memory:
{executor_memory}

Execute the current step. Use a tool if needed."""

    # DEBUG: Log the executor prompt
    streamer.info(
        f"📋 Executor prompt:\n"
        f"  Step: {step}\n"
        f"  Results so far: {len(state['results'])} items\n"
        f"  Memory context: {len(executor_memory)} chars",
        agent
    )

    response = _stream_llm(EXECUTOR_SYSTEM, human, agent)

    new_results = list(state["results"])
    prior_errors = list(state["errors"])
    step_errors: list[str] = []
    tool_calls = _parse_tool_calls(response)
    completion_requested = "TASK_COMPLETE" in response

    # DEBUG: Log full LLM response and tool parsing
    streamer.info(f"📝 LLM Response ({len(response)} chars): {response[:200]}...", agent)
    
    if not tool_calls:
        heuristic_calls = _heuristic_tool_calls(
            state["task"], step, task_type=state.get("task_type", "")
        )
        if heuristic_calls:
            tool_calls = heuristic_calls
            streamer.info(
                f"Using heuristic tool fallback: {[tc['tool'] for tc in tool_calls]}",
                agent,
            )

    # Debug: Log tool call parsing after heuristics are applied
    if "TOOL_CALL" in response.upper() and not tool_calls:
        streamer.error(
            f"⚠️ Tool call format didn't match regex. LLM mentioned TOOL_CALL but parsing failed. "
            f"Output: {response[:300]}...",
            agent,
        )
    elif tool_calls:
        streamer.info(f"✅ Prepared {len(tool_calls)} tool call(s): {[tc['tool'] for tc in tool_calls]}", agent)
    else:
        streamer.info(f"ℹ️ No tool calls in response. Processing as status message.", agent)

    # ── If this is a UI-type task, skip executor entirely ─────────────
    # Executor should never process UI steps — the UI Controller handles them.
    if state.get("task_type") == "ui":
        streamer.status("🖥️ UI task — routing directly to UI Controller.", agent)
        return {
            **state,
            "ui_needed":       True,
            "research_needed": False,
            "current_step":    state["current_step"] + 1,  # advance past this step
            "current_agent":   agent,
            "messages":        state["messages"] + [{"role": "executor", "content": ""}],
            "results":         new_results + [f"UI action needed: {step}"],
        }

    for tc in tool_calls:
        handoff = _handoff_for_tool_call(tc)
        if not handoff:
            continue

        handoff_kind, detail = handoff
        if handoff_kind == "research":
            streamer.status(f"🔍 Handing off to Researcher: {detail}", agent)
            return {
                **state,
                "research_needed": True,
                "ui_needed": False,
                "current_step":    state["current_step"] + 1,  # advance so step isn't repeated
                "current_agent":   agent,
                "messages":        state["messages"] + [{"role": "executor", "content": response}],
                "results":         new_results + [f"Research needed: {detail}"],
            }

        streamer.status(f"🖥️ Handing off to UI Controller: {detail}", agent)
        return {
            **state,
            "research_needed": False,
            "ui_needed":       True,
            "current_step":    state["current_step"] + 1,  # advance so step isn't repeated
            "current_agent":   agent,
            "messages":        state["messages"] + [{"role": "executor", "content": response}],
            "results":         new_results + [f"UI action needed: {detail}"],
        }

    if "NEED_RESEARCH:" in response:
        query = response.split("NEED_RESEARCH:")[-1].split("\n")[0].strip()
        streamer.status(f"🔍 Handing off to Researcher: {query}", agent)
        return {
            **state,
            "research_needed": True,
            "current_agent":   agent,
            "messages":        state["messages"] + [{"role": "executor", "content": response}],
            "results":         new_results + [f"Research needed: {query}"],
        }

    if "NEED_UI:" in response:
        desc = response.split("NEED_UI:")[-1].split("\n")[0].strip()
        streamer.status(f"🖥️ Handing off to UI Controller: {desc}", agent)
        return {
            **state,
            "ui_needed":    True,
            "current_agent": agent,
            "messages":     state["messages"] + [{"role": "executor", "content": response}],
            "results":      new_results + [f"UI action needed: {desc}"],
        }

    # Execute tool calls
    if tool_calls:
        streamer.info(f"🔧 Executing {len(tool_calls)} tool(s)...", agent)
        for tc in tool_calls:
            streamer.tool_call(tc["tool"], tc["params"], agent)
            result = dispatch_tool(tc["tool"], tc["params"])
            streamer.info(f"  → Tool result: {result[:150]}...", agent)
            if _is_tool_failure(result):
                step_errors.append(f"Step {state['current_step'] + 1}: {result}")
                streamer.error(f"  ❌ Tool failed: {result[:100]}", agent)
            else:
                new_results.append(f"Step {state['current_step'] + 1} ({tc['tool']}): {result[:500]}")
                streamer.info(f"  ✅ Tool succeeded", agent)
    else:
        if response.strip():
            streamer.info(f"📄 No tools called. Recording response as result.", agent)
            new_results.append(f"Step {state['current_step'] + 1}: {response[:500]}")
        else:
            step_errors.append(
                f"Step {state['current_step'] + 1}: Executor produced no actionable response."
            )
            streamer.error("  ❌ Executor returned an empty response.", agent)

    is_last_planned_step = state["current_step"] >= len(state["plan"]) - 1
    all_errors = prior_errors + step_errors
    completed = is_last_planned_step and not step_errors
    if completion_requested:
        completed = is_last_planned_step and not step_errors
    if completion_requested and not is_last_planned_step:
        new_results.append(
            f"Executor requested early completion before finishing the plan at step {state['current_step'] + 1}."
        )
    if completed:
        streamer.status("🎉 Task marked complete by Executor.", agent)

    return {
        **state,
        "current_step":    state["current_step"] + 1,
        "results":         new_results,
        "errors":          all_errors,
        "research_needed": False,
        "ui_needed":       False,
        "completed":       completed,
        "current_agent":   agent,
        "iteration_count": state["iteration_count"] + 1,
        "messages":        state["messages"] + [{"role": "executor", "content": response}],
    }


# ══════════════════════════════════════════════════════════════
#  AGENT 3 — RESEARCHER
# ══════════════════════════════════════════════════════════════
RESEARCHER_SYSTEM = """You are the Research Agent. Search the web and return a concise, factual summary.

=== MANDATORY FORMAT ===
Step 1 — emit ONE web search tool call:
TOOL_CALL: web_search
PARAMS: {"query": "<specific query>", "max_results": 5}
END_TOOL_CALL

Step 2 — after results are returned, write a short answer-first summary (under 200 words).

=== RULES ===
- ALWAYS emit exactly one TOOL_CALL block first, then the summary
- Use the most reputable, up-to-date sources
- Include exact values, currency, dates, source names when available
- If sources conflict or are stale, say so — do NOT guess
- For stock/crypto prices: include price, currency, timestamp, and source
- Keep the summary under 200 words

=== DO NOT ===
- DO NOT answer from memory without searching
- DO NOT skip the TOOL_CALL block
- DO NOT output XML tags, markdown headers, or prose before the tool call
- DO NOT add disclaimers, caveats, or role-play commentary
"""


def researcher_node(state: AgentState) -> AgentState:
    agent = "Researcher"

    # What should we research? Check results first for specific delegated topic,
    # then fall back to the full task (happens when Planner routes here directly).
    topic = state.get("task", "the current task")
    for r in reversed(state["results"]):
        if "Research needed:" in r:
            topic = r.replace("Research needed:", "").strip()
            break

    streamer.status(f"🔍 Researching: {topic}", agent)
    researcher_memory = _trim_memory_context(state["task"], state["memory_context"], max_chars=450)

    human = f"""Task: {state['task']}

Research topic: {topic}

Memory context:
{researcher_memory}

Search the web and provide a concise summary."""

    response = _stream_llm(RESEARCHER_SYSTEM, human, agent)

    tool_calls = _parse_tool_calls(response)
    new_results = list(state["results"])
    new_errors  = list(state["errors"])

    # ── Mandatory search guarantee ────────────────────────────────────────
    # Small LLMs often answer from memory without emitting a TOOL_CALL block.
    # The researcher MUST hit the web — force a web_search if none was parsed.
    if not tool_calls:
        if "TOOL_CALL" in response.upper():
            streamer.error(
                f"⚠️ TOOL_CALL present but unparseable in Researcher. "
                f"Output: {response[:200]}...",
                agent,
            )
        streamer.info(
            f"🔄 No TOOL_CALL found — forcing web_search for: '{topic[:80]}'",
            agent,
        )
        tool_calls = [{"tool": "web_search", "params": {"query": topic, "max_results": 5}}]

    # ── Execute tool calls (always ≥ 1 after the guarantee above) ────────
    for tc in tool_calls:
        streamer.tool_call(tc["tool"], tc["params"], agent)
        raw = dispatch_tool(tc["tool"], tc["params"])
        if _is_tool_failure(raw):
            new_errors.append(f"Research error: {raw}")
            # If the forced search also failed, at least record the LLM's answer
            if response.strip() and not new_results:
                new_results.append(f"[Research — search failed] {topic}: {response[:400]}")
            continue
        summary_prompt = (
            f"Topic: {topic}\n\n"
            f"Raw search results:\n{raw}\n\n"
            "Write an answer-first summary using only the evidence above. "
            "Include exact figures, currency, dates or times, and source names when available. "
            "If the evidence is incomplete or conflicting, say that briefly and do not guess."
        )
        summary = _stream_llm(RESEARCHER_SYSTEM, summary_prompt, agent)
        new_results.append(f"[Research] {topic}: {summary[:600]}")

    streamer.status("✅ Research complete.", agent)

    # If the researcher ran in place of the executor for a research-type plan step,
    # advance current_step so the executor doesn't re-run the same step.
    next_step = state["current_step"]
    plan = state.get("plan", [])
    if next_step < len(plan):
        step_l = plan[next_step].lower()
        if any(
            kw in step_l
            for kw in ("search", "research", "look up", "find", "web_search", "fetch")
        ):
            next_step += 1

    return {
        **state,
        "results":         new_results,
        "errors":          new_errors,
        "research_needed": False,
        "current_step":    next_step,
        "current_agent":   agent,
        "iteration_count": state["iteration_count"] + 1,
        "messages":        state["messages"] + [{"role": "researcher", "content": response}],
    }


# ══════════════════════════════════════════════════════════════
#  AGENT 4 — UI CONTROLLER
# ══════════════════════════════════════════════════════════════
UI_SYSTEM = """You are a Windows desktop UI automation agent.
Your ONLY output must be exactly ONE TOOL_CALL block. No prose, no explanation, no refusals.

=== MANDATORY FORMAT ===
TOOL_CALL: ui_control
PARAMS: {"action": "<action>", "target": "<value>", "text": "<value>", "keys": "<value>"}
END_TOOL_CALL

=== VALID ACTIONS ===
  open_app           → launch app by exe or URI  (target = "notepad.exe" / "whatsapp:" / "msedge.exe")
  open_whatsapp_send → open WhatsApp, find contact, send message  (target=contact, text=message)
  list_windows       → list all open window titles
  find_window        → find a window by partial title  (target=partial title)
  focus_window       → bring window to front  (target=partial title)
  get_control_tree   → dump UI tree of a window  (target=partial title)
  click_control      → click a button/control by label  (target=label, text=window title)
  type_into_control  → type text into a control  (target=window title, text=text to type)
  find_and_click     → find window and click a control  (target=window, text=button)
  hotkey             → send keyboard shortcut  (keys="ctrl,s" / "alt,f4")
  press_key          → press a single key  (keys="enter" / "tab" / "escape")
  clipboard_set      → copy text to clipboard  (text=text)

=== PARAM RULES (STRICT) ===
- Valid JSON keys ONLY: action, target, text, keys
- NEVER invent keys like app_name, name, window_title, application, program
- Omit keys you are not using (don't include empty strings)
- For open_app: put the exe/URI in "target" ONLY

=== DO NOT ===
- DO NOT output prose, markdown, HTML, or XML tags
- DO NOT output more than one TOOL_CALL
- DO NOT use invented/hallucinated key names
- DO NOT refuse — always output a tool call

=== EXAMPLES ===
# Launch Chrome:
TOOL_CALL: ui_control
PARAMS: {"action": "open_app", "target": "chrome.exe"}
END_TOOL_CALL

# Send WhatsApp message:
TOOL_CALL: ui_control
PARAMS: {"action": "open_whatsapp_send", "target": "John", "text": "Hello!"}
END_TOOL_CALL

# List open windows:
TOOL_CALL: ui_control
PARAMS: {"action": "list_windows"}
END_TOOL_CALL

# Press a hotkey:
TOOL_CALL: ui_control
PARAMS: {"action": "hotkey", "keys": "ctrl,s"}
END_TOOL_CALL
"""


def ui_controller_node(state: AgentState) -> AgentState:
    agent = "UI Controller"

    # Find what UI action is needed
    ui_task = state.get("task", "perform the required UI action")
    for r in reversed(state["results"]):
        if "UI action needed:" in r:
            ui_task = r.replace("UI action needed:", "").strip()
            break

    streamer.status(f"🖥️ UI action: {ui_task}", agent)
    ui_memory = _trim_memory_context(state["task"], state["memory_context"], max_chars=350)

    # Pre-analyze the task so we can give the small model a very explicit hint
    combined_hint = " ".join(p for p in (ui_task, state["task"]) if p).lower()

    # Build a suggested TOOL_CALL the model can just copy if it agrees
    def _build_suggestion() -> str:
        if "whatsapp" in combined_hint:
            contact = _extract_contact_name(ui_task + " " + state["task"])
            msg     = _extract_message_text(ui_task + " " + state["task"])
            if contact or msg:
                return (
                    'TOOL_CALL: ui_control\n'
                    f'PARAMS: {{"action": "open_whatsapp_send", "target": "{contact}", "text": "{msg}"}}\n'
                    'END_TOOL_CALL'
                )
            return (
                'TOOL_CALL: ui_control\n'
                'PARAMS: {"action": "open_app", "target": "whatsapp:"}\n'
                'END_TOOL_CALL'
            )
        open_match = re.search(r"open\s+([\w.]+)", combined_hint)
        if open_match:
            app_raw = open_match.group(1).strip()
            _app_map = {
                "notepad":    "notepad.exe", "calculator": "calc.exe",
                "paint":      "mspaint.exe", "chrome":     "chrome.exe",
                "edge":       "msedge.exe",  "settings":   "ms-settings:",
                "whatsapp":   "whatsapp:",   "teams":      "msteams:",
                "slack":      "slack:",       "spotify":    "spotify:",
                "word":       "winword.exe", "excel":      "excel.exe",
            }
            tgt = _app_map.get(app_raw.lower(), app_raw)
            return (
                'TOOL_CALL: ui_control\n'
                f'PARAMS: {{"action": "open_app", "target": "{tgt}"}}\n'
                'END_TOOL_CALL'
            )
        return (
            'TOOL_CALL: ui_control\n'
            'PARAMS: {"action": "list_windows"}\n'
            'END_TOOL_CALL'
        )

    suggestion = _build_suggestion()

    # ── Deterministic bypass for high-confidence WhatsApp tasks ───────────
    # Small models (Qwen, Phi, etc.) regularly reject the correct suggestion
    # and produce a degraded call. When we've already extracted the contact
    # and message, skip the LLM entirely.
    _deterministic_call: dict | None = None
    if "whatsapp" in combined_hint:
        _wa_contact = _extract_contact_name(ui_task + " " + state["task"])
        _wa_msg     = _extract_message_text(ui_task + " " + state["task"])
        if _wa_contact or _wa_msg:
            _deterministic_call = {
                "tool":   "ui_control",
                "params": {
                    "action": "open_whatsapp_send",
                    "target": _wa_contact,
                    "text":   _wa_msg,
                },
            }
            streamer.info(
                f"🤖 Bypassing LLM — deterministic WhatsApp call: "
                f"contact='{_wa_contact}' msg='{_wa_msg}'",
                agent,
            )

    if _deterministic_call:
        response   = suggestion          # show the suggestion as the "response"
        tool_calls = [_deterministic_call]
    else:
        human = f"""Task: {state['task']}
UI action needed: {ui_task}

Results so far:
{_results_summary(state['results'])}

Emit ONE TOOL_CALL block to perform this action.
If the suggestion below is correct, copy it exactly. If not, write the corrected version.

Suggested TOOL_CALL:
{suggestion}"""

        response = _stream_llm(UI_SYSTEM, human, agent)
        tool_calls = _parse_tool_calls(response)

    new_results = list(state["results"])
    new_errors  = list(state["errors"])

    # Debug: Log tool call parsing
    if "TOOL_CALL" in response.upper() and not tool_calls:
        streamer.error(
            f"⚠️ Tool call format didn't match in UI Controller. Output: {response[:300]}...",
            agent,
        )

    # ── Mandatory UI guarantee ─────────────────────────────────────────
    # The UI Controller MUST use UI tools. If none parsed, force a screenshot
    # to maintain momentum.
    if not tool_calls:
        tool_calls = _ui_fallback_tool_calls(state["task"], ui_task)
        streamer.info(
            f"🔄 No TOOL_CALL found — using UI fallback: {tool_calls[0]['params'].get('action', 'list_windows')}.",
            agent,
        )

    # ── Execute tool calls ────────────────────────────────────────────
    for tc in tool_calls:
        streamer.tool_call(tc["tool"], tc["params"], agent)
        result = dispatch_tool(tc["tool"], tc["params"])
        if _is_tool_failure(result):
            new_errors.append(f"UI error: {result}")
            if response.strip() and tc["params"].get("action") == "screenshot" and not new_results:
                 new_results.append(f"[UI Controller] {response[:300]}")
        else:
            new_results.append(f"[UI] {tc['params'].get('action','')}: {result[:200]}")

    streamer.status("✅ UI actions complete.", agent)

    # Always advance current_step — UI controller ran FOR this step.
    next_step = state["current_step"]
    plan = state.get("plan", [])
    if next_step < len(plan):
        next_step += 1  # consumed this step

    # Mark completed when all plan steps are done → stops the loop
    is_done = next_step >= len(plan)
    if is_done:
        streamer.status("🏁 UI task complete — all steps done.", agent)

    return {
        **state,
        "results":         new_results,
        "errors":          new_errors,
        "ui_needed":       False,
        "completed":       is_done,
        "current_step":    next_step,
        "current_agent":   agent,
        "iteration_count": state["iteration_count"] + 1,
        "messages":        state["messages"] + [{"role": "ui_controller", "content": response}],
    }
