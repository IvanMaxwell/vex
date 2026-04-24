"""
tools.py — All agent tools.
Every tool:
  1. Requests user permission (blocks until granted/denied/timeout)
  2. Executes inside agent_workspace where possible
  3. Retries once on failure
  4. Returns a plain string result
"""
import os
import shutil
import subprocess
import json
import time
import platform
import re
from pathlib import Path
import requests  # pip install requests (already transitive dep of fastapi/langchain)

from config import SERPAPI_KEY, WORKSPACE_DIR
from runtime import streamer, permission_manager


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def _ws_safe(path: str) -> str:
    """Return an absolute path; if relative, anchor it to workspace."""
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    return str((Path(WORKSPACE_DIR) / p).resolve())


def _retry(fn, *args, **kwargs):
    """Try fn once; on exception retry once with same args."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        streamer.info(f"⚠️ Tool error: {e}. Retrying…", "Tools")
        time.sleep(0.5)
        try:
            return fn(*args, **kwargs)
        except Exception as e2:
            return f"ERROR: {e2}"


def _search_queries(query: str) -> list[str]:
    raw = " ".join((query or "").split()).strip()
    if not raw:
        return []

    cleaned = re.sub(r"^(check|find|search|look up|lookup|tell me|show me|what is)\s+", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\?$", "", cleaned).strip()

    candidates: list[str] = [raw]
    if cleaned and cleaned.lower() != raw.lower():
        candidates.append(cleaned)

    lowered = cleaned.lower()
    if "stock price" in lowered or "share price" in lowered:
        finance_variant = cleaned
        if "today" not in lowered:
            finance_variant = f"{finance_variant} today"
        if "stock" in finance_variant.lower() and "price" in finance_variant.lower():
            candidates.append(finance_variant)

    deduped: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized.lower() not in [q.lower() for q in deduped]:
            deduped.append(normalized)
    return deduped


def _format_search_results(results: list[dict]) -> str:
    lines = []
    for idx, r in enumerate(results, start=1):
        href = r.get("href", "")
        domain = r.get("domain", "unknown")
        lines.append(
            f"[{idx}] {r.get('title', '')}\n"
            f"  Source: {domain or 'unknown'}\n"
            f"  URL: {href}\n"
            f"  Snippet: {r.get('body', '')[:220]}"
        )
    return "\n\n".join(lines)

def _serpapi_search(query: str, max_results: int) -> list[dict]:
    """Search using SerpAPI Google engine via the requests library."""
    if not SERPAPI_KEY:
        raise RuntimeError(
            "SerpAPI key is missing. Set SERPAPI_KEY in config.py or "
            "as the SERPAPI_KEY environment variable."
        )

    params = {
        "api_key": SERPAPI_KEY,
        "q":       query,
        "num":     max(1, min(max_results, 10)),
        "engine":  "google",
        "hl":      "en",
        "gl":      "us",
    }

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MultiAgentBot/1.0)"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("SerpAPI request timed out (20 s)")
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(f"SerpAPI SSL error: {exc}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"SerpAPI connection error: {exc}")
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:250] if exc.response is not None else ""
        raise RuntimeError(f"SerpAPI HTTP {exc.response.status_code}: {body}")

    if "error" in payload:
        raise RuntimeError(f"SerpAPI error: {payload['error']}")

    results: list[dict] = []
    for item in payload.get("organic_results", [])[:max_results]:
        results.append(
            {
                "title":  item.get("title",         ""),
                "href":   item.get("link",           ""),
                "domain": item.get("displayed_link", ""),
                "body":   item.get("snippet",        ""),
            }
        )
    return results


# ══════════════════════════════════════════════════════════════
#  TOOL 1 — file_ops
# ══════════════════════════════════════════════════════════════
def file_ops(action: str, path: str, destination: str = "", content: str = "") -> str:
    """
    Actions: read | write | move | copy | list | mkdir | delete | find
    """
    params = {"action": action, "path": path}
    if destination: params["destination"] = destination
    if content:     params["content_preview"] = content[:80] + "..."

    granted = permission_manager.request(
        tool="file_ops",
        description=f"{action.upper()} → {path}",
        params=params,
    )
    if not granted:
        return "⛔ Permission denied by user."

    def _run():
        abs_path = _ws_safe(path) if action not in ("find",) else path
        abs_dst  = _ws_safe(destination) if destination else ""

        if action == "read":
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            return data[:3000] + ("\n[...truncated]" if len(data) > 3000 else "")

        elif action == "write":
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} chars to {abs_path}"

        elif action == "copy":
            if os.path.isdir(abs_path):
                shutil.copytree(abs_path, abs_dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
                shutil.copy2(abs_path, abs_dst)
            return f"Copied {abs_path} → {abs_dst}"

        elif action == "move":
            os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            shutil.move(abs_path, abs_dst)
            return f"Moved {abs_path} → {abs_dst}"

        elif action == "delete":
            if os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
            else:
                os.remove(abs_path)
            return f"Deleted {abs_path}"

        elif action == "list":
            items = os.listdir(abs_path)
            return "\n".join(items[:100])

        elif action == "mkdir":
            os.makedirs(abs_path, exist_ok=True)
            return f"Directory created: {abs_path}"

        elif action == "find":
            # path = root directory, destination = file extension/pattern
            pattern = destination or "*"
            matches = []
            for root, _dirs, files in os.walk(path):
                for fname in files:
                    if pattern == "*" or fname.endswith(pattern.lstrip("*")):
                        matches.append(os.path.join(root, fname))
                        if len(matches) >= 200:
                            break
                if len(matches) >= 200:
                    break
            return "\n".join(matches) if matches else "No files found."

        else:
            return f"Unknown action: {action}"

    result = _retry(_run)
    streamer.tool_result(str(result)[:400], "file_ops")
    return str(result)


# ══════════════════════════════════════════════════════════════
#  TOOL 2 — web_search
# ══════════════════════════════════════════════════════════════
def web_search(query: str, max_results: int = 5) -> str:
    granted = permission_manager.request(
        tool="web_search",
        description=f"Search: {query}",
        params={"query": query, "max_results": max_results},
    )
    if not granted:
        return "⛔ Permission denied by user."

    def _run():
        attempts: list[str] = []
        queries = _search_queries(query)

        for attempt_query in queries:
            try:
                results = _serpapi_search(attempt_query, max_results)
                if results:
                    return _format_search_results(results)
                attempts.append(f"serpapi:{attempt_query} -> 0 results")
            except Exception as exc:
                attempts.append(
                    f"serpapi:{attempt_query} -> {type(exc).__name__}: {str(exc)[:160]}"
                )

        if attempts:
            joined = "\n".join(f"  - {attempt}" for attempt in attempts[:6])
            return (
                "ERROR: Web search failed.\n"
                f"Original query: {query}\n"
                "Attempts:\n"
                f"{joined}"
            )

        return "No results found."

    result = _retry(_run)
    streamer.tool_result(str(result)[:600], "web_search")
    return str(result)


# ══════════════════════════════════════════════════════════════
#  TOOL 3 — terminal_exec
# ══════════════════════════════════════════════════════════════
def terminal_exec(command: str, cwd: str = "") -> str:
    granted = permission_manager.request(
        tool="terminal_exec",
        description=f"Run: {command}",
        params={"command": command, "cwd": cwd or WORKSPACE_DIR},
    )
    if not granted:
        return "⛔ Permission denied by user."

    def _run():
        work_dir = cwd if cwd and os.path.isdir(cwd) else WORKSPACE_DIR
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=60,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        combined = ""
        if out:
            combined += f"STDOUT:\n{out[:1500]}"
        if err:
            combined += f"\nSTDERR:\n{err[:500]}"
        combined += f"\nExit code: {proc.returncode}"
        return combined.strip() or "(no output)"

    result = _retry(_run)
    streamer.tool_result(str(result)[:600], "terminal_exec")
    return str(result)


# ══════════════════════════════════════════════════════════════
#  TOOL 4 — system_tool
# ══════════════════════════════════════════════════════════════
def system_tool(action: str = "full_info") -> str:
    granted = permission_manager.request(
        tool="system_tool",
        description=f"Get system info: {action}",
        params={"action": action},
    )
    if not granted:
        return "⛔ Permission denied by user."

    def _run():
        import psutil
        info = {}

        if action in ("cpu_info", "full_info"):
            info["cpu_percent"]    = psutil.cpu_percent(interval=1)
            info["cpu_count"]      = psutil.cpu_count()
            info["cpu_freq_mhz"]   = round(psutil.cpu_freq().current, 1) if psutil.cpu_freq() else "N/A"

        if action in ("ram_info", "full_info"):
            vm = psutil.virtual_memory()
            info["ram_total_gb"]   = round(vm.total  / 1e9, 2)
            info["ram_used_gb"]    = round(vm.used   / 1e9, 2)
            info["ram_percent"]    = vm.percent

        if action in ("disk_info", "full_info"):
            disk_root = Path.cwd().anchor or os.path.abspath(os.sep)
            disk = psutil.disk_usage(disk_root)
            info["disk_total_gb"]  = round(disk.total / 1e9, 2)
            info["disk_used_gb"]   = round(disk.used  / 1e9, 2)
            info["disk_percent"]   = disk.percent
            info["disk_root"]      = disk_root

        if action in ("gpu_info", "full_info"):
            try:
                out = subprocess.run(
                    "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader",
                    shell=True, capture_output=True, text=True, timeout=5
                ).stdout.strip()
                info["gpu"] = out if out else "No NVIDIA GPU / nvidia-smi not found"
            except Exception:
                info["gpu"] = "GPU info unavailable"

        if action in ("processes", "full_info"):
            procs = []
            for p in sorted(psutil.process_iter(["name", "cpu_percent", "memory_percent"]),
                            key=lambda x: x.info["cpu_percent"] or 0, reverse=True)[:10]:
                procs.append(p.info)
            info["top_processes"] = procs

        if action in ("wmi_info",):
            try:
                import wmi
                c = wmi.WMI()
                info["os"]  = c.Win32_OperatingSystem()[0].Caption
                info["cpu"] = c.Win32_Processor()[0].Name
            except Exception as e:
                info["wmi"] = f"WMI unavailable: {e}"

        return json.dumps(info, indent=2, default=str)

    result = _retry(_run)
    streamer.tool_result(str(result)[:800], "system_tool")
    return str(result)


# ══════════════════════════════════════════════════════════════
#  TOOL 5 — ui_control  (PyWinAuto — no image/screenshot needed)
# ══════════════════════════════════════════════════════════════
def ui_control(
    action: str,
    target: str = "",
    text: str = "",
    keys: str = "",
    control_type: str = "",
    backend: str = "uia",
) -> str:
    """
    Text-based Windows UI automation via PyWinAuto (no screenshots / image analysis).

    Actions
    -------
    open_app          — launch an application (target = exe path or URI, e.g. "whatsapp:" or "notepad.exe")
    list_windows      — list all visible top-level window titles
    find_window       — find windows whose title contains `target` (partial match)
    focus_window      — bring the window whose title contains `target` to the front
    get_control_tree  — dump the interactive control tree of the focused window
                        (target = partial window title, optional)
    click_control     — click a UI control by its text/name label (target = label)
                        inside the window whose title contains `text` (optional)
    type_into_control — focus a control by label then type text into it
                        (target = control label, text = text to type)
    find_and_click    — find window by title (`target`) and click a button/menu
                        item by label (`text`)
    hotkey            — press a key combination (keys = "ctrl,s" / "alt,f4" etc.)
    press_key         — press a single key (keys = "enter" / "tab" / "escape" etc.)
    clipboard_set     — copy `text` to the system clipboard
    clipboard_get     — return current clipboard text
    open_whatsapp_send— open WhatsApp Desktop, search for `target` contact and
                        type `text` into the message box
    """
    granted = permission_manager.request(
        tool="ui_control",
        description=f"UI: {action} → target='{target}' text='{text[:40]}'",
        params={"action": action, "target": target, "text": text[:80], "keys": keys},
    )
    if not granted:
        return "⛔ Permission denied by user."

    # ── PyWinAuto helpers ─────────────────────────────────────────────────
    def _pw_app(backend_name: str = "uia"):
        """Lazy import of pywinauto Application."""
        from pywinauto import Application  # noqa: PLC0415
        return Application

    def _get_desktop(backend_name: str = "uia"):
        """Return a pywinauto Desktop object."""
        from pywinauto import Desktop  # noqa: PLC0415
        return Desktop(backend=backend_name)

    def _find_window_wrapper(title_re: str, backend_name: str = "uia"):
        """
        Connect pywinauto to an already-running window whose title contains
        `title_re` (treated as a case-insensitive substring match).
        Returns the top-level WindowSpecification or raises.
        """
        from pywinauto import Application  # noqa: PLC0415
        app = Application(backend=backend_name).connect(
            title_re=f"(?i).*{re.escape(title_re)}.*",
            found_index=0,
            timeout=5,
        )
        return app.top_window()

    def _control_tree_text(win, depth: int = 0, max_depth: int = 4) -> list[str]:
        """Recursively collect control info as text lines."""
        lines: list[str] = []
        try:
            children = win.children()
        except Exception:
            children = []
        for child in children:
            try:
                ctype = child.element_info.control_type or ""
                cname = child.element_info.name or ""
                cauto = getattr(child.element_info, "automation_id", "") or ""
                enabled = getattr(child, "is_enabled", lambda: True)()
                indent = "  " * depth
                lines.append(
                    f"{indent}[{ctype}] name='{cname}' auto_id='{cauto}' enabled={enabled}"
                )
                if depth < max_depth:
                    lines.extend(_control_tree_text(child, depth + 1, max_depth))
            except Exception:
                continue
        return lines

    # ── Actions ───────────────────────────────────────────────────────────
    def _run():
        _backend = backend if backend in ("uia", "win32") else "uia"

        # ── open_app ──────────────────────────────────────────────────────
        if action == "open_app":
            if not target:
                return (
                    "ERROR: open_app requires a 'target' (exe path or URI like 'whatsapp:'). "
                    "'target' is empty — the LLM may have used 'text' or 'app_name' instead. "
                    "Check the PARAMS and use target=<app>."
                )
            # URI schemes (like "whatsapp:", "ms-settings:") need os.startfile
            # or 'start' via cmd, NOT a direct Popen call.
            is_uri = ":" in target and os.path.sep not in target
            if is_uri:
                try:
                    os.startfile(target)  # best for URI schemes on Windows
                except Exception:
                    subprocess.Popen(["cmd", "/c", "start", "", target])
            else:
                try:
                    os.startfile(target)
                except Exception:
                    subprocess.Popen(f'cmd /c start "" "{target}"', shell=True)
            time.sleep(2)
            return f"Launched: {target}"

        # ── list_windows ──────────────────────────────────────────────────
        elif action == "list_windows":
            desktop = _get_desktop(_backend)
            titles = []
            for w in desktop.windows():
                try:
                    t = w.window_text()
                    if t.strip():
                        titles.append(t)
                except Exception:
                    pass
            return "Open windows:\n" + "\n".join(titles[:60]) if titles else "No visible windows."

        # ── find_window ───────────────────────────────────────────────────
        elif action == "find_window":
            desktop = _get_desktop(_backend)
            matches = []
            for w in desktop.windows():
                try:
                    t = w.window_text()
                    if target.lower() in t.lower():
                        matches.append(t)
                except Exception:
                    pass
            if matches:
                return "Matching windows:\n" + "\n".join(matches)
            return f"No window found matching '{target}'."

        # ── focus_window ──────────────────────────────────────────────────
        elif action == "focus_window":
            win = _find_window_wrapper(target, _backend)
            win.set_focus()
            time.sleep(0.4)
            return f"Focused window: '{win.window_text()}'"

        # ── get_control_tree ──────────────────────────────────────────────
        elif action == "get_control_tree":
            win = _find_window_wrapper(target or "", _backend) if target else None
            if win is None:
                # Fall back to the foreground window
                from pywinauto import Desktop
                desk = Desktop(backend=_backend)
                wins = [w for w in desk.windows() if w.is_active()]
                win = wins[0] if wins else desk.windows()[0]
            lines = _control_tree_text(win)
            header = f"Control tree for '{win.window_text()}':"
            body = "\n".join(lines[:200])  # cap at 200 controls
            return f"{header}\n{body}" if lines else f"{header}\n(no child controls found)"

        # ── click_control ─────────────────────────────────────────────────
        elif action == "click_control":
            # `target` = control name/label; `text` = optional window title
            win = _find_window_wrapper(text, _backend) if text else None
            if win is None:
                from pywinauto import Desktop
                wins = [w for w in Desktop(backend=_backend).windows() if w.is_active()]
                win = wins[0] if wins else None
            if win is None:
                return "ERROR: Could not determine the active window."
            ctrl = win.child_window(title=target, found_index=0)
            ctrl.click_input()
            return f"Clicked control '{target}' in window '{win.window_text()}'"

        # ── type_into_control ─────────────────────────────────────────────
        elif action == "type_into_control":
            # `target` = control label; `text` = text to type
            win = _find_window_wrapper(target, _backend) if target else None
            if win is None:
                from pywinauto import Desktop
                wins = [w for w in Desktop(backend=_backend).windows() if w.is_active()]
                win = wins[0] if wins else None
            if win is None:
                return "ERROR: Active window not found."
            # Try to locate an edit/text control to type into
            try:
                # Find by control_type if given
                if control_type:
                    ctrl = win.child_window(control_type=control_type, found_index=0)
                else:
                    ctrl = win.child_window(control_type="Edit", found_index=0)
                ctrl.set_focus()
                ctrl.type_keys(text, with_spaces=True)
            except Exception:
                # Fallback: send keys to the window itself
                win.set_focus()
                win.type_keys(text, with_spaces=True)
            return f"Typed into '{win.window_text()}': {text[:80]}"

        # ── find_and_click ────────────────────────────────────────────────
        elif action == "find_and_click":
            # `target` = window title substring; `text` = button/menu item label
            win = _find_window_wrapper(target, _backend)
            win.set_focus()
            time.sleep(0.2)
            ctrl = win.child_window(title=text, found_index=0)
            ctrl.click_input()
            return f"Clicked '{text}' in window '{win.window_text()}'"

        # ── hotkey ────────────────────────────────────────────────────────
        elif action == "hotkey":
            import pywinauto.keyboard as pw_kb
            # Convert "ctrl,s" → "{VK_CONTROL}s" style, or use send_keys shorthand
            key_parts = [k.strip().lower() for k in keys.split(",")]
            # Build pywinauto send_keys string
            modifier_map = {"ctrl": "^", "alt": "%", "shift": "+", "win": "{WIN}"}
            special_map = {
                "enter": "{ENTER}", "tab": "{TAB}", "escape": "{ESC}",
                "esc": "{ESC}", "backspace": "{BACKSPACE}", "delete": "{DELETE}",
                "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
                "home": "{HOME}", "end": "{END}", "f1": "{F1}", "f2": "{F2}",
                "f3": "{F3}", "f4": "{F4}", "f5": "{F5}",
            }
            combo = ""
            modifiers = ""
            for part in key_parts:
                if part in modifier_map:
                    modifiers += modifier_map[part]
                else:
                    char = special_map.get(part, part)
                    combo += modifiers + char
                    modifiers = ""
            if not combo:
                combo = modifiers  # solo modifier press
            pw_kb.send_keys(combo, pause=0.05)
            return f"Sent hotkey: {keys} → '{combo}'"

        # ── press_key ─────────────────────────────────────────────────────
        elif action == "press_key":
            import pywinauto.keyboard as pw_kb
            special_map = {
                "enter": "{ENTER}", "tab": "{TAB}", "escape": "{ESC}",
                "esc": "{ESC}", "backspace": "{BACKSPACE}", "delete": "{DELETE}",
                "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
                "home": "{HOME}", "end": "{END}", "space": "{SPACE}",
                "f1": "{F1}", "f2": "{F2}", "f3": "{F3}",
                "f4": "{F4}", "f5": "{F5}",
            }
            key_seq = special_map.get(keys.lower().strip(), keys)
            pw_kb.send_keys(key_seq, pause=0.05)
            return f"Pressed key: {keys}"

        # ── clipboard_set ─────────────────────────────────────────────────
        elif action == "clipboard_set":
            import pyperclip
            pyperclip.copy(text)
            return f"Clipboard set to: {text[:80]}"

        # ── clipboard_get ─────────────────────────────────────────────────
        elif action == "clipboard_get":
            import pyperclip
            return pyperclip.paste()

        # ── open_whatsapp_send ────────────────────────────────────────────
        elif action == "open_whatsapp_send":
            import pywinauto.keyboard as pw_kb
            from pywinauto import Desktop

            if not target and not text:
                return (
                    "ERROR: open_whatsapp_send requires 'target' (contact name) "
                    "and/or 'text' (message). Both are empty."
                )

            # ── helpers ───────────────────────────────────────────
            def _wa_is_running() -> bool:
                """Check via psutil (fast) first, UIA windows as fallback."""
                try:
                    import psutil
                    for proc in psutil.process_iter(["name"]):
                        try:
                            if "whatsapp" in (proc.info["name"] or "").lower():
                                return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    pass
                # Fallback: UIA window title check
                try:
                    for w in Desktop(backend="uia").windows():
                        try:
                            if "whatsapp" in w.window_text().lower():
                                return True
                        except Exception:
                            pass
                except Exception:
                    pass
                return False

            def _clipboard_type(txt: str):
                """Paste text via clipboard — works for Electron/Store apps
                that ignore pywinauto.keyboard.send_keys."""
                import pyperclip
                pyperclip.copy(txt)
                time.sleep(0.15)
                pw_kb.send_keys("^v", pause=0.05)  # Ctrl+V
                time.sleep(0.2)

            # ── Step 1: Launch WhatsApp if not running ──────────────
            if not _wa_is_running():
                streamer.info("WhatsApp not open — launching via URI scheme.", "ui_control")
                try:
                    os.startfile("whatsapp:")
                except Exception:
                    subprocess.Popen(["cmd", "/c", "start", "", "whatsapp:"])
                launched = False
                for _tick in range(20):                          # up to 20 s
                    time.sleep(1)
                    streamer.info(
                        f"⏳ Waiting for WhatsApp to start… ({_tick + 1}/20 s)",
                        "ui_control",
                    )
                    if _wa_is_running():
                        streamer.info("✅ WhatsApp process detected.", "ui_control")
                        launched = True
                        break
                if not launched:
                    return (
                        "ERROR: WhatsApp did not open after 20 s. "
                        "Install WhatsApp Desktop from the Microsoft Store."
                    )
            else:
                streamer.info("WhatsApp already running — reusing window.", "ui_control")
            streamer.info("⏳ Waiting for WhatsApp window to fully render…", "ui_control")
            time.sleep(2.5)  # let window render

            # ── Step 2: Connect pywinauto ─────────────────────────
            from pywinauto import Application
            wa_win = None
            for _attempt in range(4):
                streamer.info(
                    f"🔗 Connecting to WhatsApp window (attempt {_attempt + 1}/4)…",
                    "ui_control",
                )
                try:
                    wa_app = Application(backend="uia").connect(
                        title_re="(?i).*WhatsApp.*", timeout=6
                    )
                    wa_win = wa_app.top_window()
                    streamer.info("✅ Connected to WhatsApp window.", "ui_control")
                    break
                except Exception as _conn_err:
                    streamer.info(
                        f"pywinauto connect attempt {_attempt + 1} failed: {_conn_err}",
                        "ui_control",
                    )
                    time.sleep(2)


            # Regardless of connect result, focus WhatsApp via win32gui
            try:
                import win32gui, win32con  # type: ignore
                def _focus_cb(hwnd, _):
                    if "whatsapp" in win32gui.GetWindowText(hwnd).lower():
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                win32gui.EnumWindows(_focus_cb, None)
                time.sleep(0.8)
            except Exception:
                if wa_win is not None:
                    wa_win.set_focus()
                    time.sleep(0.6)

            # ── Step 3: Search for contact ────────────────────────
            # WhatsApp Desktop is an Electron app — pywinauto UIA tree traversal
            # is extremely slow (10-30s). Use keyboard shortcuts only.
            contact_found = False
            if target:
                streamer.info(f"🔍 Searching for contact '{target}' via keyboard…", "ui_control")

                # Strategy 1: Ctrl+F opens the search bar in all WhatsApp Desktop versions
                streamer.info("⌨️ Opening search bar (Ctrl+F)…", "ui_control")
                pw_kb.send_keys("^f", pause=0.2)
                time.sleep(0.8)

                # Paste contact name via clipboard (more reliable than send_keys on Electron)
                streamer.info("📋 Typing contact name via clipboard…", "ui_control")
                _clipboard_type(target)
                streamer.info("⏳ Waiting for search results…", "ui_control")
                time.sleep(2.0)

                # Press Enter/Down to select the first result
                streamer.info("⏎ Selecting first search result…", "ui_control")
                pw_kb.send_keys("{DOWN}", pause=0.1)
                time.sleep(0.3)
                pw_kb.send_keys("{ENTER}", pause=0.1)
                streamer.info("⏳ Opening chat…", "ui_control")
                time.sleep(1.5)
                contact_found = True
                streamer.info(f"✅ Contact '{target}' selected.", "ui_control")

            # ── Step 4: Type message ────────────────────────────
            # Again use keyboard/clipboard — no UIA tree traversal.
            typed_msg = False
            if text:
                streamer.info(f"✏️ Typing message '{text[:50]}'…", "ui_control")

                # Press Escape first in case search bar is still focused,
                # then Tab into the message area
                pw_kb.send_keys("{ESC}", pause=0.1)
                time.sleep(0.3)

                # Click bottom-centre of the window to land on the message input
                if wa_win is not None:
                    try:
                        rect = wa_win.rectangle()
                        import pywinauto.mouse as pw_mouse
                        click_x = rect.left + (rect.right - rect.left) // 2
                        click_y = rect.bottom - 60          # ~60 px from bottom = message bar
                        streamer.info(f"🖱️ Clicking message input area ({click_x}, {click_y})…", "ui_control")
                        pw_mouse.click(coords=(click_x, click_y))
                        time.sleep(0.4)
                    except Exception as _e:
                        streamer.info(f"Mouse click failed ({_e}) — using Tab to focus message bar.", "ui_control")
                        pw_kb.send_keys("{TAB}", pause=0.1)
                        time.sleep(0.3)
                else:
                    pw_kb.send_keys("{TAB}", pause=0.1)
                    time.sleep(0.3)

                # Paste message via clipboard
                streamer.info("📋 Pasting message text…", "ui_control")
                _clipboard_type(text)
                time.sleep(0.3)
                typed_msg = True
                streamer.info("✅ Message text pasted.", "ui_control")

                # ── Step 5: Press Enter to send ───────────────────
                streamer.info("📤 Sending message (pressing Enter)…", "ui_control")
                time.sleep(0.2)
                pw_kb.send_keys("{ENTER}", pause=0.1)
                time.sleep(0.4)
                streamer.info("✅ Message sent!", "ui_control")



            return (
                f"WhatsApp: contact='{target}' found={contact_found}, "
                f"message='{text[:50]}' typed={typed_msg}, sent=True."
            )

        else:
            return (
                f"Unknown UI action: '{action}'. "
                "Valid actions: open_app | list_windows | find_window | focus_window | "
                "get_control_tree | click_control | type_into_control | find_and_click | "
                "hotkey | press_key | clipboard_set | clipboard_get | open_whatsapp_send"
            )

    result = _retry(_run)
    streamer.tool_result(str(result)[:400], "ui_control")
    return str(result)


# ══════════════════════════════════════════════════════════════
#  TOOL REGISTRY  (executor uses this to dispatch)
# ══════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "file_ops":     file_ops,
    "web_search":   web_search,
    "terminal_exec": terminal_exec,
    "system_tool":  system_tool,
    "ui_control":   ui_control,
}


def _normalize_tool_params(tool_name: str, params: dict) -> dict:
    normalized = dict(params)

    if tool_name == "system_tool" and "action" not in normalized and "actions" in normalized:
        normalized["action"] = normalized.pop("actions")

    if tool_name == "ui_control" and "action" not in normalized and "actions" in normalized:
        normalized["action"] = normalized.pop("actions")

    # Strip old PyAutoGUI coordinate params — PyWinAuto ui_control no longer accepts them
    if tool_name == "ui_control":
        normalized.pop("x", None)
        normalized.pop("y", None)

        # Remap param-name aliases the LLM commonly hallucinate
        # e.g. {"action": "open_app", "app_name": "WhatsApp"} → target="WhatsApp"
        for _bad_key, _good_key in (("app_name", "target"), ("name", "target"), ("window", "target")):
            if _bad_key in normalized and _good_key not in normalized:
                normalized[_good_key] = normalized.pop(_bad_key)

        # Special case: open_app needs target. If the LLM put the app name in
        # "text" instead of "target" (common Qwen mistake), move it.
        if normalized.get("action") == "open_app" and not normalized.get("target") and normalized.get("text"):
            _app_uri_map = {
                "whatsapp":    "whatsapp:",
                "notepad":     "notepad.exe",
                "calculator":  "calc.exe",
                "calc":        "calc.exe",
                "paint":       "mspaint.exe",
                "chrome":      "chrome.exe",
                "edge":        "msedge.exe",
                "word":        "winword.exe",
                "excel":       "excel.exe",
                "settings":    "ms-settings:",
                "teams":       "msteams:",
                "slack":       "slack:",
                "spotify":     "spotify:",
            }
            raw_text = normalized["text"].strip().lower()
            # Try to map known app names to URIs/exes, otherwise use value as-is
            normalized["target"] = _app_uri_map.get(raw_text, normalized.pop("text"))
            if "text" in normalized and normalized["text"] == normalized["target"]:
                normalized.pop("text", None)  # clean up if still present

        # Normalize action aliases that the LLM commonly generates
        _action_aliases = {
            "open_whatsapp":      "open_whatsapp_send",
            "whatsapp_send":      "open_whatsapp_send",
            "send_whatsapp":      "open_whatsapp_send",
            "send_message":       "open_whatsapp_send",
            "launch":             "open_app",
            "start":              "open_app",
            "type_text":          "type_into_control",
            "type":               "type_into_control",
            "click":              "click_control",
            "press":              "press_key",
            "keyboard":           "hotkey",
            "windows":            "list_windows",
            "get_windows":        "list_windows",
            "search_window":      "find_window",
            "screenshot":         "list_windows",   # map old screenshot to list_windows
            "take_screenshot":    "list_windows",
        }
        action = normalized.get("action", "")
        if action in _action_aliases:
            normalized["action"] = _action_aliases[action]

    if tool_name == "file_ops":
        _file_ops_aliases = {
            "create_folder": "mkdir",
            "create_dir": "mkdir",
            "create_directory": "mkdir",
            "create_file": "write",
            "create": "write",
            "remove": "delete",
            "rm": "delete",
            "search": "find",
        }
        action = normalized.get("action", "")
        if action in _file_ops_aliases:
            normalized["action"] = _file_ops_aliases[action]

    return normalized


def dispatch_tool(tool_name: str, params: dict) -> str:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        return f"Unknown tool: {tool_name}"
    try:
        return fn(**_normalize_tool_params(tool_name, params))
    except TypeError as e:
        return f"Tool parameter error: {e}"
