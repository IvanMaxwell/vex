"""
runtime.py — Streamer (sync → async bridge) and Permission Manager.

Both use asyncio.run_coroutine_threadsafe so LangGraph nodes (sync threads)
can push messages and permission requests to the async WebSocket loop.
"""
import asyncio
import threading
import uuid
from config import PERMISSION_TIMEOUT


_runtime_local = threading.local()


# ══════════════════════════════════════════════════════════════
#  STREAMER  – agents call these from sync threads
# ══════════════════════════════════════════════════════════════
class Streamer:
    def __init__(self):
        self._queue: asyncio.Queue | None = None
        self._loop:  asyncio.AbstractEventLoop | None = None

    def setup(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self._loop  = loop
        self._queue = queue

    def send(self, payload: dict):
        if self._loop and self._queue:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(payload), self._loop
            )

    # Convenience helpers -------------------------------------------------
    def thought(self, text: str, agent: str):
        """Agent's raw thinking / LLM token stream."""
        self.send({"type": "thought", "agent": agent, "content": text})

    def status(self, text: str, agent: str):
        """Short status banner (e.g. "✅ Plan created")."""
        self.send({"type": "status", "agent": agent, "content": text})

    def tool_call(self, tool: str, params: dict, agent: str):
        """Announce a tool is about to run."""
        self.send({"type": "tool_call", "agent": agent,
                   "content": f"🔧 {tool}({params})"})

    def tool_result(self, result: str, agent: str):
        """Show tool output."""
        self.send({"type": "tool_result", "agent": agent, "content": result})

    def error(self, text: str, agent: str):
        self.send({"type": "error", "agent": agent, "content": text})

    def info(self, text: str, agent: str = "System"):
        self.send({"type": "info", "agent": agent, "content": text})


class StreamerProxy:
    """Route streamer calls to the runtime bound to the current worker thread."""

    def __getattr__(self, name):
        return getattr(get_streamer(), name)


# ══════════════════════════════════════════════════════════════
#  PERMISSION MANAGER
# ══════════════════════════════════════════════════════════════
class PermissionManager:
    def __init__(self):
        self._loop:   asyncio.AbstractEventLoop | None = None
        self._ws = None                         # WebSocket instance
        self._events:  dict[str, threading.Event] = {}
        self._results: dict[str, bool]            = {}

    def setup(self, loop, ws, events: dict, results: dict):
        self._loop    = loop
        self._ws      = ws
        self._events  = events
        self._results = results

    # Called from sync agent thread – BLOCKS until user responds or timeout
    def request(self, tool: str, description: str, params: dict) -> bool:
        perm_id = str(uuid.uuid4())[:8]
        event   = threading.Event()
        self._events[perm_id] = event

        payload = {
            "type":        "permission_request",
            "id":          perm_id,
            "tool":        tool,
            "description": description,
            "params":      str(params)[:300],
        }

        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._ws.send_json(payload), self._loop
            )
        else:
            print(f"[PermissionManager] WARNING: No loop or WebSocket to send permission request for {perm_id}")

        granted = event.wait(timeout=PERMISSION_TIMEOUT)

        result = self._results.pop(perm_id, False) if granted else False
        
        if not granted:
            # Log timeout for debugging
            import sys
            timeout_msg = f"⏱️ Permission TIMEOUT ({PERMISSION_TIMEOUT}s) for tool '{tool}': {description[:100]}"
            print(f"[PermissionManager] {timeout_msg}", file=sys.stderr)
        
        self._events.pop(perm_id, None)
        return result

    def grant(self, perm_id: str):
        self._results[perm_id] = True
        if perm_id in self._events:
            self._events[perm_id].set()

    def deny(self, perm_id: str):
        self._results[perm_id] = False
        if perm_id in self._events:
            self._events[perm_id].set()


class PermissionManagerProxy:
    """Route permission calls to the runtime bound to the current worker thread."""

    def __getattr__(self, name):
        return getattr(get_permission_manager(), name)


def bind_runtime(streamer_instance: Streamer, permission_manager_instance: PermissionManager):
    _runtime_local.streamer = streamer_instance
    _runtime_local.permission_manager = permission_manager_instance


def clear_runtime():
    for attr in ("streamer", "permission_manager"):
        if hasattr(_runtime_local, attr):
            delattr(_runtime_local, attr)


def get_streamer() -> Streamer:
    return getattr(_runtime_local, "streamer", _default_streamer)


def get_permission_manager() -> PermissionManager:
    return getattr(_runtime_local, "permission_manager", _default_permission_manager)


# ══════════════════════════════════════════════════════════════
#  SINGLETONS
# ══════════════════════════════════════════════════════════════
_default_streamer = Streamer()
_default_permission_manager = PermissionManager()

streamer = StreamerProxy()
permission_manager = PermissionManagerProxy()
