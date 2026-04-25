"""
main.py — FastAPI application with WebSocket streaming and permission system.

Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""
import asyncio
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import uuid
import time
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import Request
import system_metrics
import queue

# Start threshold watcher
threading.Thread(target=system_metrics._threshold_watcher, daemon=True).start()

_SESSIONS = []

from config import MODEL_NAME, OLLAMA_NUM_CTX, TEMPLATES_DIR, WORKSPACE_DIR
from memory_system import memory_manager
from runtime import Streamer, PermissionManager, bind_runtime, clear_runtime

app = FastAPI(title="Multi-Agent System", version="1.0.0")

# LangGraph runs synchronously in background threads.
_executor = ThreadPoolExecutor(max_workers=2)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())



@app.get("/session")
async def get_sessions():
    return JSONResponse(_SESSIONS)

@app.post("/session/new")
async def new_session(request: Request):
    body = await request.json()
    sess_id = str(uuid.uuid4())
    sess = {
        "id": sess_id,
        "created_at": int(time.time() * 1000),
        "agent_mode": body.get("agent_mode", "pipeline")
    }
    _SESSIONS.append(sess)
    return JSONResponse(sess)

@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "service": "multi-agent-system"})

@app.get("/metrics")
async def get_metrics():
    return JSONResponse(system_metrics.get_metrics())

@app.get("/graph")
async def get_graph():
    try:
        return JSONResponse(system_metrics.get_graph())
    except Exception as exc:
        return JSONResponse(
            {"nodes": [], "edges": [], "ts": time.time(), "error": f"graph_unavailable: {exc}"},
            status_code=200,
        )

@app.get("/monitor/sample")
async def monitor_sample(proc: str, metrics: str = "cpu,mem"):
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()][:2]
    return JSONResponse(system_metrics.get_process_sample(proc, metric_list))

@app.get("/notify/pending")
async def notify_pending():
    items = []
    while not system_metrics._notif_queue.empty():
        try: items.append(system_metrics._notif_queue.get_nowait())
        except queue.Empty: break
    return JSONResponse(items)

@app.post("/notify")
async def notify_post(request: Request):
    body = await request.json()
    n = system_metrics._push_notification(
        title=body.get("title", "Notification"),
        message=body.get("message", ""),
        level=body.get("level", "info"),
        duration_ms=int(body.get("duration_ms", 5000))
    )
    return JSONResponse({"ok": True, "id": n["id"]})

@app.get("/notify/stream")
async def notify_stream(request: Request):
    q = queue.Queue()
    with system_metrics._sse_lock:
        system_metrics._sse_listeners.append(q)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.to_thread(q.get, True, 1)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with system_metrics._sse_lock:
                try: system_metrics._sse_listeners.remove(q)
                except: pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws")
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str = None):

    await ws.accept()
    loop = asyncio.get_event_loop()

    out_q: asyncio.Queue = asyncio.Queue()
    stop_flag = threading.Event()
    perm_events: dict[str, threading.Event] = {}
    perm_results: dict[str, bool] = {}
    session_streamer = Streamer()
    session_permissions = PermissionManager()
    session_streamer.setup(loop, out_q)
    session_permissions.setup(loop, ws, perm_events, perm_results)

    session_streamer.send(
        {
            "type": "model_info",
            "agent": "System",
            "content": MODEL_NAME,
            "num_ctx": OLLAMA_NUM_CTX,
        }
    )

    async def _sender():
        while True:
            try:
                msg = await asyncio.wait_for(out_q.get(), timeout=0.1)
                try:
                    await ws.send_json(msg)
                except Exception:
                    break
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    sender_task = asyncio.create_task(_sender())
    graph_thread: threading.Thread | None = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "run":
                task = data.get("task", "").strip()
                if not task:
                    await ws.send_json(
                        {"type": "error", "agent": "System", "content": "Empty task."}
                    )
                    continue

                if graph_thread and graph_thread.is_alive():
                    await ws.send_json(
                        {
                            "type": "error",
                            "agent": "System",
                            "content": "A task is already running in this session.",
                        }
                    )
                    continue

                stop_flag.clear()
                perm_events.clear()
                perm_results.clear()
                session_streamer.info(f"Starting pipeline for: {task}", "System")

                def _run_pipeline():
                    bind_runtime(session_streamer, session_permissions)
                    try:
                        from graph import build_graph, create_initial_state

                        memory_ctx = memory_manager.get_full_context(task)
                        initial = create_initial_state(task, memory_ctx)
                        graph = build_graph()

                        final_state = None
                        for step_output in graph.stream(initial):
                            if stop_flag.is_set():
                                session_streamer.info("Stopped by user.", "System")
                                break
                            for _, state in step_output.items():
                                final_state = state

                        if final_state:
                            results = final_state.get("results", [])
                            plan = final_state.get("plan", [])
                            errors = final_state.get("errors", [])
                            success = final_state.get("completed", False)

                            summary_lines = ["EXECUTION SUMMARY", "=" * 50]
                            if plan:
                                summary_lines.append("\nPlan Executed:")
                                for i, step in enumerate(plan, 1):
                                    summary_lines.append(f"  {i}. {step}")

                            if results:
                                summary_lines.append("\nExecution Results:")
                                for result in results[-3:]:
                                    if result:
                                        summary_lines.append(f"  - {result.splitlines()[0][:100]}")

                            if errors:
                                summary_lines.append("\nIssues Encountered:")
                                for error in errors[-2:]:
                                    summary_lines.append(f"  - {error[:100]}")

                            final_answer = results[-1] if results else "Pipeline finished."
                            summary_lines.append("\n" + "=" * 50)
                            summary_lines.append("FINAL ANSWER:")
                            summary_lines.append("=" * 50)
                            summary_lines.append(final_answer)

                            session_streamer.send(
                                {
                                    "type": "final_summary",
                                    "agent": "System",
                                    "content": "\n".join(summary_lines),
                                    "success": success,
                                }
                            )
                            memory_manager.save_session(task, final_answer, success)
                    except Exception as exc:
                        tb = traceback.format_exc()
                        memory_manager.save_session(task, f"Pipeline error: {exc}", False)
                        session_streamer.error(
                            f"Pipeline error: {exc}\n{tb[:500]}",
                            "System",
                        )
                    finally:
                        clear_runtime()
                        session_streamer.send(
                            {"type": "done", "content": "Pipeline complete."}
                        )

                graph_thread = threading.Thread(target=_run_pipeline, daemon=True)
                graph_thread.start()

            elif msg_type == "permission_response":
                perm_id = data.get("id", "")
                granted = data.get("granted", False)
                if granted:
                    session_permissions.grant(perm_id)
                else:
                    session_permissions.deny(perm_id)

            elif msg_type == "stop":
                stop_flag.set()
                for pid in list(perm_events.keys()):
                    session_permissions.deny(pid)
                session_streamer.info("Stop signal sent.", "System")

            elif msg_type == "update_preference":
                key = data.get("key", "")
                value = data.get("value", "")
                if key:
                    memory_manager.update_preferences({key: value})
                    await ws.send_json(
                        {
                            "type": "info",
                            "agent": "Memory",
                            "content": f"Preference saved: {key} = {value}",
                        }
                    )

            elif msg_type == "get_memory":
                import json

                lt_data = memory_manager.long_term._data
                await ws.send_json(
                    {
                        "type": "memory_data",
                        "content": json.dumps(
                            {
                                "preferences": lt_data.get("user_preferences", {}),
                                "learned_facts": lt_data.get("learned_facts", {}),
                                "task_count": len(lt_data.get("task_history", [])),
                                "vector_ok": memory_manager.vector.available,
                            },
                            indent=2,
                        ),
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WebSocket] Error: {exc}")
    finally:
        sender_task.cancel()
        stop_flag.set()
        for pid in list(perm_events.keys()):
            session_permissions.deny(pid)


@app.on_event("startup")
async def startup():
    print(
        "\n"
        "===================================================\n"
        "  Multi-Agent System - Ready\n"
        "  Open http://localhost:8001 in your browser\n"
        "===================================================\n"
    )
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
