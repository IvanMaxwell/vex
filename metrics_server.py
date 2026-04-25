"""
Standalone launcher for the metrics polling server used by the frontend.

Run:
    python metrics_server.py
"""

import threading
from http.server import HTTPServer

import psutil

from system_metrics import (
    GPU_TYPE,
    HAS_GPU,
    Handler,
    _push_notification,
    _threshold_watcher,
    get_metrics,
)


def main():
    port = 7861
    print(f"\n[PC Assist] Metrics server -> http://localhost:{port}")
    print("[PC Assist] Endpoints:")
    print("  GET  /metrics            system stats")
    print("  GET  /graph              process graph (top 20)")
    print("  GET  /monitor/sample     single-process poll  ?proc=X&metrics=cpu,mem")
    print("  GET  /notify/stream      SSE notification stream")
    print("  GET  /notify/pending     poll pending notifications")
    print("  POST /notify             push notification from your backend")
    print(f"[PC Assist] GPU: {'YES (' + GPU_TYPE + ')' if HAS_GPU else 'NOT DETECTED'}")
    print("[PC Assist] Press Ctrl+C to stop\n")

    psutil.cpu_percent(interval=0.5)
    get_metrics()

    threading.Thread(target=_threshold_watcher, daemon=True).start()
    _push_notification("PC Assist started", f"Monitoring server live on port {port}", "ok", 4000)

    HTTPServer(("localhost", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
