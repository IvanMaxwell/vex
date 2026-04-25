"""
metrics_server.py  —  PC Assist backend
Run:  python metrics_server.py
Port: 7861

Endpoints
─────────────────────────────────────────────────
GET  /metrics                  System-wide stats
GET  /graph                    Top-20 process graph
GET  /monitor/sample           Single-process metric snapshot
                               ?proc=chrome.exe&metrics=cpu,mem&interval=2
GET  /notify/stream            SSE stream of notifications
GET  /notify/pending           Poll for queued notifications (clears queue)
POST /notify                   Push a notification from your backend
                               Body: { title, message, level, duration_ms }

Notification levels:  info | warn | error | ok
"""

import time, json, os, sys, subprocess, threading, queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed.  Run: pip install psutil")
    sys.exit(1)

# ── GPU detection ─────────────────────────────────────────────
HAS_GPU, GPU_TYPE = False, None
for _cmd, _t in [(["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,name","--format=csv,noheader,nounits"],"nvidia"),
                  (["rocm-smi","--showuse","--showtemp","--csv"],"amd")]:
    try:
        r = subprocess.run(_cmd, capture_output=True, text=True, timeout=2)
        if r.returncode == 0: HAS_GPU, GPU_TYPE = True, _t; break
    except Exception: pass

# ── Rolling state ─────────────────────────────────────────────
_last_net = _last_net_time = _last_disk = _last_disk_time = None

if os.name == "nt":
    _system_drive = os.environ.get("SystemDrive") or (os.path.splitdrive(sys.executable)[0] or "C:")
    DISK_ROOT = _system_drive + "\\"
else:
    DISK_ROOT = "/"

def _prime_process_cpu():
    for _proc in psutil.process_iter():
        try: _proc.cpu_percent(interval=None)
        except: pass

_prime_process_cpu()

# ── Notification queue ────────────────────────────────────────
# Each item: { id, title, message, level, duration_ms, ts }
_notif_queue   = queue.Queue()          # pending for /notify/pending poll
_sse_listeners = []                     # open SSE connections
_sse_lock      = threading.Lock()

def _push_notification(title: str, message: str, level: str = "info", duration_ms: int = 5000):
    n = {
        "id":          f"n{int(time.time()*1000)}",
        "title":       title,
        "message":     message,
        "level":       level,       # info | warn | error | ok
        "duration_ms": duration_ms,
        "ts":          time.time()
    }
    _notif_queue.put(n)
    # Fan out to all SSE listeners
    data = json.dumps(n)
    with _sse_lock:
        dead = []
        for q in _sse_listeners:
            try:   q.put(data)
            except: dead.append(q)
        for q in dead: _sse_listeners.remove(q)
    return n


# ── System metrics ────────────────────────────────────────────
def _gpu_stats():
    if not HAS_GPU: return {}
    try:
        if GPU_TYPE == "nvidia":
            parts = [p.strip() for p in subprocess.run(
                ["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2).stdout.strip().split(",")]
            return dict(gpu_percent=float(parts[0]), gpu_temp_c=float(parts[1]),
                        gpu_mem_used_mb=float(parts[2]), gpu_mem_total_mb=float(parts[3]),
                        gpu_name=parts[4] if len(parts)>4 else "NVIDIA GPU")
    except: pass
    return {}

def get_metrics():
    global _last_net, _last_net_time, _last_disk, _last_disk_time
    m = {}
    m["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    m["cpu_cores"]   = psutil.cpu_count(logical=False) or 1
    m["cpu_threads"] = psutil.cpu_count(logical=True)  or 1
    try:
        f = psutil.cpu_freq()
        m["cpu_freq_mhz"] = round(f.current) if f else 0
    except: m["cpu_freq_mhz"] = 0
    try:
        temps = psutil.sensors_temperatures()
        for k in ["coretemp","k10temp","cpu_thermal","acpitz"]:
            if k in temps and temps[k]:
                m["cpu_temp_c"] = round(temps[k][0].current, 1); break
    except: pass

    ram = psutil.virtual_memory()
    m.update(ram_total_gb=round(ram.total/1e9,1), ram_used_gb=round(ram.used/1e9,1), ram_percent=ram.percent)
    d = psutil.disk_usage(DISK_ROOT)
    m.update(disk_total_gb=round(d.total/1e9,1), disk_used_gb=round(d.used/1e9,1), disk_percent=d.percent)
    now = time.time()
    try:
        dio = psutil.disk_io_counters()
        if _last_disk and _last_disk_time:
            dt = now - _last_disk_time
            if dt > 0:
                m["disk_read_mbps"]  = round((dio.read_bytes  - _last_disk.read_bytes)  / dt / 1e6, 2)
                m["disk_write_mbps"] = round((dio.write_bytes - _last_disk.write_bytes) / dt / 1e6, 2)
        _last_disk, _last_disk_time = dio, now
    except: pass
    try:
        net = psutil.net_io_counters()
        if _last_net and _last_net_time:
            dt = now - _last_net_time
            if dt > 0:
                m["net_dl_mbps"] = round((net.bytes_recv - _last_net.bytes_recv) / dt / 1e6, 2)
                m["net_ul_mbps"] = round((net.bytes_sent - _last_net.bytes_sent) / dt / 1e6, 2)
        _last_net, _last_net_time = net, now
    except: pass
    m["uptime_sec"] = int(time.time() - psutil.boot_time())
    m.update(_gpu_stats())

    # Top processes
    procs = []
    for p in psutil.process_iter(["pid","name","status"]):
        try:
            procs.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "cpu": round(p.cpu_percent(interval=None) or 0, 1),
                "mem": round(p.memory_percent() or 0, 1),
                "status": p.info["status"]
            })
        except: pass
    procs.sort(key=lambda x: x["cpu"], reverse=True)
    m["processes"] = procs[:20]
    return m


# ── Per-process metric sample ─────────────────────────────────
# State for per-process net delta
_proc_net_prev: dict = {}

def get_process_sample(proc_name: str, metrics: list[str]) -> dict:
    """
    Returns one sample for the named process.
    metrics: list of keys from ['cpu','mem','gpu','disk','net','power']
    Only computes what's requested — not all at once.
    """
    sample = {"ts": time.time(), "proc": proc_name, "found": False}

    # Find the process(es) matching name — sum/avg if multiple instances
    targets = []
    for p in psutil.process_iter(["pid","name","cpu_percent","memory_percent","memory_info","status"]):
        try:
            if p.info["name"] and p.info["name"].lower() == proc_name.lower():
                targets.append(p)
        except: pass

    if not targets:
        sample["error"] = f"Process '{proc_name}' not found"
        # Fire notification if process disappears mid-poll
        _push_notification(f"Process missing", f"'{proc_name}' not found during monitoring", "warn", 5000)
        return sample

    sample["found"]       = True
    sample["instance_count"] = len(targets)

    if "cpu" in metrics:
        vals = []
        for p in targets:
            try: vals.append(p.cpu_percent(interval=0.05))
            except: pass
        sample["cpu"] = round(sum(vals), 2) if vals else 0.0

    if "mem" in metrics:
        vals = []
        for p in targets:
            try: vals.append(p.memory_percent())
            except: pass
        sample["mem"] = round(sum(vals) / len(vals), 2) if vals else 0.0

    if "disk" in metrics:
        total_rd, total_wr = 0.0, 0.0
        for p in targets:
            try:
                io = p.io_counters()
                pid = p.pid
                prev = _proc_net_prev.get(f"disk_{pid}")
                if prev:
                    dt = sample["ts"] - prev["ts"]
                    if dt > 0:
                        total_rd += (io.read_bytes  - prev["rb"]) / dt / 1e6
                        total_wr += (io.write_bytes - prev["wb"]) / dt / 1e6
                _proc_net_prev[f"disk_{pid}"] = {"ts":sample["ts"],"rb":io.read_bytes,"wb":io.write_bytes}
            except: pass
        sample["disk"] = round(total_rd + total_wr, 2)

    if "net" in metrics:
        total = 0.0
        for p in targets:
            try:
                conns = p.net_connections(kind="all")
                sample["net_conns"] = len(conns)
                # Use system-wide net as proxy per process (psutil limitation)
                # Real per-proc net requires root / extended APIs
                pass
            except: pass
        # Fallback: use system net / process count
        try:
            sys_net = psutil.net_io_counters()
            pid = targets[0].pid
            prev = _proc_net_prev.get(f"net_{pid}")
            now_t = sample["ts"]
            if prev:
                dt = now_t - prev["ts"]
                if dt > 0:
                    total = (sys_net.bytes_recv + sys_net.bytes_sent - prev["total"]) / dt / 1e6 / max(1, len(targets))
            _proc_net_prev[f"net_{pid}"] = {"ts":now_t,"total":sys_net.bytes_recv+sys_net.bytes_sent}
        except: pass
        sample["net"] = round(total, 3)

    if "power" in metrics:
        # Real power requires platform APIs (RAPL on Linux, WMI on Windows)
        # Estimate: proportional CPU usage × TDP guess
        cpu_val = sample.get("cpu", 0)
        tdp     = 65.0  # watts — adjust for your CPU
        sample["power"] = round(cpu_val / 100.0 * tdp, 1)

    if "gpu" in metrics:
        gpu = _gpu_stats()
        sample["gpu"] = gpu.get("gpu_percent", 0.0)

    # Auto threshold notifications
    if sample.get("cpu", 0) > 90:
        _push_notification(f"High CPU", f"{proc_name} CPU at {sample['cpu']}%", "warn", 5000)
    if sample.get("mem", 0) > 80:
        _push_notification(f"High Memory", f"{proc_name} mem at {sample['mem']}%", "warn", 5000)

    return sample


# ── Graph data ────────────────────────────────────────────────
def get_graph():
    now = time.time()
    nodes, edges, node_set = [], [], set()

    def add_node(nid, **kw):
        if nid not in node_set: nodes.append({"id":nid,**kw}); node_set.add(nid)
    def add_edge(src, dst, **kw):
        if src in node_set and dst in node_set: edges.append({"src":src,"dst":dst,**kw})

    all_procs = []
    for p in psutil.process_iter(["pid","name","memory_info","status","ppid","num_threads","username"]):
        try: all_procs.append(p)
        except: pass

    def score(p):
        try: return (p.cpu_percent(interval=None) or 0)*.6 + (p.memory_percent() or 0)*.4
        except: return 0

    top20 = sorted(all_procs, key=score, reverse=True)[:20]

    for p in top20:
        try:
            info = getattr(p, "info", {}) or {}
            pid = info.get("pid")
            if pid is None:
                continue
            name = (info.get("name") or f"pid:{pid}")[:28]
            conns=[]; ofiles=[]
            try: conns = p.net_connections(kind="all")
            except: pass
            try: ofiles = p.open_files()
            except: pass
            try: cpu = round(p.cpu_percent(interval=None) or 0, 1)
            except: cpu = 0.0
            try: mem = round(p.memory_percent() or 0, 2)
            except: mem = 0.0
            mem_info = info.get("memory_info")
            try: mem_mb = round((getattr(mem_info, "rss", 0) or 0) / 1e6, 1)
            except: mem_mb = 0.0
            add_node(f"proc:{pid}", type="process", pid=pid, name=name,
                     cpu=cpu, mem=mem,
                     mem_mb=mem_mb,
                     status=info.get("status") or "?",
                     threads=info.get("num_threads") or 0,
                     user=(info.get("username") or "")[:20],
                     conn_count=len(conns), file_count=len(ofiles))
        except:
            continue

    for p in top20:
        try:
            info = getattr(p, "info", {}) or {}
            pid = info.get("pid")
            if pid is None:
                continue
            src=f"proc:{pid}"
            ppid = info.get("ppid")
            if ppid and f"proc:{ppid}" in node_set:
                add_edge(f"proc:{ppid}", src, type="parent_child")
            try:
                for f in p.open_files()[:8]:
                    fpath=f.path;
                    if not fpath: continue
                    fid=f"file:{fpath}"; ext=os.path.splitext(fpath)[1].lower()
                    ftype=("dll" if ext in(".dll",".so",".dylib") else "log" if ext in(".log",".txt") else
                           "db" if ext in(".db",".sqlite",".db3") else "config" if ext in(".cfg",".ini",".conf",".toml",".json",".yaml") else
                           "exe" if ext in(".exe",".py",".sh") else "file")
                    add_node(fid,type="file",path=fpath,ftype=ftype,label=os.path.basename(fpath) or fpath[:30])
                    add_edge(src,fid,type="file_open",mode=getattr(f,"mode","r"))
            except: pass
            try:
                for c in p.net_connections(kind="all")[:8]:
                    if c.raddr and c.raddr.ip:
                        rip,rport=c.raddr.ip,c.raddr.port; nid=f"net:{rip}:{rport}"
                        proto=("TCP" if c.type==1 else "UDP" if c.type==2 else "?")
                        conn_status = getattr(c, "status", "?") or "?"
                        add_node(nid,type="network",ip=rip,port=rport,proto=proto,
                                 status=conn_status,label=f"{rip}:{rport}")
                        add_edge(src,nid,type="network",proto=proto,status=conn_status)
            except: pass
        except:
            continue

    return {"nodes":nodes,"edges":edges,"ts":now}


# ── HTTP Handler ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-API-Key")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/metrics":
            self._json(get_metrics())

        elif path == "/graph":
            self._json(get_graph())

        elif path == "/monitor/sample":
            # ?proc=chrome.exe&metrics=cpu,mem&interval=2
            proc    = qs.get("proc",    [""])[0].strip()
            metrics = [m.strip() for m in qs.get("metrics", ["cpu,mem"])[0].split(",") if m.strip()]
            if not proc:
                self._json({"error": "proc param required"}, 400); return
            # Enforce max 2 metrics
            metrics = metrics[:2]
            sample  = get_process_sample(proc, metrics)
            self._json(sample)

        elif path == "/notify/stream":
            # SSE — keep connection open
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()

            q = queue.Queue()
            with _sse_lock:
                _sse_listeners.append(q)

            try:
                # Send a heartbeat every 15s to keep alive
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # heartbeat comment
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _sse_lock:
                    try: _sse_listeners.remove(q)
                    except: pass

        elif path == "/notify/pending":
            # Returns all queued notifications and clears the queue
            items = []
            while not _notif_queue.empty():
                try: items.append(_notif_queue.get_nowait())
                except: break
            self._json(items)

        elif path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/notify":
            # Your backend calls this to push a notification to the frontend
            # Body: { "title": "...", "message": "...", "level": "info", "duration_ms": 5000 }
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length).decode()) if length else {}
                n = _push_notification(
                    title       = body.get("title",   "Notification"),
                    message     = body.get("message", ""),
                    level       = body.get("level",   "info"),
                    duration_ms = int(body.get("duration_ms", 5000))
                )
                self._json({"ok": True, "id": n["id"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self.send_response(404); self.end_headers()


# ── Background: auto-alert on system thresholds ───────────────
def _threshold_watcher():
    """Watches system-wide metrics and fires notifications on thresholds."""
    THRESHOLDS = {
        "cpu_percent": (85, "warn",  "High CPU",    "System CPU usage above {val}%"),
        "ram_percent": (90, "warn",  "High RAM",    "RAM usage at {val}%"),
        "disk_percent":(95, "error", "Disk Full",   "Disk usage at {val}% — low space!"),
    }
    cooldowns = {}  # key -> last_notified_ts
    COOLDOWN = 60   # seconds between repeated alerts for same key

    while True:
        try:
            m = get_metrics()
            now = time.time()
            for key, (thresh, level, title, msg_tmpl) in THRESHOLDS.items():
                val = m.get(key)
                if val is None: continue
                if val >= thresh:
                    last = cooldowns.get(key, 0)
                    if now - last > COOLDOWN:
                        cooldowns[key] = now
                        _push_notification(title, msg_tmpl.format(val=round(val,1)), level, 7000)
        except Exception: pass
        time.sleep(30)


if __name__ == "__main__":
    PORT = 7861
    print(f"\n[PC Assist] Metrics server → http://localhost:{PORT}")
    print(f"[PC Assist] Endpoints:")
    print(f"  GET  /metrics            system stats")
    print(f"  GET  /graph              process graph (top 20)")
    print(f"  GET  /monitor/sample     single-process poll  ?proc=X&metrics=cpu,mem")
    print(f"  GET  /notify/stream      SSE notification stream")
    print(f"  GET  /notify/pending     poll pending notifications")
    print(f"  POST /notify             push notification from your backend")
    print(f"[PC Assist] GPU: {'YES (' + GPU_TYPE + ')' if HAS_GPU else 'NOT DETECTED'}")
    print(f"[PC Assist] Press Ctrl+C to stop\n")

    # Warm up
    psutil.cpu_percent(interval=0.5)
    get_metrics()

    # Start threshold watcher in background
    t = threading.Thread(target=_threshold_watcher, daemon=True)
    t.start()

    # Send a startup notification
    _push_notification("PC Assist started", f"Monitoring server live on port {PORT}", "ok", 4000)

    HTTPServer(("localhost", PORT), Handler).serve_forever()
