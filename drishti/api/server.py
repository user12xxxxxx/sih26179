"""A small JSON API + static dashboard over the SQLite event store.

Standard library only (works anywhere Python runs, including the edge board). It reads the
database the pipeline writes, so it can serve a live store or a recorded simulation.
Swap in FastAPI later if you need auth or websockets - keep the same endpoints.

Endpoints (all GET, JSON):
    /api/health
    /api/kpis                  headline numbers for the tiles
    /api/alerts?limit=20
    /api/events?type=entry&limit=100
    /api/heatmap/<camera>      latest heatmap grid
    /api/series/entries?bucket_s=300   visitors per time bucket
    /                          the dashboard page (drishti/api/static/index.html)
    /alerts                    phone-friendly alert feed, beeps + vibrates on new alerts (N2)
"""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"


class Store:
    def __init__(self, db: str | Path):
        self.db = str(db)

    def _q(self, sql: str, params: tuple = ()) -> list[tuple]:
        con = sqlite3.connect(self.db)  # queries below are SELECT-only
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def latest(self, typ: str, camera: str | None = None) -> dict[str, Any] | None:
        sql, params = "SELECT ts, camera_id, data FROM events WHERE type = ?", [typ]
        if camera:
            sql += " AND camera_id = ?"
            params.append(camera)
        rows = self._q(sql + " ORDER BY ts DESC LIMIT 1", tuple(params))
        if not rows:
            return None
        ts, cam, data = rows[0]
        return {"ts": ts, "camera_id": cam, **json.loads(data)}

    def count(self, typ: str) -> int:
        return self._q("SELECT COUNT(*) FROM events WHERE type = ?", (typ,))[0][0]

    def kpis(self) -> dict[str, Any]:
        now = self._q("SELECT MAX(ts) FROM events")[0][0]
        entries, exits, bills = self.count("entry"), self.count("exit"), self.count("pos_bill")
        waits = self._q("SELECT AVG(json_extract(data, '$.wait_s')) FROM events WHERE type = 'service_complete'")[0][0]
        recent_alerts = self._q("SELECT COUNT(*) FROM alerts WHERE ts > ?", ((now or 0) - 1800,))[0][0]
        return {
            "now": now,
            "entries": entries,
            "exits": exits,
            "in_store": max(entries - exits, 0),
            "bills": bills,
            "conversion": round(bills / entries, 3) if entries else None,
            "avg_wait_s": round(waits, 1) if waits is not None else None,
            "queue": self.latest("queue_update"),
            "forecast": self.latest("queue_forecast"),
            "shelves": self.latest("shelf_status"),
            "alerts_last_30_min": recent_alerts,
        }

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        rows = self._q("SELECT ts, kind, severity, message, camera_id FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(zip(("ts", "kind", "severity", "message", "camera_id"), r)) for r in rows]

    def events(self, typ: str | None, limit: int) -> list[dict[str, Any]]:
        if typ:
            rows = self._q("SELECT ts, type, camera_id, data FROM events WHERE type = ? ORDER BY ts DESC LIMIT ?", (typ, limit))
        else:
            rows = self._q("SELECT ts, type, camera_id, data FROM events ORDER BY ts DESC LIMIT ?", (limit,))
        return [{"ts": ts, "type": t, "camera_id": c, "data": json.loads(d)} for ts, t, c, d in rows]

    def entries_series(self, bucket_s: int) -> list[dict[str, Any]]:
        rows = self._q("SELECT CAST(ts / ? AS INTEGER) * ?, COUNT(*) FROM events WHERE type = 'entry' GROUP BY 1 ORDER BY 1",
                       (bucket_s, bucket_s))
        return [{"ts": ts, "entries": n} for ts, n in rows]


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, name: str) -> None:
            body = (STATIC / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            try:
                if path in ("/", "/index.html"):
                    self._html("index.html")
                elif path in ("/alerts", "/alerts.html"):
                    self._html("alerts.html")
                elif path == "/api/health":
                    self._json({"ok": True})
                elif path == "/api/kpis":
                    self._json(store.kpis())
                elif path == "/api/alerts":
                    self._json(store.alerts(int(qs.get("limit", 20))))
                elif path == "/api/events":
                    self._json(store.events(qs.get("type"), int(qs.get("limit", 100))))
                elif path.startswith("/api/heatmap/"):
                    self._json(store.latest("heatmap", path.rsplit("/", 1)[1]) or {})
                elif path == "/api/series/entries":
                    self._json(store.entries_series(int(qs.get("bucket_s", 300))))
                else:
                    self._json({"error": "not found"}, 404)
            except sqlite3.Error as e:
                self._json({"error": str(e)}, 500)

    return Handler


def serve(db: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    if not Path(db).exists():
        raise SystemExit(f"{db} not found - run a pipeline first (python -m drishti run configs/synthetic_demo.yaml)")
    httpd = ThreadingHTTPServer((host, port), make_handler(Store(db)))
    print(f"dashboard on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
