"""End to end: the synthetic store through every analytics module, the alert engine, the event
store, the report and the API. If this passes, the wiring between modules is intact."""

import json
import sqlite3
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from drishti.api.server import Store, make_handler
from drishti.core.pipeline import Pipeline, load_config
from drishti.reports.daily import build_report

ROOT = Path(__file__).resolve().parents[1]


def run_demo(tmp_path, respond: bool = True) -> Path:
    cfg = load_config(ROOT / "configs" / "synthetic_demo.yaml")
    cfg["modules"][0]["config"]["respond_to_alerts"] = respond
    cfg["modules"] = [m for m in cfg["modules"] if m["use"] != "output.console"]
    Pipeline(cfg, out_dir=tmp_path).run(timeout=60)
    return tmp_path / "events.db"


def test_forecast_opens_counters_before_the_queue_builds(tmp_path):
    db = run_demo(tmp_path / "acting")
    con = sqlite3.connect(db)
    entries = con.execute("SELECT COUNT(*) FROM events WHERE type = 'entry'").fetchone()[0]
    kinds = [k for (k,) in con.execute("SELECT kind FROM alerts")]
    queue = [json.loads(d)["total_length"] for (d,) in con.execute("SELECT data FROM events WHERE type = 'queue_update'")]
    assert entries > 50 and "open_counter" in kinds and max(queue) <= 5

    # same store, nobody acts on the alerts: the queue gets much longer
    db2 = run_demo(tmp_path / "ignoring", respond=False)
    queue2 = [json.loads(d)["total_length"] for (d,) in
              sqlite3.connect(db2).execute("SELECT data FROM events WHERE type = 'queue_update'")]
    assert max(queue2) >= 2 * max(queue)


def test_report_and_api_read_the_store(tmp_path):
    db = run_demo(tmp_path)
    report = build_report(str(db))
    assert "Visitors in" in report and "Conversion" in report

    kpis = Store(db).kpis()
    assert kpis["entries"] > 50 and kpis["forecast"]["recommended_counters"] >= 1

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Store(db)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert json.load(urllib.request.urlopen(f"{base}/api/kpis"))["entries"] == kpis["entries"]
        assert json.load(urllib.request.urlopen(f"{base}/api/heatmap/entrance"))["cols"] == 64
        assert b"DukaanDrishti" in urllib.request.urlopen(f"{base}/").read()
        assert b"Enable alerts" in urllib.request.urlopen(f"{base}/alerts").read()
    finally:
        httpd.shutdown()
