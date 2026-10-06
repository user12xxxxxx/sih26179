"""Command line: python -m drishti <command>

    run CONFIG.yaml [--out DIR] [--timeout S]   run a pipeline
    modules                                     list every module in the catalog
    report --db out/x/events.db [--date D]      daily KPI report (markdown)
    serve  --db out/x/events.db [--port 8000]   dashboard + JSON API over the event store
    snapshot VIDEO [--at S]                     one frame with a pixel grid, for drawing zones
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drishti")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a pipeline YAML")
    p_run.add_argument("config")
    p_run.add_argument("--out", default=None, help="output folder (default: out_dir from the YAML)")
    p_run.add_argument("--timeout", type=float, default=None, help="stop after this many wall-clock seconds")

    sub.add_parser("modules", help="list available modules")

    p_rep = sub.add_parser("report", help="print a KPI report from an event store")
    p_rep.add_argument("--db", required=True)
    p_rep.add_argument("--date", default=None, help="YYYY-MM-DD in store time (default: whole database)")
    p_rep.add_argument("--tz", default="+05:30")

    p_srv = sub.add_parser("serve", help="serve the dashboard and JSON API")
    p_srv.add_argument("--db", required=True)
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    p_snap = sub.add_parser("snapshot", help="save a video frame with a numbered pixel grid")
    p_snap.add_argument("video")
    p_snap.add_argument("--at", type=float, default=1.0, help="seconds into the video (default 1)")
    p_snap.add_argument("--out", default=None, help="output image (default: <video>_grid.jpg)")
    p_snap.add_argument("--grid", type=int, default=100, help="grid spacing in pixels")

    # Windows consoles default to a legacy code page; print the rupee sign and friends safely.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "run":
        from .core.pipeline import Pipeline, load_config

        Pipeline(load_config(args.config), out_dir=args.out).run(timeout=args.timeout)
    elif args.cmd == "modules":
        from .core.catalog import CATALOG

        for name, path in CATALOG.items():
            print(f"{name:32s} {path}")
    elif args.cmd == "report":
        from .reports.daily import build_report

        print(build_report(args.db, day=args.date, tz=args.tz))
    elif args.cmd == "serve":
        from .api.server import serve

        serve(args.db, host=args.host, port=args.port)
    elif args.cmd == "snapshot":
        from .snapshot import snapshot

        snapshot(args.video, at_s=args.at, out=args.out, grid=args.grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
