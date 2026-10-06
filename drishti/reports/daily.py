"""Daily / weekly KPI report from the event store (markdown, printable or pasteable).

    python -m drishti report --db out/synthetic/events.db [--date 2026-10-01]

KPIs: footfall by hour, conversion (bills / visitors), dwell per zone, queue wait and service
times, counter recommendations, shelf gaps and restock alerts.
TODO(owner: reports): weekly roll-up, staff-efficiency KPI (alerts acknowledged / time to
restock), PDF export.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean


def parse_tz(s: str) -> timezone:
    sign = -1 if s.startswith("-") else 1
    hh, mm = s.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def build_report(db: str, day: str | None = None, tz: str = "+05:30") -> str:
    tzinfo = parse_tz(tz)
    con = sqlite3.connect(db)
    where, params = "", ()
    if day:
        start = datetime.fromisoformat(day).replace(tzinfo=tzinfo).timestamp()
        where, params = "WHERE ts >= ? AND ts < ?", (start, start + 86400)
    events = [(ts, typ, json.loads(data)) for ts, typ, data in
              con.execute(f"SELECT ts, type, data FROM events {where} ORDER BY ts", params)]
    alerts = con.execute(f"SELECT ts, kind, message FROM alerts {where} ORDER BY ts", params).fetchall()
    con.close()
    if not events:
        return "No events in this period."

    by_type: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for ts, typ, data in events:
        by_type[typ].append((ts, data))

    def hhmm(ts: float) -> str:
        return datetime.fromtimestamp(ts, tzinfo).strftime("%H:%M")

    first, last = events[0][0], events[-1][0]
    entries, exits = len(by_type["entry"]), len(by_type["exit"])
    bills = by_type["pos_bill"]
    lines = [
        f"# Store report {datetime.fromtimestamp(first, tzinfo):%d %b %Y}, {hhmm(first)}-{hhmm(last)}",
        "",
        "## Footfall",
        f"- Visitors in: **{entries}**, out: {exits}",
    ]
    per_hour = Counter(datetime.fromtimestamp(ts, tzinfo).strftime("%H:00") for ts, _ in by_type["entry"])
    if per_hour:
        peak = max(per_hour, key=per_hour.get)
        lines.append(f"- Busiest hour: {peak} ({per_hour[peak]} visitors)")
        lines.append("- By hour: " + ", ".join(f"{h} {n}" for h, n in sorted(per_hour.items())))
    if bills:
        total = sum(d.get("amount", 0) for _, d in bills)
        conv = f"{len(bills) / entries:.0%}" if entries else "n/a"
        lines += ["", "## Sales (from POS)", f"- Bills: {len(bills)}, revenue: ₹{total:,.0f}", f"- Conversion (bills / visitors): **{conv}**"]

    dwell: dict[str, list[float]] = defaultdict(list)
    for _, d in by_type["dwell"]:
        dwell[d["zone"]].append(d["seconds"])
    if dwell:
        lines += ["", "## Dwell time"]
        lines += [f"- {z}: {len(v)} stops, average {mean(v):.0f} s" for z, v in sorted(dwell.items())]

    served = by_type["service_complete"]
    if served or by_type["queue_update"]:
        lines += ["", "## Billing queues"]
        if served:
            lines.append(f"- Customers served: {len(served)}; average wait **{mean(d['wait_s'] for _, d in served):.0f} s**, "
                         f"average billing time {mean(d['service_s'] for _, d in served):.0f} s")
        if by_type["queue_update"]:
            ts_max, d_max = max(by_type["queue_update"], key=lambda x: x[1].get("total_length", 0))
            lines.append(f"- Longest queue: {d_max.get('total_length', 0)} people at {hhmm(ts_max)}")
        if by_type["queue_abandoned"]:
            lines.append(f"- Left the queue without paying: {len(by_type['queue_abandoned'])}")

    gaps = by_type["shelf_gap"]
    if gaps or by_type["shelf_restocked"]:
        lines += ["", "## Shelves"]
        lines.append(f"- Gaps detected: {len(gaps)} ({sum(1 for _, d in gaps if d.get('state') == 'EMPTY')} empty), "
                     f"restocked: {len(by_type['shelf_restocked'])}")

    if alerts:
        kinds = Counter(k for _, k, _ in alerts)
        lines += ["", "## Alerts", "- " + ", ".join(f"{k}: {n}" for k, n in kinds.most_common())]
        lines += [f"  - {hhmm(ts)} {msg}" for ts, _, msg in alerts[:10]]
        if len(alerts) > 10:
            lines.append(f"  - ... and {len(alerts) - 10} more")
    return "\n".join(lines)
