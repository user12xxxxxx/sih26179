"""Compare what the pipeline counted against a hand count of the same clip.

    python tools/benchmark.py --db out/live/events.db --truth data/truth_entrance.yaml

truth YAML (count by hand while watching the clip once):
    entries: 37
    exits: 35
    queue_samples:            # optional: minutes from the first event, true queue length
      - [2.0, 3]
      - [5.5, 6]
    empty_slots: [A2, B4]     # optional: slots that really went empty

Numbers like "counting error 3%, queue length off by 0.4 people on average" are what hold
up when judges ask how accurate the system is.
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    truth = yaml.safe_load(open(args.truth))
    con = sqlite3.connect(args.db)

    def count(t: str) -> int:
        return con.execute("SELECT COUNT(*) FROM events WHERE type = ?", (t,)).fetchone()[0]

    for key, typ in (("entries", "entry"), ("exits", "exit")):
        if key in truth:
            got, want = count(typ), int(truth[key])
            err = abs(got - want) / want if want else 0.0
            print(f"{key:8s} counted {got:4d}  true {want:4d}  error {err:.1%}")

    if truth.get("queue_samples"):
        t0 = con.execute("SELECT MIN(ts) FROM events").fetchone()[0]
        updates = [(ts, json.loads(d)["total_length"]) for ts, d in
                   con.execute("SELECT ts, data FROM events WHERE type = 'queue_update' ORDER BY ts")]
        errs = []
        for minute, true_len in truth["queue_samples"]:
            target = t0 + float(minute) * 60
            ts, got = min(updates, key=lambda u: abs(u[0] - target))
            errs.append(abs(got - true_len))
            print(f"queue @ {minute:5.1f} min: counted {got}  true {true_len}")
        print(f"queue length mean absolute error: {sum(errs) / len(errs):.2f} people")

    if "empty_slots" in truth:
        flagged = {json.loads(d)["slot"] for (d,) in
                   con.execute("SELECT data FROM events WHERE type = 'shelf_gap'")}
        real = set(truth["empty_slots"])
        tp = len(flagged & real)
        print(f"shelf gaps: caught {tp}/{len(real)} real ones, {len(flagged - real)} false alarms")


if __name__ == "__main__":
    main()
