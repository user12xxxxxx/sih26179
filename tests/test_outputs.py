import json
import sqlite3

from drishti.core import Alert, Event
from drishti.modules.outputs.alert_engine import AlertEngine, matches, render
from drishti.modules.outputs.event_store import EventStore
from tests.helpers import make

RULE = {"event": "queue_forecast", "where": {"recommended_counters": {"gt": "$open_counters"}},
        "kind": "open_counter", "severity": "critical", "message": "Open counter {recommended_counters}",
        "key": "queue", "cooldown_s": 300}


def test_rule_conditions():
    assert matches({"state": "EMPTY"}, {"state": "EMPTY"})
    assert not matches({"state": "EMPTY"}, {"state": "LOW"})
    assert matches({"n": {"gte": 6}}, {"n": 6}) and not matches({"n": {"gt": 6}}, {"n": 6})
    assert matches({"a": {"gt": "$b"}}, {"a": 3, "b": 2})
    assert not matches({"a": {"gt": 1}}, {})  # missing field never matches
    assert render("Refill {product} ({slot})", {"product": "Maggi"}) == "Refill Maggi (?)"


def test_alert_engine_fires_once_per_cooldown():
    _, bus, rec = make(AlertEngine, {"rules": [RULE]})
    for ts in (0, 60, 400):
        bus.publish("event/queue_forecast", Event("queue_forecast", "checkout", ts,
                                                   {"recommended_counters": 2, "open_counters": 1}))
    bus.publish("event/queue_forecast", Event("queue_forecast", "checkout", 800, {"recommended_counters": 2, "open_counters": 2}))
    alerts = rec.of("alert/")
    assert [a.ts for a in alerts] == [0, 400] and alerts[0].message == "Open counter 2"


def test_event_store_writes_everything(tmp_path):
    mod, bus, _ = make(EventStore, {"path": "e.db"}, out_dir=tmp_path)
    bus.publish("event/entry", Event("entry", "entrance", 1.0, {"track_id": 3}))
    bus.publish("alert/restock", Alert("restock", "Refill Maggi", "warn", 2.0))
    mod.teardown()
    con = sqlite3.connect(tmp_path / "e.db")
    assert json.loads(con.execute("SELECT data FROM events").fetchone()[0]) == {"track_id": 3}
    assert con.execute("SELECT message FROM alerts").fetchone()[0] == "Refill Maggi"
