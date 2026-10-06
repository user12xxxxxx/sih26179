import math

from drishti.core import Event
from drishti.modules.analytics.heatmap import Heatmap
from drishti.modules.analytics.line_counter import LineCounter
from drishti.modules.analytics.planogram import PlanogramCheck
from drishti.modules.analytics.queue_forecast import QueueForecast, erlang_c, expected_arrivals, expected_wait_s
from drishti.modules.analytics.queue_monitor import QueueMonitor
from drishti.modules.analytics.shelf_gaps import analyse_row
from drishti.modules.analytics.shelf_state import ShelfState
from drishti.modules.analytics.zone_dwell import ZoneDwell
from tests.helpers import make, tracks

CAMS = {
    "entrance": {
        "lines": {"door": {"a": [0, 400], "b": [1280, 400], "in_point": [640, 700]}},
        "zones": {"promo": [[900, 450], [1280, 450], [1280, 720], [900, 720]]},
    },
    "checkout": {
        "zones": {"queue_1": [[100, 100], [300, 100], [300, 560], [100, 560]],
                  "counter_1": [[100, 560], [300, 560], [300, 720], [100, 720]]},
    },
}


def test_line_counter_counts_in_and_out():
    _, bus, rec = make(LineCounter, {"camera": "entrance", "line": "door"}, CAMS)
    for i, y in enumerate([300, 350, 450, 500]):  # walks in
        bus.publish("track/entrance", tracks("entrance", i, {1: (640, y)}))
    for i, y in enumerate([500, 420, 380]):  # walks out
        bus.publish("track/entrance", tracks("entrance", 10 + i, {2: (600, y)}))
    assert [e.type for e in rec.of("event/")] == ["entry", "exit"]


def test_zone_dwell_reports_time_spent():
    _, bus, rec = make(ZoneDwell, {"camera": "entrance", "zones": ["promo"], "min_dwell_s": 3}, CAMS)
    for t in range(0, 11):
        bus.publish("track/entrance", tracks("entrance", float(t), {7: (1000, 600)}))
    bus.publish("track/entrance", tracks("entrance", 11.0, {7: (500, 600)}))  # walks out of the zone
    dwell = [e for e in rec.of("event/dwell")]
    assert len(dwell) == 1 and dwell[0].data["seconds"] == 10.0


def test_heatmap_accumulates_where_people_stand(tmp_path):
    mod, bus, rec = make(Heatmap, {"camera": "entrance", "grid": [4, 4], "interval_s": 5, "save_png": False}, CAMS, tmp_path)
    for t in range(0, 7):
        bus.publish("track/entrance", tracks("entrance", float(t), {1: (100, 100)}))
    grid = rec.of("event/heatmap")[0].data["grid"]
    assert grid[0][0] == 1.0 and sum(map(sum, grid)) == 1.0


def test_queue_monitor_measures_wait_and_service():
    _, bus, rec = make(QueueMonitor, {"camera": "checkout", "interval_s": 1,
                                      "counters": [{"id": 1, "queue_zone": "queue_1", "counter_zone": "counter_1"}]}, CAMS)
    for t in range(0, 30):  # 30 s in the queue
        bus.publish("track/checkout", tracks("checkout", float(t), {5: (200, 400)}))
    for t in range(30, 50):  # 20 s at the counter
        bus.publish("track/checkout", tracks("checkout", float(t), {5: (200, 640)}))
    for t in range(50, 54):  # gone
        bus.publish("track/checkout", tracks("checkout", float(t), {}))
    done = rec.of("event/service_complete")[0].data
    assert done["wait_s"] == 30.0 and done["service_s"] == 19.0
    assert any(u.data["total_length"] == 1 for u in rec.of("event/queue_update"))


def test_erlang_c_textbook_values():
    assert abs(erlang_c(1, 0.5) - 0.5) < 1e-9  # one server: P(wait) = utilisation
    assert abs(erlang_c(2, 1.0) - 1 / 3) < 1e-9
    assert expected_wait_s(lam=1 / 60, mu=1 / 60, c=1) == math.inf  # overloaded
    assert abs(expected_wait_s(lam=0.5 / 60, mu=1 / 60, c=1) - 60.0) < 1e-6  # Wq = rho / (mu - lam) = 1 min


def test_expected_arrivals_uses_shopping_time():
    # Someone who entered 8 min ago with shopping time U(6, 12) min: 3 of the remaining... window
    n = expected_arrivals([0.0], now=8 * 60, lo_s=6 * 60, hi_s=12 * 60, horizon_s=5 * 60)
    assert abs(n - 4 / 6) < 1e-9  # P(8 < L <= 12) with L ~ U(6, 12)
    assert expected_arrivals([0.0], now=13 * 60, lo_s=360, hi_s=720, horizon_s=300) == 0.0


def test_forecast_asks_for_more_counters_before_the_rush():
    _, bus, rec = make(QueueForecast, {"shop_time_min": [6, 12], "horizon_min": 5, "max_counters": 3,
                                       "default_service_s": 45, "interval_s": 1, "checkout_prob": 1.0})
    t = 0.0
    for i in range(30):  # 30 people walk in within 5 minutes
        t = i * 10.0
        bus.publish("event/entry", Event("entry", "entrance", t))
    bus.publish("event/queue_update", Event("queue_update", "checkout", 9 * 60.0, {"total_length": 0}))
    last = rec.of("event/queue_forecast")[-1].data
    assert last["queue_length"] == 0 and last["recommended_counters"] >= 2


def test_shelf_row_occupancy_and_gaps():
    row = {"id": "A", "rect": [0, 0, 1000, 100],
           "slots": [{"id": "A1", "x": [0, 500]}, {"id": "A2", "x": [500, 1000]}]}
    products = [(0, 10, 100, 90), (100, 10, 200, 90), (200, 10, 450, 90)]  # A2 is empty
    out = analyse_row(row, products, min_gap_frac=0.08)
    occ = {s["id"]: s["occupancy"] for s in out["slots"]}
    assert occ["A1"] == 0.9 and occ["A2"] == 0.0 and out["gaps"] == [[450, 1000]]


def test_shelf_state_needs_consistent_evidence():
    _, bus, rec = make(ShelfState, {"window": 3})

    def obs(ts, occ):
        bus.publish("event/shelf_observation", Event("shelf_observation", "shelf_a", ts,
                                                     {"row": "A", "slots": [{"id": "A1", "product": "Maggi", "occupancy": occ}]}))
    for ts, occ in enumerate([1.0, 1.0, 0.0, 1.0, 1.0]):  # one bad frame: no alert
        obs(ts, occ)
    assert rec.of("event/shelf_gap") == []
    for ts in range(5, 8):  # really empty now
        obs(ts, 0.0)
    gaps = rec.of("event/shelf_gap")
    assert len(gaps) == 1 and gaps[0].data["state"] == "EMPTY"
    for ts in range(8, 11):
        obs(ts, 0.9)
    assert len(rec.of("event/shelf_restocked")) == 1


def test_planogram_flags_wrong_label():
    cams = {"shelf_a": {"shelves": [{"id": "A", "rect": [0, 0, 1000, 100], "slots": [
        {"id": "A1", "x": [0, 500], "product": "Maggi Noodles", "price": 14}]}]}}
    _, bus, rec = make(PlanogramCheck, {"camera": "shelf_a"}, cams)
    bus.publish("event/text_read", Event("text_read", "shelf_a", 1.0, {"text": "Sunfeast Yippee", "bbox": (100, 90, 200, 110),
                                                                        "parsed": {}}))
    bus.publish("event/text_read", Event("text_read", "shelf_a", 2.0, {"text": "MRP Rs 16", "bbox": (100, 90, 200, 110),
                                                                        "parsed": {"price": 16.0}}))
    assert rec.of("event/planogram_mismatch")[0].data["slot"] == "A1"
    assert rec.of("event/price_mismatch")[0].data["read_price"] == 16.0
