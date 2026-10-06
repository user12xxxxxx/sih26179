from drishti.core import Bus, Module, topic_matches
from drishti.core.geometry import (Homography, covered_length, merge_intervals, point_in_polygon,
                                   segments_intersect, side_of_line)


def test_topic_matching():
    assert topic_matches("track/+", "track/entrance")
    assert not topic_matches("track/+", "track/entrance/x")
    assert topic_matches("event/#", "event/entry")
    assert topic_matches("#", "anything/at/all")
    assert not topic_matches("event/entry", "event/exit")


def test_broken_module_does_not_stop_others():
    bus = Bus()
    seen = []

    class Broken(Module):
        def on_message(self, topic, msg):
            raise RuntimeError("boom")

    class Fine(Module):
        def on_message(self, topic, msg):
            seen.append(msg)

    broken, fine = Broken("broken", bus), Fine("fine", bus)
    bus.subscribe("x/#", broken.dispatch)
    bus.subscribe("x/#", fine.dispatch)
    bus.publish("x/1", 42)
    assert seen == [42] and broken.errors == 1


def test_unsubscribe_removes_a_bound_method():
    # `m.dispatch` is a fresh object on every access, so unsubscribe must compare by equality
    bus = Bus()
    seen = []

    class Listener(Module):
        def on_message(self, topic, msg):
            seen.append(msg)

    m = Listener("listener", bus)
    bus.subscribe("x/#", m.dispatch)
    bus.publish("x/1", "before")
    bus.unsubscribe(m.dispatch)
    bus.publish("x/1", "after")
    assert seen == ["before"]


def test_geometry():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square) and not point_in_polygon((15, 5), square)
    assert segments_intersect((5, -5), (5, 5), (0, 0), (10, 0))
    assert not segments_intersect((15, -5), (15, 5), (0, 0), (10, 0))
    assert side_of_line((5, 5), (0, 0), (10, 0)) == -side_of_line((5, -5), (0, 0), (10, 0))
    assert merge_intervals([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert covered_length([(0, 2), (1, 3)], 1, 10) == 2


def test_homography_maps_corners():
    h = Homography([(0, 0), (100, 0), (100, 100), (0, 100)], [(0, 0), (10, 0), (10, 10), (0, 10)])
    x, y = h((50, 50))
    assert abs(x - 5) < 1e-6 and abs(y - 5) < 1e-6
