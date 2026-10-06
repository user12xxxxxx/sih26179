"""analytics.zone_dwell - how long people spend in named zones (promo display, aisle, ...).

Config:
    camera: entrance
    zones: [promo]          names under cameras.<camera>.zones (polygons)
    min_dwell_s: 3          ignore people just walking through
    ttl_s: 2.0              a track unseen this long has left

Listens:   track/<camera>
Publishes: event/dwell   {zone, track_id, seconds}
           event/zone_occupancy   {zone, count}   every `occupancy_every_s` (default 30)
"""

from __future__ import annotations

from typing import Any

from drishti.core import Module, Tracks
from drishti.core.geometry import point_in_polygon


class ZoneDwell(Module):
    subscribes = ("track/{camera}",)

    def setup(self) -> None:
        self.require("camera", "zones")
        cam = self.cfg["camera"]
        self.zones = {z: self.ctx.zone(cam, z) for z in self.cfg["zones"]}
        self.min_dwell = float(self.cfg.get("min_dwell_s", 3))
        self.ttl = float(self.cfg.get("ttl_s", 2.0))
        self.occ_every = float(self.cfg.get("occupancy_every_s", 30))
        self.inside: dict[tuple[int, str], list[float]] = {}  # (tid, zone) -> [enter_ts, last_ts]
        self.last_occ = -1e18

    def _close(self, key: tuple[int, str], cam: str) -> None:
        enter, last = self.inside.pop(key)
        secs = last - enter
        if secs >= self.min_dwell:
            self.emit("dwell", cam, last, zone=key[1], track_id=key[0], seconds=round(secs, 1))

    def on_message(self, topic: str, tracks: Any) -> None:
        assert isinstance(tracks, Tracks)
        cam, ts = tracks.camera_id, tracks.ts
        seen: set[tuple[int, str]] = set()
        for t in tracks.items:
            for z, poly in self.zones.items():
                if point_in_polygon(t.anchor, poly):
                    key = (t.track_id, z)
                    seen.add(key)
                    if key in self.inside:
                        self.inside[key][1] = ts
                    else:
                        self.inside[key] = [ts, ts]
        for key in [k for k, (_, last) in self.inside.items() if k not in seen and ts - last > self.ttl]:
            self._close(key, cam)
        # a track still visible but outside the zone has left it
        visible = {t.track_id for t in tracks.items}
        for key in [k for k in self.inside if k not in seen and k[0] in visible]:
            self._close(key, cam)
        if ts - self.last_occ >= self.occ_every:
            self.last_occ = ts
            for z in self.zones:
                self.emit("zone_occupancy", cam, ts, zone=z, count=sum(1 for k in self.inside if k[1] == z))

    def teardown(self) -> None:
        for key in list(self.inside):
            self._close(key, self.cfg["camera"])
