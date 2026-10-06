"""Builds and runs a pipeline from YAML: which modules, with which settings, in which order."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from .bus import Bus
from .catalog import resolve
from .module import Context, Module, Source

log = logging.getLogger("drishti.pipeline")

_LAYER_RANK = {"source": 0, "perception": 1, "analytics": 2, "integration": 3, "ops": 4, "output": 5}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    # cameras may live in separate files written by tools/setup_zones.py
    for cam_id, cam in list((cfg.get("cameras") or {}).items()):
        if isinstance(cam, str):
            cfg["cameras"][cam_id] = yaml.safe_load((path.parent / cam).read_text())
    cfg.setdefault("_base_dir", str(path.parent))
    return cfg


class Pipeline:
    def __init__(self, cfg: dict[str, Any], out_dir: str | Path | None = None):
        self.cfg = cfg
        self.bus = Bus()
        self.ctx = Context(
            store=cfg.get("store") or {},
            cameras=cfg.get("cameras") or {},
            out_dir=Path(out_dir or cfg.get("out_dir", "out")),
        )
        self.modules: list[Module] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        names: set[str] = set()
        for spec in cfg.get("modules") or []:
            if not spec.get("enabled", True):
                continue
            use = spec["use"]
            name = spec.get("name") or use
            if name in names:
                raise ValueError(f"two modules are both named '{name}' - give one a 'name:'")
            names.add(name)
            cls = resolve(use)
            mod = cls(name, self.bus, spec.get("config") or {}, self.ctx)
            mod.use = use
            self.modules.append(mod)

    def get(self, name: str) -> Module:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(name)

    def start(self) -> None:
        self.ctx.out_dir.mkdir(parents=True, exist_ok=True)
        for m in self.modules:
            m.setup()
        for m in self.modules:
            for pattern in m.topics():
                self.bus.subscribe(pattern, m.dispatch)
        for m in self.modules:
            if isinstance(m, Source):
                t = threading.Thread(target=self._run_source, args=(m,), name=m.name, daemon=True)
                self._threads.append(t)
                t.start()
        log.info("started %d modules (%d sources)", len(self.modules), len(self._threads))

    def _run_source(self, src: Source) -> None:
        try:
            src.run(self._stop)
        except Exception:  # noqa: BLE001
            src.errors += 1
            log.exception("source %s crashed", src.name)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every source has finished (end of clip / simulation) or Ctrl-C."""
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while any(t.is_alive() for t in self._threads):
                if deadline is not None and time.monotonic() > deadline:
                    break
                for t in self._threads:
                    t.join(timeout=0.2)
        except KeyboardInterrupt:
            log.info("stopping (Ctrl-C)")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        # Upstream first: analytics may publish final events while tearing down, so the
        # alert engine and the event store must still be listening. Outputs close last.
        for m in sorted(self.modules, key=lambda m: _LAYER_RANK.get(getattr(m, "use", "").split(".")[0], 2)):
            try:
                m.teardown()
            except Exception:  # noqa: BLE001
                log.exception("teardown failed for %s", m.name)
            self.bus.unsubscribe(m.dispatch)

    def run(self, timeout: float | None = None) -> None:
        self.start()
        try:
            self.wait(timeout)
        finally:
            self.stop()
