"""The module catalog: short names used in pipeline YAML -> Python class.

Adding a module = add a file under drishti/modules/ and one line here.
A YAML ``use:`` may also give a full path ("my_pkg.my_file:MyClass") for experiments.
Classes are imported lazily, so a missing optional library only breaks the module that needs it.
"""

from __future__ import annotations

import importlib

CATALOG: dict[str, str] = {
    # sources ---------------------------------------------------------------------------------
    "source.video_file": "drishti.modules.sources.video_file:VideoFileSource",
    "source.rtsp": "drishti.modules.sources.rtsp:RtspSource",
    "source.synthetic": "drishti.modules.sources.synthetic:SyntheticStore",
    # perception (the only layer that sees frames) --------------------------------------------
    "perception.detector": "drishti.modules.perception.detector:Detector",
    "perception.tracker": "drishti.modules.perception.tracker:Tracker",
    "perception.ocr": "drishti.modules.perception.ocr:OcrReader",
    "perception.redacted_snapshots": "drishti.modules.perception.privacy:RedactedSnapshots",
    "perception.shelf_reference": "drishti.modules.perception.shelf_reference:ShelfReference",
    # analytics (metadata in, events out) -----------------------------------------------------
    "analytics.line_counter": "drishti.modules.analytics.line_counter:LineCounter",
    "analytics.zone_dwell": "drishti.modules.analytics.zone_dwell:ZoneDwell",
    "analytics.heatmap": "drishti.modules.analytics.heatmap:Heatmap",
    "analytics.queue_monitor": "drishti.modules.analytics.queue_monitor:QueueMonitor",
    "analytics.queue_forecast": "drishti.modules.analytics.queue_forecast:QueueForecast",
    "analytics.shelf_gaps": "drishti.modules.analytics.shelf_gaps:ShelfGaps",
    "analytics.shelf_state": "drishti.modules.analytics.shelf_state:ShelfState",
    "analytics.planogram": "drishti.modules.analytics.planogram:PlanogramCheck",
    # outputs ---------------------------------------------------------------------------------
    "output.alert_engine": "drishti.modules.outputs.alert_engine:AlertEngine",
    "output.console": "drishti.modules.outputs.console:ConsoleNotifier",
    "output.event_store": "drishti.modules.outputs.event_store:EventStore",
    "output.mqtt_bridge": "drishti.modules.outputs.mqtt_bridge:MqttBridge",
    # integrations ----------------------------------------------------------------------------
    "integration.pos_mock": "drishti.modules.integrations.pos_mock:PosMock",
    "integration.pos_csv": "drishti.modules.integrations.pos_csv:PosCsv",
    "integration.cloud_sync": "drishti.modules.integrations.cloud_sync:CloudSync",
    # ops -------------------------------------------------------------------------------------
    "ops.health": "drishti.modules.ops.health:Health",
}


def resolve(name: str) -> type:
    path = CATALOG.get(name, name)
    if ":" not in path:
        known = ", ".join(sorted(CATALOG))
        raise KeyError(f"unknown module '{name}'. Known: {known}")
    mod_name, _, cls_name = path.partition(":")
    return getattr(importlib.import_module(mod_name), cls_name)
