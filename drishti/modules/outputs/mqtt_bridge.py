"""output.mqtt_bridge - forwards events, alerts and metrics to an MQTT broker (Mosquitto).

Uses: staff phones (any MQTT app, or ntfy / a small PWA), a second process such as the
dashboard, or other devices in the store. Frames are never forwarded - the bridge refuses
any local-only topic, which is how the "video never leaves the box" promise is enforced.

Config:
    host: localhost
    port: 1883
    prefix: drishti/demo-store        topic prefix on the broker
    forward: ["alert/#", "event/#", "metric/#"]

Listens: whatever `forward` lists. Needs: pip install paho-mqtt
TODO(owner: alerts): not run in CI - test against a local mosquitto before relying on it.
"""

from __future__ import annotations

import json
from typing import Any

from drishti.core import Module
from drishti.core.bus import is_local_only


class MqttBridge(Module):
    def topics(self) -> list[str]:
        pats = self.cfg.get("forward", ["alert/#", "event/#", "metric/#"])
        bad = [p for p in pats if p.startswith("frame") or p in ("#", "+/#")]
        if bad:
            raise ValueError(f"mqtt_bridge must not forward frames: {bad}")
        return pats

    def setup(self) -> None:
        import paho.mqtt.client as mqtt

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt 2.x
        except AttributeError:
            self.client = mqtt.Client()  # paho-mqtt 1.x
        self.client.connect(self.cfg.get("host", "localhost"), int(self.cfg.get("port", 1883)))
        self.client.loop_start()
        self.prefix = self.cfg.get("prefix", "drishti/store")

    def on_message(self, topic: str, msg: Any) -> None:
        if is_local_only(topic) or not hasattr(msg, "to_dict"):
            return
        self.client.publish(f"{self.prefix}/{topic}", json.dumps(msg.to_dict()), qos=1)

    def teardown(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
