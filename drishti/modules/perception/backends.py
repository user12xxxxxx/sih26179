"""Inference backends: the one place that knows which chip runs the model.

Every backend has the same shape: ``infer(blob) -> list[np.ndarray]``. The detector does not
care which one it gets, so moving from the laptop to the Qualcomm board is a config change:

    backend: onnxruntime   providers: [CPUExecutionProvider]        # AMD 3500U laptop
    backend: onnxruntime   providers: [CUDAExecutionProvider, ...]  # Lenovo RTX 4050 (onnxruntime-gpu)
    backend: litert_qnn    model: yolov8n_qcs6490.tflite            # RUBIK Pi 3 / RB3 Gen 2 / Snapdragon phone
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Backend(Protocol):
    input_shape: tuple[int, ...]

    def infer(self, blob: np.ndarray) -> list[np.ndarray]: ...


class OnnxRuntimeBackend:
    def __init__(self, model: str, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(model, sess_options=so, providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = tuple(d if isinstance(d, int) else -1 for d in inp.shape)

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        return self.sess.run(None, {self.input_name: blob})


class LiteRtQnnBackend:
    """Runs a model exported by Qualcomm AI Hub on the Hexagon NPU via the QNN delegate.

    TODO(owner: Qualcomm port): implement once a board or phone is available.
      1. tools/aihub_export.py -> compile the ONNX model for "QCS6490 (Proxy)" with
         --target_runtime tflite (optionally quantize to INT8 first).
      2. On the board: pip install ai-edge-litert (or tflite-runtime) and load the QNN delegate
         (libQnnTFLiteDelegate.so from the QNN / Qualcomm Linux SDK):
             delegate = load_delegate("libQnnTFLiteDelegate.so", {"backend_type": "htp"})
             interpreter = Interpreter(model_path=..., experimental_delegates=[delegate])
      3. Match input layout: TFLite exports are usually NHWC, ONNX is NCHW - transpose in infer().
    """

    def __init__(self, model: str, **_: Any):
        raise NotImplementedError("LiteRT + QNN backend is a stub - see the TODO in backends.py")

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:  # pragma: no cover
        raise NotImplementedError


def make_backend(cfg: dict[str, Any]) -> Backend:
    kind = cfg.get("backend", "onnxruntime")
    if kind == "onnxruntime":
        return OnnxRuntimeBackend(cfg["model"], cfg.get("providers"), cfg.get("threads"))
    if kind == "litert_qnn":
        return LiteRtQnnBackend(cfg["model"])
    raise ValueError(f"unknown backend '{kind}'")
