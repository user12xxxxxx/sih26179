"""Compile an ONNX model for a Qualcomm chip with Qualcomm AI Hub and measure it on a real device.

Free account: https://aihub.qualcomm.com  ->  pip install qai-hub  ->  qai-hub configure --api_token <token>

    python tools/aihub_export.py --model models/yolov8n.onnx --device "QCS6490 (Proxy)"
    python tools/aihub_export.py --model models/yolov8n.onnx --runtime qnn_context_binary

What you get:
  * the on-device latency and memory numbers for the "feasibility" slide,
  * a compiled model to deploy with the litert_qnn backend (drishti/modules/perception/backends.py).

`qai-hub list-devices` shows every hosted device. INT8 quantisation (much faster on the NPU)
needs calibration images - see AI Hub's quantize job docs.
TODO(owner: Qualcomm port): not run in CI; check option names against the current AI Hub docs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to an .onnx file")
    ap.add_argument("--device", default="QCS6490 (Proxy)")
    ap.add_argument("--runtime", default="tflite", help="tflite | qnn_context_binary | onnx")
    ap.add_argument("--out", default="models/compiled")
    args = ap.parse_args()

    import qai_hub as hub

    device = hub.Device(args.device)
    compile_job = hub.submit_compile_job(model=args.model, device=device, options=f"--target_runtime {args.runtime}")
    target = compile_job.get_target_model()
    if target is None:
        raise SystemExit(f"compile failed - see {compile_job.url}")
    profile_job = hub.submit_profile_job(model=target, device=device)
    profile = profile_job.download_profile()
    summary = profile.get("execution_summary", {})
    us = summary.get("estimated_inference_time")
    peak = summary.get("estimated_inference_peak_memory")
    print(f"device: {args.device}  runtime: {args.runtime}")
    if us is not None:
        print(f"inference: {us / 1000:.2f} ms  (~{1e6 / us:.0f} inferences/s)")
    if peak is not None:
        print(f"peak memory: {peak / 1e6:.1f} MB")
    print(f"details: {profile_job.url}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = target.download(str(Path(args.out) / Path(args.model).stem))
    print(f"compiled model saved to {path}")


if __name__ == "__main__":
    main()
