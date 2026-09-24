"""External CUDA worker. Run with the Python environment used by SEA-RAFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .core import CHANNELS, cache_identity, fingerprint, frame_pairs, frame_path, layer, validate_job


def _dependencies():
    # Nuke sets a process DLL directory that Windows passes to child processes.
    # Its OpenEXR libraries can crash the worker's OpenEXR wheel during I/O.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        import OpenEXR
        import Imath
    except ImportError as exc:
        raise RuntimeError("Worker requires numpy, torch, OpenEXR and Imath in the SEA-RAFT Python environment") from exc
    return np, torch, functional, OpenEXR, Imath


def _read_rgb(path: Path, width: int, height: int, np, OpenEXR, Imath):
    if not path.is_file():
        raise FileNotFoundError(path)
    source = OpenEXR.InputFile(str(path))
    try:
        header = source.header()
        window = header["dataWindow"]
        display = header.get("displayWindow", window)
        actual = (display.max.x - display.min.x + 1, display.max.y - display.min.y + 1)
        if actual != (width, height):
            raise ValueError(f"Expected displayWindow {width}x{height}: {path}; got {actual}")
        data_width = window.max.x - window.min.x + 1
        data_height = window.max.y - window.min.y + 1
        x0, x1 = max(window.min.x, display.min.x), min(window.max.x, display.max.x) + 1
        y0, y1 = max(window.min.y, display.min.y), min(window.max.y, display.max.y) + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        image = np.zeros((height, width, 3), dtype=np.float32)
        if x0 < x1 and y0 < y1:
            for index, channel in enumerate(("R", "G", "B")):
                data = np.frombuffer(source.channel(channel, pixel_type), dtype=np.float32).reshape(data_height, data_width)
                image[y0 - display.min.y:y1 - display.min.y, x0 - display.min.x:x1 - display.min.x, index] = \
                    data[y0 - window.min.y:y1 - window.min.y, x0 - window.min.x:x1 - window.min.x]
        return image
    finally:
        source.close()


def _write_vectors(path: Path, arrays: dict, width: int, height: int, np, OpenEXR, Imath):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = OpenEXR.Header(width, height)
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    header["channels"] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in arrays}
    temporary = path.with_name(path.name + ".partial")
    output = OpenEXR.OutputFile(str(temporary), header)
    try:
        output.writePixels({name: np.ascontiguousarray(value, dtype=np.float32).tobytes()
                            for name, value in arrays.items()})
    finally:
        output.close()
    os.replace(temporary, path)


def _write_manifest(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _make_model(job, torch):
    root = Path(job["sea_raft_root"]).resolve()
    if not (root / job["config"]).is_file():
        raise FileNotFoundError(root / job["config"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "core"))
    from config.parser import json_to_args
    from raft import RAFT

    args = json_to_args(str(root / job["config"]))
    checkpoint = job["model"]
    if Path(checkpoint).is_file():
        from utils.utils import load_ckpt
        model = RAFT(args)
        load_ckpt(model, checkpoint)
    else:
        model = RAFT.from_pretrained(checkpoint, args=args)
    device = torch.device(job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return model.to(device).eval(), args, device


def _prepare(image, job, np, torch, functional, device):
    if job["preprocess"] == "clamp":
        image = np.clip(image, 0.0, 1.0)
    elif job["preprocess"] == "normalize":
        lo, hi = np.percentile(image, (1, 99))
        image = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    elif job["preprocess"] != "raw":
        raise ValueError(f"Unknown preprocessing mode: {job['preprocess']}")
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()[None].to(device)
    tensor = functional.interpolate(tensor, size=tuple(reversed(job["inference_resolution"])),
                                    mode="bilinear", align_corners=False)
    return tensor * 255.0


def _flow(model, args, source, target, job, np, torch, functional):
    with torch.no_grad():
        result = model(source, target, iters=args.iters, test_mode=True)["flow"][-1]
        result = functional.interpolate(result, size=(job["height"], job["width"]),
                                        mode="bilinear", align_corners=False)
        infer_width, infer_height = job["inference_resolution"]
        result[:, 0] *= job["width"] / infer_width
        result[:, 1] *= job["height"] / infer_height
        if job["invert_v"]:
            result[:, 1] *= -1
    return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def run(job: dict):
    job = validate_job(job)
    digest_input = hashlib.sha256()
    for frame in range(job["first"], job["last"] + 1):
        path = frame_path(job["input"], frame)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest_input.update(chunk)
    job["input_digest"] = digest_input.hexdigest()
    np, torch, functional, OpenEXR, Imath = _dependencies()
    output = frame_path(job["output"], job["first"])
    sidecar = output.parent / "smartvectors.json"
    identity = cache_identity(job)
    digest = fingerprint(job)
    if sidecar.is_file():
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        if old.get("fingerprint") != digest and not job.get("overwrite"):
            raise RuntimeError("Cache settings changed. Enable Overwrite Existing or choose another cache path.")
    # Input bake is refreshed by Nuke before every run; skip is safe only when its identity matches.
    existing_valid = sidecar.is_file() and json.loads(sidecar.read_text(encoding="utf-8")).get("fingerprint") == digest
    pending = [frame for frame in range(job["first"], job["last"] + 1)
               if job.get("overwrite") or not existing_valid or not frame_path(job["output"], frame).is_file()]
    if not pending:
        print("DONE cache already complete", flush=True)
        return
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(sidecar, {"fingerprint": digest, "settings": identity, "complete": False})
    model, args, device = _make_model(job, torch)
    loaded = {}

    def image(frame):
        if frame not in loaded:
            loaded[frame] = _prepare(_read_rgb(frame_path(job["input"], frame), job["width"], job["height"],
                                               np, OpenEXR, Imath), job, np, torch, functional, device)
        return loaded[frame]

    for index, frame in enumerate(pending, 1):
        zero = np.zeros((job["height"], job["width"]), dtype=np.float32)
        arrays = {f"{layer(level)}.{channel}": zero
                  for level in job["levels"] for channel in CHANNELS}
        source = image(frame)
        for level, direction, target in frame_pairs(frame, job["first"], job["last"], job["levels"]):
            print(f"PROGRESS {frame} {job['last']} f{level:02d} {index}/{len(pending)}", flush=True)
            flow = _flow(model, args, source, image(target), job, np, torch, functional)
            arrays[f"{layer(level)}.{direction}_u"] = flow[..., 0]
            arrays[f"{layer(level)}.{direction}_v"] = flow[..., 1]
            if target != frame:
                loaded.pop(target, None)
        _write_vectors(frame_path(job["output"], frame), arrays, job["width"], job["height"],
                       np, OpenEXR, Imath)
        print(f"FRAME {frame} {index}/{len(pending)}", flush=True)
    _write_manifest(sidecar, {"fingerprint": digest, "settings": identity, "complete": True})
    print("DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    arguments = parser.parse_args()
    try:
        run(json.loads(arguments.job.read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
