"""Dependency-light job planning and cache identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LEVELS = (1, 2, 4, 8, 16, 32, 64)
CHANNELS = ("p_u", "p_v", "n_u", "n_v")
MODEL = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
CONFIG = "config/eval/spring-M.json"
SCHEMA = 1


def layer(level: int) -> str:
    if level not in LEVELS:
        raise ValueError(f"Invalid temporal level: {level}")
    return f"smartvector_f{level:02d}_v01"


def frame_path(pattern: str, frame: int) -> Path:
    if "####" not in pattern:
        raise ValueError("Sequence pattern must contain ####")
    return Path(pattern.replace("####", f"{frame:04d}"))


def inference_size(width: int, height: int, max_dimension: int) -> tuple[int, int]:
    if min(width, height) <= 0:
        raise ValueError("Input dimensions must be positive")
    if max_dimension < 0:
        raise ValueError("max_dimension must be nonnegative")
    scale = min(1.0, max_dimension / max(width, height)) if max_dimension else 1.0
    return max(1, round(width * scale)), max(1, round(height * scale))


def validate_job(job: dict) -> dict:
    required = ("input", "output", "first", "last", "width", "height", "max_dimension", "levels", "sea_raft_root")
    missing = [key for key in required if key not in job]
    if missing:
        raise ValueError(f"Missing job fields: {', '.join(missing)}")
    first, last = int(job["first"]), int(job["last"])
    width, height = int(job["width"]), int(job["height"])
    levels = tuple(sorted(set(int(value) for value in job["levels"])))
    if first > last or not levels or any(value not in LEVELS for value in levels):
        raise ValueError("Invalid frame range or temporal levels")
    frame_path(job["input"], first)
    frame_path(job["output"], first)
    size = inference_size(width, height, int(job["max_dimension"]))
    return {**job, "first": first, "last": last, "width": width, "height": height,
            "levels": list(levels), "max_dimension": int(job["max_dimension"]),
            "inference_resolution": list(size), "device": job.get("device", "cuda"),
            "preprocess": job.get("preprocess", "clamp"), "invert_v": bool(job.get("invert_v", False)),
            "model": job.get("model", MODEL), "config": job.get("config", CONFIG)}


def frame_pairs(frame: int, first: int, last: int, levels: list[int]):
    """Yield (level, direction, target) only at anchored frames and valid clip edges."""
    for level in levels:
        if (frame - first) % level:
            continue
        if frame - level >= first:
            yield level, "p", frame - level
        if frame + level <= last:
            yield level, "n", frame + level


def cache_identity(job: dict) -> dict:
    keys = ("first", "last", "width", "height", "levels", "max_dimension", "inference_resolution",
            "device", "preprocess", "invert_v", "model", "config")
    identity = {key: job[key] for key in keys}
    identity["schema"] = SCHEMA
    identity["input"] = str(Path(job["input"]).resolve())
    identity["input_digest"] = job.get("input_digest", "")
    return identity


def fingerprint(job: dict) -> str:
    data = json.dumps(cache_identity(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
