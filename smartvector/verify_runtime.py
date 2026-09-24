"""One-time native Windows runtime check, including model load and CUDA inference."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .core import CHANNELS, CONFIG, LEVELS, MODEL, frame_path, layer
from .worker import _dependencies, _read_rgb, _write_vectors, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sea-raft-root", required=True)
    args = parser.parse_args()
    np, torch, _, OpenEXR, Imath = _dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.exr"
        _write_vectors(path, {"R": np.ones((2, 2), np.float32),
                              "G": np.zeros((2, 2), np.float32),
                              "B": np.zeros((2, 2), np.float32)}, 2, 2, np, OpenEXR, Imath)
        image = _read_rgb(path, 2, 2, np, OpenEXR, Imath)
        if image.shape != (2, 2, 3) or image[0, 0, 0] != 1:
            raise RuntimeError("OpenEXR round-trip failed")
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        input_pattern = str(directory / "input.####.exr")
        output_pattern = str(directory / "vectors.####.exr")
        gradient = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (128, 1))
        for frame, offset in ((1, 0), (2, 4)):
            red = np.roll(gradient, offset, axis=1)
            _write_vectors(frame_path(input_pattern, frame),
                           {"R": red, "G": gradient, "B": red}, 128, 128, np, OpenEXR, Imath)
        run({"input": input_pattern, "output": output_pattern, "first": 1, "last": 2,
             "width": 128, "height": 128, "max_dimension": 128,
             "levels": list(LEVELS), "sea_raft_root": args.sea_raft_root,
             "config": CONFIG, "model": MODEL, "device": "cuda"})
        output = OpenEXR.InputFile(str(frame_path(output_pattern, 1)))
        try:
            names = set(output.header()["channels"])
            expected = {f"{layer(level)}.{channel}" for level in LEVELS for channel in CHANNELS}
            if names != expected:
                raise RuntimeError(f"Unexpected SmartVector channels: {names ^ expected}")
            zero = np.frombuffer(output.channel(f"{layer(1)}.p_u", Imath.PixelType(Imath.PixelType.FLOAT)),
                                 dtype=np.float32)
            if not np.all(zero == 0):
                raise RuntimeError("First-frame backward vectors are not zero")
        finally:
            output.close()
    print(f"Runtime verified: torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, "
          f"OpenEXR and two-frame SmartVector bake OK")


if __name__ == "__main__":
    main()
