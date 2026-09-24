"""Optional GPU acceptance check: a four-pixel horizontal image translation."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from smartvector.core import frame_path
from smartvector.worker import _dependencies, _write_vectors, run


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / "smartvector" / "install_config.json"
    if not config_path.is_file():
        raise RuntimeError("Run scripts/install_windows.ps1 before this check")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    os.environ["HF_HOME"] = config["hf_home"]
    os.environ["TORCH_HOME"] = config["torch_home"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    np, _, _, OpenEXR, Imath = _dependencies()
    from scipy.ndimage import gaussian_filter

    random = np.random.default_rng(42)
    image = gaussian_filter(random.random((128, 128, 3), dtype=np.float32), sigma=(1.5, 1.5, 0))
    image = (image - image.min()) / (image.max() - image.min())

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        input_pattern = str(directory / "input.####.exr")
        output_pattern = str(directory / "vectors.####.exr")
        for frame, pixels in ((1, image), (2, np.roll(image, 4, axis=1))):
            _write_vectors(frame_path(input_pattern, frame),
                           {channel: pixels[:, :, index] for index, channel in enumerate("RGB")},
                           128, 128, np, OpenEXR, Imath)
        run({"input": input_pattern, "output": output_pattern, "first": 1, "last": 2,
             "width": 128, "height": 128, "max_dimension": 128, "levels": [1],
             "sea_raft_root": config["sea_raft_root"], "device": "cuda"})
        for frame, channel, expected in ((1, "n_u", 4.0), (2, "p_u", -4.0)):
            exr = OpenEXR.InputFile(str(frame_path(output_pattern, frame)))
            try:
                values = np.frombuffer(
                    exr.channel(f"smartvector_f01_v01.{channel}",
                                Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32
                ).reshape(128, 128)
            finally:
                exr.close()
            median = float(np.median(values[24:104, 24:104]))
            print(f"Frame {frame} {channel}: median {median:.3f} px (expected {expected:+.1f})")
            if abs(median - expected) > 1.0:
                raise AssertionError(f"Incorrect flow direction or magnitude: {channel} = {median}")
    print("MOTION_CHECK_OK")


if __name__ == "__main__":
    main()
