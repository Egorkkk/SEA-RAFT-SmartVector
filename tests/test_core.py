import pytest

from smartvector.core import CHANNELS, LEVELS, frame_pairs, frame_path, inference_size, layer, validate_job


def test_sparse_grid_and_clip_edges():
    levels = list(LEVELS)
    assert list(frame_pairs(1001, 1001, 1009, levels)) == [(1, "n", 1002), (2, "n", 1003),
                                                     (4, "n", 1005), (8, "n", 1009)]
    assert list(frame_pairs(1002, 1001, 1009, levels)) == [(1, "p", 1001), (1, "n", 1003)]
    assert list(frame_pairs(1005, 1001, 1009, levels)) == [(1, "p", 1004), (1, "n", 1006),
                                                     (2, "p", 1003), (2, "n", 1007),
                                                     (4, "p", 1001), (4, "n", 1009)]


def test_resolution_and_channel_names():
    assert inference_size(3840, 2160, 1280) == (1280, 720)
    assert inference_size(1920, 1080, 0) == (1920, 1080)
    assert inference_size(800, 600, 1440) == (800, 600)
    assert len([f"{layer(level)}.{channel}" for level in LEVELS for channel in CHANNELS]) == 28
    assert layer(4) == "smartvector_f04_v01"


def test_job_validation():
    job = {"input": "in.####.exr", "output": "out.####.exr", "first": 1001, "last": 1010,
           "width": 3840, "height": 2160, "max_dimension": 1280, "levels": [4, 1, 4],
           "sea_raft_root": "/raft"}
    result = validate_job(job)
    assert result["levels"] == [1, 4]
    assert result["inference_resolution"] == [1280, 720]
    assert frame_path(result["output"], 1001).name == "out.1001.exr"
    with pytest.raises(ValueError):
        validate_job({**job, "levels": [3]})
