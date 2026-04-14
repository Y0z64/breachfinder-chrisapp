from pathlib import Path

import numpy as np

from animation_utils import build_detection_animation_frames, save_detection_animation


def test_detection_animation_frames_are_square():
    seg_slice = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 42, 42],
            [0, 1, 161, 0, 42, 160],
            [0, 1, 1, 0, 42, 42],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    frames = build_detection_animation_frames(seg_slice)

    assert frames
    assert all(frame.width == frame.height for frame in frames)


def test_save_detection_animation(tmp_path: Path):
    seg_slice = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 161, 0],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )

    out_path = save_detection_animation(seg_slice, tmp_path / "breach-animation")

    assert out_path.suffix == ".gif"
    assert out_path.exists()
    assert out_path.stat().st_size > 0
