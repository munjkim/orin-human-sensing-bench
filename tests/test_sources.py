import numpy as np
import pytest

from ohsb.config import SourceConfig
from ohsb.sources import available_sources, build_source


def test_synthetic_is_deterministic_and_correctly_shaped():
    cfg = SourceConfig(type="synthetic", width=96, height=64, count=5, seed=7)
    a = build_source(cfg).prepare().frames
    b = build_source(cfg).prepare().frames
    assert len(a) == 5
    assert a[0].image.shape == (64, 96, 3)
    assert a[0].image.dtype == np.uint8
    # Same seed must give identical frames, or repeat runs are not comparable.
    assert np.array_equal(a[0].image, b[0].image)


def test_frames_are_contiguous():
    # MediaPipe wraps the buffer without copying; a non-contiguous array here
    # is a crash on the board, not a slowdown.
    frames = build_source(SourceConfig(type="synthetic", count=2)).prepare().frames
    assert frames[0].image.flags["C_CONTIGUOUS"]


def test_unknown_source_names_the_alternatives():
    with pytest.raises(ValueError, match="image_dir"):
        build_source(SourceConfig(type="webcam"))


def test_registry_lists_all_three():
    assert available_sources() == ["image_dir", "synthetic", "video"]
