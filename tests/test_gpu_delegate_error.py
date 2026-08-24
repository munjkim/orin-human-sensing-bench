"""The GPU delegate error message, clarified.

Confirmed on the board: `pip install mediapipe==0.10.9` (the pinned version
for this project's Python 3.8 target) is compiled with GPU support removed
entirely — even the OpenGL ES delegate, not just CUDA. It fails with a bare
NotImplementedError naming an internal calculator, which explains nothing.
create_from_options() turns that into an actionable message instead.
"""

import pytest

from ohsb.config import TaskConfig
from ohsb.tasks.mediapipe_common import create_from_options


class _FakeFactory:
    def __init__(self, message):
        self._message = message

    def create_from_options(self, options):
        raise NotImplementedError(self._message)


def test_gpu_build_flag_error_is_clarified():
    factory = _FakeFactory(
        "ValidatedGraphConfig Initialization failed.\n"
        "ImageCloneCalculator: GPU processing is disabled in build flags"
    )
    cfg = TaskConfig(delegate="gpu")
    with pytest.raises(NotImplementedError, match="compiled out entirely"):
        create_from_options(factory, options=None, cfg=cfg)


def test_unrelated_not_implemented_error_passes_through_unchanged():
    factory = _FakeFactory("some other internal MediaPipe failure")
    cfg = TaskConfig(delegate="gpu")
    with pytest.raises(NotImplementedError, match="some other internal MediaPipe failure"):
        create_from_options(factory, options=None, cfg=cfg)


def test_cpu_delegate_error_is_not_reinterpreted_as_a_gpu_problem():
    # Same message, but delegate=cpu means the GPU-specific rewrite must not
    # fire — misattributing a CPU-path failure to the GPU build flag would
    # send someone chasing the wrong cause.
    factory = _FakeFactory("ImageCloneCalculator: GPU processing is disabled in build flags")
    cfg = TaskConfig(delegate="cpu")
    with pytest.raises(NotImplementedError, match="GPU processing is disabled"):
        create_from_options(factory, options=None, cfg=cfg)


def test_success_path_is_unaffected():
    class _Ok:
        def create_from_options(self, options):
            return "the-created-object"

    assert create_from_options(_Ok(), options=None, cfg=TaskConfig()) == "the-created-object"
