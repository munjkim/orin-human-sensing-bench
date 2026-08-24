"""v4l2-ctl format parsing.

The fixture is the shape of a typical cheap UVC webcam, and it encodes the
single most important fact about one: the same 1280x720 frame is available
at 30 fps compressed and 10 fps raw, because USB 2.0 cannot carry raw
YUYV at that size. A benchmark that ignores this measures the bus.
"""

from ohsb.camera_probe import best_mode, format_modes, parse_formats

V4L2_OUTPUT = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.100s (10.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""


def test_parses_every_format_size_fps_combination():
    modes = parse_formats(V4L2_OUTPUT)
    assert len(modes) == 5
    assert {m["fourcc"] for m in modes} == {"MJPG", "YUYV"}


def test_compressed_flag_tracks_the_pixel_format():
    modes = parse_formats(V4L2_OUTPUT)
    assert all(m["compressed"] for m in modes if m["fourcc"] == "MJPG")
    assert not any(m["compressed"] for m in modes if m["fourcc"] == "YUYV")


def test_fps_is_a_property_of_format_and_size_together():
    modes = parse_formats(V4L2_OUTPUT)
    mjpg_720 = [m for m in modes if m["fourcc"] == "MJPG" and m["width"] == 1280]
    yuyv_720 = [m for m in modes if m["fourcc"] == "YUYV" and m["width"] == 1280]
    # The USB bandwidth wall: same resolution, 3x the frame rate compressed.
    assert mjpg_720[0]["fps"] == 30.0
    assert yuyv_720[0]["fps"] == 10.0


def test_best_mode_picks_the_fastest_at_a_resolution():
    modes = parse_formats(V4L2_OUTPUT)
    best = best_mode(modes, 1280, 720)
    assert best["fourcc"] == "MJPG"
    assert best["fps"] == 30.0


def test_best_mode_returns_none_for_unsupported_resolution():
    assert best_mode(parse_formats(V4L2_OUTPUT), 1920, 1080) is None


def test_parser_tolerates_unrelated_output():
    assert parse_formats("VIDIOC_ENUM_FMT: failed: Inappropriate ioctl") == []


def test_format_modes_renders_without_a_device_name():
    text = format_modes({"device": "/dev/video0", "source": "v4l2-ctl",
                         "modes": parse_formats(V4L2_OUTPUT)})
    assert "/dev/video0" in text
    assert "MJPG" in text
    assert "compressed" in text
