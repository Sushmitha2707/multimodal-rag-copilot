import pytest

pytest.importorskip("langchain_core")

from copilot.ingest import group_segments  # noqa: E402


def test_group_segments_windows():
    segs = [{"timestamp": (i * 10.0, i * 10.0 + 10.0), "text": f"s{i}"} for i in range(7)]  # 70 s total
    groups = group_segments(segs, window_s=30.0)
    assert [g["text"] for g in groups] == ["s0 s1 s2", "s3 s4 s5", "s6"]
    assert groups[0]["start"] == 0.0 and groups[0]["end"] == 30.0


def test_group_segments_handles_missing_timestamps():
    groups = group_segments([{"timestamp": (None, None), "text": "hello"}])
    assert groups == [{"start": 0.0, "end": 0.0, "text": "hello"}]
