import pytest

from copilot import feedback


def test_log_and_feedback_roundtrip(tmp_path):
    db = str(tmp_path / "fb.sqlite")
    iid = feedback.log_interaction("How many leave days?", "It is 30.", "[1] ctx", [], "gemini", db_path=db)
    assert feedback.set_feedback(iid, -1, "It is 24 days [1].", db_path=db)
    pairs = feedback.build_preference_pairs(db_path=db)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "It is 24 days [1]."
    assert pairs[0]["rejected"] == "It is 30."
    assert "How many leave days?" in pairs[0]["prompt"] and "[1] ctx" in pairs[0]["prompt"]


def test_thumbs_down_without_correction_is_not_a_pair(tmp_path):
    db = str(tmp_path / "fb.sqlite")
    iid = feedback.log_interaction("q?", "a", "ctx", db_path=db)
    feedback.set_feedback(iid, -1, db_path=db)
    assert feedback.build_preference_pairs(db_path=db) == []


def test_thumbs_up_is_not_a_pair(tmp_path):
    db = str(tmp_path / "fb.sqlite")
    iid = feedback.log_interaction("q?", "a", "ctx", db_path=db)
    feedback.set_feedback(iid, 1, db_path=db)
    assert feedback.build_preference_pairs(db_path=db) == []


def test_invalid_rating_and_unknown_id(tmp_path):
    db = str(tmp_path / "fb.sqlite")
    with pytest.raises(ValueError):
        feedback.set_feedback(1, 5, db_path=db)
    assert feedback.set_feedback(999, 1, db_path=db) is False
