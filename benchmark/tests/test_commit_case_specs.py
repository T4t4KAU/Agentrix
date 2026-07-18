from commit_case_specs import is_safe_commit


def test_safe_commit_filter_accepts_ordinary_functional_change() -> None:
    assert is_safe_commit(
        "Add configurable tile support",
        ["libavcodec/encoder.c", "tests/encoder_test.c"],
    )


def test_safe_commit_filter_rejects_security_and_crash_topics() -> None:
    assert not is_safe_commit("Fix security issue in parser", ["parser.c"])
    assert not is_safe_commit("Avoid crash on malformed input", ["decoder.c"])
    assert not is_safe_commit("Refine parser", ["tests/fuzz/parser.c"])
    assert not is_safe_commit("Avoid a buffer overread", ["session.c"])
    assert not is_safe_commit("Update password hasher", ["auth/hashers.py"])
