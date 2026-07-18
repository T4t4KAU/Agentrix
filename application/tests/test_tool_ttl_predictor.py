from pathlib import Path

from agentrix_application import OnlineHorizonTTLPredictor, ToolTTLContext


def test_predictor_learns_short_and_long_tool_families() -> None:
    predictor = OnlineHorizonTTLPredictor(min_training_samples=20)
    short = ToolTTLContext(tool_family="read", argument_bytes=200)
    long = ToolTTLContext(tool_family="public_test", argument_bytes=200)

    for _ in range(150):
        predictor.observe(short, 40)
        predictor.observe(long, 4000)

    short_prediction = predictor.predict(short)
    long_prediction = predictor.predict(long)
    assert short_prediction.ttl_ms == predictor.fallback_ttl_ms
    assert long_prediction.ttl_ms == predictor.min_ttl_ms
    assert short_prediction.survival_probabilities[0] < 0.3
    assert long_prediction.survival_probabilities[3] > 0.7


def test_survival_curve_is_monotonic() -> None:
    predictor = OnlineHorizonTTLPredictor(min_training_samples=0)
    context = ToolTTLContext(tool_family="search")
    for duration in (50, 120, 300, 800, 2500, 7000) * 20:
        predictor.observe(context, duration)

    probabilities = predictor.predict(context).survival_probabilities
    assert all(left >= right for left, right in zip(probabilities, probabilities[1:]))


def test_cold_start_uses_fixed_fallback() -> None:
    predictor = OnlineHorizonTTLPredictor(min_training_samples=10)
    prediction = predictor.predict(ToolTTLContext(tool_family="unknown"))
    assert prediction.used_fallback
    assert prediction.ttl_ms == 500


def test_model_state_round_trip(tmp_path: Path) -> None:
    predictor = OnlineHorizonTTLPredictor(min_training_samples=0)
    context = ToolTTLContext(
        tool_family="test",
        argument_bytes=512,
        kv_tokens=8192,
        pressure=0.8,
        active_tool_sessions=4,
        shared_prefix_ratio=0.75,
        timeout_ms=30_000,
    )
    for _ in range(30):
        predictor.observe(context, 2500)
    expected = predictor.predict(context)
    path = tmp_path / "ttl-model.json"
    predictor.save(path)

    restored = OnlineHorizonTTLPredictor.load(path)
    assert restored.state_dict() == predictor.state_dict()
    assert restored.predict(context) == expected


def test_context_rejects_raw_invalid_numeric_features() -> None:
    try:
        ToolTTLContext(tool_family="test", pressure=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN pressure must be rejected")
