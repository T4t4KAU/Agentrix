from tokens import count_tokens, fit_text_to_tokens


def test_fit_text_exact_token_budget() -> None:
    fitted = fit_text_to_tokens("hello world", 100, "unknown-future-model")
    assert count_tokens(fitted, "unknown-future-model") == 100
