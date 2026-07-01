from __future__ import annotations

import tiktoken


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def fit_text_to_tokens(text: str, target: int, model: str) -> str:
    """Repeat a source context and truncate it to an exact tokenizer token budget."""
    if target <= 0:
        raise ValueError("target must be positive")
    encoding = encoding_for_model(model)
    tokens = encoding.encode(text)
    if not tokens:
        raise ValueError("cannot fit an empty context")
    if len(tokens) >= target:
        return encoding.decode(tokens[:target])
    repeated: list[int] = []
    separator = encoding.encode("\n\n--- 共享背景的后续材料 ---\n\n")
    while len(repeated) < target:
        if repeated:
            repeated.extend(separator)
        repeated.extend(tokens)
    return encoding.decode(repeated[:target])
