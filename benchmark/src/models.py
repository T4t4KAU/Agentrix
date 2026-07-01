from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BranchTrace:
    branch_id: int
    suffix_tokens: int
    decode_tokens: int
    input_tokens: int | None = None
    latency_ms: float | None = None
    strategy: str | None = None

    def __post_init__(self) -> None:
        if self.branch_id < 0 or self.suffix_tokens < 0 or self.decode_tokens < 0:
            raise ValueError("branch_id and token counts must be non-negative")


@dataclass(frozen=True)
class BenchmarkTrace:
    case_id: str
    prefix_tokens: int
    branches: list[BranchTrace]
    suffix_distribution: str = "observed"
    output_tokens: int = 0
    arrival_mode: str = "simultaneous"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if not self.branches:
            raise ValueError("a trace must contain at least one branch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
