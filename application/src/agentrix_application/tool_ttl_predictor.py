"""Small dependency-free online predictor for tool-call KV retention TTLs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_TTL_HORIZONS_MS = (100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)


@dataclass(frozen=True)
class ToolTTLContext:
    """Privacy-preserving features available when a tool call starts."""

    tool_family: str = "unknown"
    argument_bytes: int = 0
    kv_tokens: int = 0
    pressure: float | None = None
    active_tool_sessions: int = 0
    shared_prefix_ratio: float | None = None
    timeout_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.tool_family:
            raise ValueError("tool_family must not be empty")
        for name in ("argument_bytes", "kv_tokens", "active_tool_sessions"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.pressure is not None and (
            not math.isfinite(self.pressure) or not 0 <= self.pressure <= 1
        ):
            raise ValueError("pressure must be finite and in [0, 1]")
        if self.shared_prefix_ratio is not None and (
            not math.isfinite(self.shared_prefix_ratio)
            or not 0 <= self.shared_prefix_ratio <= 1
        ):
            raise ValueError("shared_prefix_ratio must be finite and in [0, 1]")
        if self.timeout_ms is not None and (
            not math.isfinite(self.timeout_ms) or self.timeout_ms < 0
        ):
            raise ValueError("timeout_ms must be finite and non-negative")


@dataclass(frozen=True)
class TTLPrediction:
    """One TTL recommendation and its survival probabilities."""

    ttl_ms: float
    survival_probabilities: tuple[float, ...]
    trained_samples: int
    used_fallback: bool


class ToolTTLPredictor(Protocol):
    """Interface consumed by the KV trimmer."""

    def predict(self, context: ToolTTLContext) -> TTLPrediction: ...

    def observe(self, context: ToolTTLContext, duration_ms: float) -> None: ...


class OnlineHorizonTTLPredictor:
    """Online logistic survival model over a small set of TTL horizons.

    One logistic classifier predicts ``P(duration > horizon)`` for every
    horizon. Feature hashing keeps the state bounded and avoids storing raw
    tool arguments. Independent probabilities are projected to a monotonic
    survival curve before a TTL is selected.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        horizons_ms: tuple[float, ...] = DEFAULT_TTL_HORIZONS_MS,
        feature_dimension: int = 128,
        learning_rate: float = 0.12,
        l2: float = 1e-4,
        confidence: float = 0.7,
        fallback_ttl_ms: float = 500.0,
        min_ttl_ms: float = 100.0,
        min_training_samples: int = 50,
    ) -> None:
        if not horizons_ms or any(
            not math.isfinite(value) or value <= 0 for value in horizons_ms
        ):
            raise ValueError("horizons_ms must contain positive finite values")
        if tuple(sorted(set(horizons_ms))) != horizons_ms:
            raise ValueError("horizons_ms must be strictly increasing")
        if feature_dimension < 8:
            raise ValueError("feature_dimension must be at least 8")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be non-negative and finite")
        if not 0.5 <= confidence < 1:
            raise ValueError("confidence must be in [0.5, 1)")
        if min_training_samples < 0:
            raise ValueError("min_training_samples must be non-negative")
        if not 0 < min_ttl_ms <= fallback_ttl_ms:
            raise ValueError("TTL bounds must satisfy 0 < min <= fallback")

        self.horizons_ms = horizons_ms
        self.feature_dimension = feature_dimension
        self.learning_rate = learning_rate
        self.l2 = l2
        self.confidence = confidence
        self.fallback_ttl_ms = fallback_ttl_ms
        self.min_ttl_ms = min_ttl_ms
        self.min_training_samples = min_training_samples
        self.sample_count = 0
        self.positive_counts = [0] * len(horizons_ms)
        self.weights = [[0.0] * feature_dimension for _ in range(len(horizons_ms))]

    def predict(self, context: ToolTTLContext) -> TTLPrediction:
        """Predict a bounded soft TTL for one tool call."""

        features = self._features(context)
        probabilities = self._survival_probabilities(features)
        if self.sample_count < self.min_training_samples:
            return TTLPrediction(
                ttl_ms=self.fallback_ttl_ms,
                survival_probabilities=probabilities,
                trained_samples=self.sample_count,
                used_fallback=True,
            )

        confidently_survived = [
            horizon
            for horizon, probability in zip(
                self.horizons_ms, probabilities, strict=True
            )
            if probability >= self.confidence
        ]
        if confidently_survived:
            longest_horizon = max(confidently_survived)
            # A tool predicted to live longer should become eligible sooner.
            # The inverse mapping is bounded by the fixed fallback TTL, so the
            # model can only make retention more aggressive, never less safe.
            ttl_ms = self.fallback_ttl_ms * self.min_ttl_ms / longest_horizon
            ttl_ms = max(self.min_ttl_ms, min(self.fallback_ttl_ms, ttl_ms))
        else:
            ttl_ms = self.fallback_ttl_ms
        return TTLPrediction(
            ttl_ms=ttl_ms,
            survival_probabilities=probabilities,
            trained_samples=self.sample_count,
            used_fallback=False,
        )

    def observe(self, context: ToolTTLContext, duration_ms: float) -> None:
        """Update every horizon classifier from one completed tool call."""

        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        features = self._features(context)
        rate = self.learning_rate / math.sqrt(1 + self.sample_count / 1000)
        for index, horizon in enumerate(self.horizons_ms):
            label = 1.0 if duration_ms > horizon else 0.0
            if label:
                self.positive_counts[index] += 1
            weights = self.weights[index]
            probability = self._sigmoid(self._dot(weights, features))
            error = probability - label
            for feature_index, value in features.items():
                gradient = error * value + self.l2 * weights[feature_index]
                weights[feature_index] = max(
                    -20.0,
                    min(20.0, weights[feature_index] - rate * gradient),
                )
        self.sample_count += 1

    def save(self, path: Path) -> None:
        """Atomically persist model state as versioned JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), separators=(",", ":")))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "OnlineHorizonTTLPredictor":
        """Load a predictor previously written by :meth:`save`."""

        state = json.loads(path.read_text())
        if not isinstance(state, dict) or state.get("version") != cls.STATE_VERSION:
            raise ValueError("unsupported TTL predictor state")
        config = state.get("config")
        if not isinstance(config, dict):
            raise ValueError("TTL predictor state has no config")
        predictor = cls(
            horizons_ms=tuple(float(value) for value in config["horizons_ms"]),
            feature_dimension=int(config["feature_dimension"]),
            learning_rate=float(config["learning_rate"]),
            l2=float(config["l2"]),
            confidence=float(config["confidence"]),
            fallback_ttl_ms=float(config["fallback_ttl_ms"]),
            min_ttl_ms=float(config["min_ttl_ms"]),
            min_training_samples=int(config["min_training_samples"]),
        )
        weights = state.get("weights")
        positives = state.get("positive_counts")
        if not isinstance(weights, list) or not isinstance(positives, list):
            raise ValueError("TTL predictor state is incomplete")
        if len(weights) != len(predictor.horizons_ms) or any(
            not isinstance(row, list) or len(row) != predictor.feature_dimension
            for row in weights
        ):
            raise ValueError("TTL predictor weight shape does not match config")
        if len(positives) != len(predictor.horizons_ms):
            raise ValueError("TTL predictor count shape does not match config")
        predictor.weights = [[float(value) for value in row] for row in weights]
        predictor.positive_counts = [int(value) for value in positives]
        predictor.sample_count = int(state.get("sample_count", 0))
        return predictor

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable model snapshot."""

        return {
            "version": self.STATE_VERSION,
            "config": {
                "horizons_ms": list(self.horizons_ms),
                "feature_dimension": self.feature_dimension,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "confidence": self.confidence,
                "fallback_ttl_ms": self.fallback_ttl_ms,
                "min_ttl_ms": self.min_ttl_ms,
                "min_training_samples": self.min_training_samples,
            },
            "sample_count": self.sample_count,
            "positive_counts": self.positive_counts,
            "weights": self.weights,
        }

    def context_dict(self, context: ToolTTLContext) -> dict[str, Any]:
        """Expose the privacy-preserving context for shadow event logs."""

        return asdict(context)

    def _survival_probabilities(self, features: dict[int, float]) -> tuple[float, ...]:
        probabilities = []
        previous = 1.0
        for index, weights in enumerate(self.weights):
            if self.sample_count < 8:
                probability = (self.positive_counts[index] + 1) / (
                    self.sample_count + 2
                )
            else:
                probability = self._sigmoid(self._dot(weights, features))
            probability = min(previous, max(0.0, min(1.0, probability)))
            probabilities.append(probability)
            previous = probability
        return tuple(probabilities)

    def _features(self, context: ToolTTLContext) -> dict[int, float]:
        features = {0: 1.0}
        self._hashed_feature(features, f"tool={context.tool_family}", 1.0)
        self._hashed_feature(
            features,
            f"tool_arg_bucket={context.tool_family}:{self._log_bucket(context.argument_bytes)}",
            1.0,
        )
        features[1] = min(1.0, math.log1p(context.argument_bytes) / 16)
        features[2] = min(1.0, math.log1p(context.kv_tokens) / 16)
        features[3] = min(1.0, math.log1p(context.active_tool_sessions) / 5)
        if context.pressure is not None:
            features[4] = context.pressure
        if context.shared_prefix_ratio is not None:
            features[5] = context.shared_prefix_ratio
        if context.timeout_ms is not None:
            features[6] = min(1.0, math.log1p(context.timeout_ms) / 16)
        return features

    def _hashed_feature(
        self, features: dict[int, float], value: str, magnitude: float
    ) -> None:
        digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
        encoded = int.from_bytes(digest, "little")
        index = 7 + encoded % (self.feature_dimension - 7)
        sign = -1.0 if encoded >> 63 else 1.0
        features[index] = features.get(index, 0.0) + sign * magnitude

    @staticmethod
    def _log_bucket(value: int) -> int:
        return 0 if value <= 0 else int(math.log2(value))

    @staticmethod
    def _dot(weights: list[float], features: dict[int, float]) -> float:
        return sum(weights[index] * value for index, value in features.items())

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            inverse = math.exp(-min(value, 60))
            return 1 / (1 + inverse)
        exponent = math.exp(max(value, -60))
        return exponent / (1 + exponent)
