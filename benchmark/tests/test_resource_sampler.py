from io import BytesIO

import resource_sampler


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_sample_kv_supports_current_metric_and_engine_order(monkeypatch) -> None:
    payload = b"\n".join(
        (
            b'vllm:kv_cache_usage_perc{engine="1"} 0.75',
            b'vllm:kv_cache_usage_perc{engine="0"} 0.5',
        )
    )
    monkeypatch.setattr(
        resource_sampler.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    assert resource_sampler.sample_kv("http://metrics") == [0.5, 0.75]
