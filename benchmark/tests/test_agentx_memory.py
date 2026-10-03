"""Resource accounting must aggregate ranks without double-counting exports."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "agentx_memory",
    Path(__file__).resolve().parents[1] / "scripts/summarize_agentx_memory.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_counter_aggregation_ignores_histograms_and_creation_timestamps():
    text = '''# TYPE vllm:kv_offload_total_bytes_total counter
vllm:kv_offload_total_bytes_total{engine="0",transfer_type="GPU_to_CPU"} 100
vllm:kv_offload_total_bytes_total{engine="1",transfer_type="GPU_to_CPU"} 200
vllm:kv_offload_total_bytes_total{engine="1",transfer_type="CPU_to_GPU"} 50
vllm:kv_offload_total_bytes_created{engine="1",transfer_type="CPU_to_GPU"} 12345
vllm:kv_offload_size_sum{engine="1",transfer_type="CPU_to_GPU"} 50
vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 12
vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 24
vllm:prompt_tokens_by_source_total{engine="1",source="external_kv_transfer"} 8
vllm:num_preemptions_total{engine="0"} 1
vllm:num_preemptions_total{engine="1"} 0
'''
    assert module.counters(text) == {
        "GPU_to_CPU": 300,
        "CPU_to_GPU": 50,
        "local_compute": 36,
        "external_kv_transfer": 8,
        "preemptions": 1,
    }
