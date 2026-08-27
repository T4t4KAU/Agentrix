# Qwen2.5-VL E/PD heterogeneity experiments

This directory contains the reproducible inputs and drivers for the two-H20
encoder/prefill-decode experiments. Large datasets and generated results stay
outside the Git repository.

## Dataset preparation

Download the compact COCO 2017 validation repository from ModelScope:

```bash
modelscope download \
  --dataset modelscope/coco2017val \
  --local_dir "${DATASET_DIR:?Set DATASET_DIR}"
```

Generate deterministic, 28-pixel-aligned image variants:

```bash
cd benchmark
uv run --extra data python epd_heterogeneity/prepare_coco.py \
  --archive "${DATASET_DIR:?Set DATASET_DIR}/coco2017val.zip" \
  --output "${CONTROL_DIR:?Set CONTROL_DIR}" \
  --samples 96
```

The four buckets are designed around the processed tensor shape, not the
source image resolution. Their expected Qwen2.5-VL token counts are recorded
in `images.jsonl` and must be checked against the processor output before a
formal run.

## Experimental order

1. A0: compare monolithic and 1E+1PD output correctness and calibrate clocks.
2. A1: run the small encoder, prefill, and decode profiling matrix.
3. B: run sparse and burst traces only after stage timing is decomposed.
4. C: replay VisionArena-Chat and real Agentrix trajectories.

Do not interpret proxy latency as encoder compute time. The reference encoder
cache connector also includes D2H, serialization, shared-storage, loading, and
H2D costs. Keep these intervals separate in analysis.
