from datasets import load_dataset

dataset = load_dataset(
    "SWE-bench/SWE-bench_Verified",
    split="test",
)

dataset.save_to_disk(
    "data/swebench_verified"
)

dataset.to_json(
    "data/swebench_verified.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

print(dataset)
print(dataset[0]["instance_id"])