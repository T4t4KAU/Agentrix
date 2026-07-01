from datasets import load_dataset

dataset = load_dataset(
    "GAIR/AgencyBench",
    "V2",
    split="train",
)

dataset.save_to_disk(
    "data/agencybench_v2"
)

dataset.to_json(
    "data/agencybench_v2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

print(dataset)
print(dataset[0])