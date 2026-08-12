from dataset.dataloader import create_dataloader

MANIFEST = "reports/sample_manifest_dataset_1.csv"

loader = create_dataloader(
    MANIFEST,
    split="train",
    batch_size=4,
    num_workers=0,
)

inputs, targets, metadata = next(iter(loader))

print("inputs :", tuple(inputs.shape), inputs.dtype, float(inputs.min()), float(inputs.max()))
print("targets:", tuple(targets.shape), targets.dtype, float(targets.min()), float(targets.max()))
print("metadata keys:", metadata.keys())
