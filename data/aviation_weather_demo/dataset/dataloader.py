from pathlib import Path
from torch.utils.data import DataLoader
from dataset.radar_dataset import RadarSequenceDataset

def create_dataloader(
    manifest_path: str | Path,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool | None = None,
):
    dataset = RadarSequenceDataset(manifest_path, split=split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
