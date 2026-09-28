import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import PROCESSED_DATA_ROOT


class BrainTumorDataset(Dataset):
    """Patch-level dataset backed by the preprocessing output."""

    def __init__(self, processed_root=PROCESSED_DATA_ROOT, split="train"):
        self.processed_root = processed_root
        self.split = split
        self.samples = []

        split_names = [split] if split not in (None, "all") else ["train", "val", "test"]

        for split_name in split_names:
            split_dir = os.path.join(processed_root, split_name)
            if not os.path.isdir(split_dir):
                continue

            for patient_id in sorted(os.listdir(split_dir)):
                patient_dir = os.path.join(split_dir, patient_id)
                image_path = os.path.join(patient_dir, "images.npy")
                mask_path = os.path.join(patient_dir, "masks.npy")
                metadata_path = os.path.join(patient_dir, "metadata.json")

                if not all(os.path.exists(p) for p in (image_path, mask_path, metadata_path)):
                    continue

                images = np.load(image_path, mmap_mode="r")
                masks = np.load(mask_path, mmap_mode="r")
                if images.ndim != 5 or images.shape[1:] != (4, 64, 64, 64):
                    raise ValueError(f"Invalid images.npy shape for {patient_id}: {images.shape}")
                if masks.shape != (images.shape[0], 64, 64, 64):
                    raise ValueError(f"Invalid masks.npy shape for {patient_id}: {masks.shape}")

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if len(metadata.get("patch_coordinates", [])) != images.shape[0]:
                    raise ValueError(f"Patch-coordinate count mismatch for {patient_id}")

                for patch_index in range(images.shape[0]):
                    self.samples.append((image_path, mask_path, patch_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, patch_index = self.samples[index]
        images = np.load(image_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(images[patch_index], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(masks[patch_index], dtype=np.int64))
        return image, mask


def create_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
