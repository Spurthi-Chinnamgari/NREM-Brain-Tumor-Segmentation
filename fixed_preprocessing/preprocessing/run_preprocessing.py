import argparse
import gc
import json
import os
import shutil

import numpy as np

from ..config import (
    DATASET_ROOT,
    PROCESSED_DATA_ROOT,
    SPLIT_SEED,
    DEFAULT_PATIENT_PERCENTAGE,
    PATCH_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from .data_loader import get_patient_ids, load_patient
from .normalization import normalize_patient
from .cropping import crop_patient
from .stacking import stack_modalities
from .patch_extraction import extract_patches


def save_split(patient_ids, split_name, split_root):
    split_path = os.path.join(split_root, f"{split_name}.txt")
    with open(split_path, "w", encoding="utf-8") as file:
        file.write("\n".join(patient_ids) + ("\n" if patient_ids else ""))


def build_patient_split(
    patient_ids,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=SPLIT_SEED,
):
    if not patient_ids:
        return [], [], []
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    rng = np.random.default_rng(seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    test_count = total - train_count - val_count

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count:train_count + val_count]
    test_ids = shuffled[train_count + val_count:]
    return train_ids, val_ids, test_ids


def prepare_processed_root(processed_root):
    os.makedirs(processed_root, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_dir = os.path.join(processed_root, split_name)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)


def process_patient(patient_id, split_name):
    patient_dir = os.path.join(DATASET_ROOT, patient_id)

    patient = load_patient(patient_dir, patient_id)
    patient = normalize_patient(patient)
    patient = crop_patient(patient)

    stacked = stack_modalities(patient)
    image_patches, mask_patches, patch_coordinates, patch_info = extract_patches(
        stacked,
        patient["segmentation"],
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    save_dir = os.path.join(PROCESSED_DATA_ROOT, split_name, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "images.npy"), image_patches)
    np.save(os.path.join(save_dir, "masks.npy"), mask_patches)
    np.save(os.path.join(save_dir, "patch_coordinates.npy"), patch_coordinates)

    metadata = {
        "patient_id": patient_id,
        "split": split_name,
        "modality_order": ["t1n", "t1c", "t2w", "t2f"],
        "image_shape": list(image_patches.shape),
        "mask_shape": list(mask_patches.shape),
        "patch_coordinates": patch_coordinates.tolist(),
        "patch_info": patch_info,
        "crop_bbox": patient["crop_bbox"],
        "cropped_shape": patient["cropped_shape"],
        "original_shape": patient["spatial"]["original_shape"],
        "spacing": patient["spatial"]["spacing"],
        "affine": patient["spatial"]["affine"],
        "segmentation_labels": [0, 1, 2, 3],
    }

    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    del patient, stacked, image_patches, mask_patches, patch_coordinates
    gc.collect()

    print(
        f"Processed {patient_id} -> {split_name} | "
        f"patches={metadata['image_shape'][0]} | "
        f"cropped={metadata['cropped_shape']} | "
        f"padded={patch_info['padded_shape']}"
    )


def run_preprocessing(percentage=DEFAULT_PATIENT_PERCENTAGE, split_root=None):
    if split_root is None:
        split_root = os.path.abspath(os.path.join(PROCESSED_DATA_ROOT, "..", "splits"))
    else:
        split_root = os.path.abspath(split_root)

    os.makedirs(split_root, exist_ok=True)
    prepare_processed_root(PROCESSED_DATA_ROOT)

    patient_ids = get_patient_ids(DATASET_ROOT, percentage=percentage)
    train_ids, val_ids, test_ids = build_patient_split(patient_ids)

    save_split(train_ids, "train", split_root)
    save_split(val_ids, "val", split_root)
    save_split(test_ids, "test", split_root)

    for patient_id in train_ids:
        process_patient(patient_id, "train")
    for patient_id in val_ids:
        process_patient(patient_id, "val")
    for patient_id in test_ids:
        process_patient(patient_id, "test")

    print(
        f"Finished preprocessing: {len(train_ids)} train, "
        f"{len(val_ids)} val, {len(test_ids)} test"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BraTS patients into 64x64x64 patches.")
    parser.add_argument("--percentage", type=int, default=DEFAULT_PATIENT_PERCENTAGE)
    parser.add_argument("--split-root", type=str, default=None)
    args = parser.parse_args()
    run_preprocessing(percentage=args.percentage, split_root=args.split_root)
