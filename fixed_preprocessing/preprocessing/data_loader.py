import os
from typing import Dict

import nibabel as nib
import numpy as np

from ..config import MODALITIES, SEGMENTATION


def _load_nifti(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    return data, nii


def load_modality(patient_dir: str, patient_id: str, modality: str) -> np.ndarray:
    path = os.path.join(patient_dir, f"{patient_id}-{modality}.nii.gz")
    data, _ = _load_nifti(path)
    return data


def load_patient(patient_dir: str, patient_id: str) -> Dict:
    """Load a BraTS patient and preserve the spatial metadata needed later
    for patch-to-volume reconstruction and NIfTI output.
    """
    patient = {
        "modalities": {},
        "segmentation": None,
        "spatial": None,
    }

    reference_shape = None
    reference_affine = None
    reference_spacing = None

    for modality in MODALITIES:
        path = os.path.join(patient_dir, f"{patient_id}-{modality}.nii.gz")
        data, nii = _load_nifti(path)

        if reference_shape is None:
            reference_shape = tuple(data.shape)
            reference_affine = nii.affine.copy()
            reference_spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
        else:
            if tuple(data.shape) != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {patient_id}: {modality} has {data.shape}, "
                    f"expected {reference_shape}."
                )
            if not np.allclose(nii.affine, reference_affine, atol=1e-5):
                raise ValueError(
                    f"Affine mismatch for {patient_id}: modality {modality} is not "
                    "spatially aligned with the reference modality."
                )

        patient["modalities"][modality] = data

    seg_path = os.path.join(patient_dir, f"{patient_id}-{SEGMENTATION}.nii.gz")
    segmentation, seg_nii = _load_nifti(seg_path)

    if tuple(segmentation.shape) != reference_shape:
        raise ValueError(
            f"Segmentation shape mismatch for {patient_id}: {segmentation.shape}; "
            f"expected {reference_shape}."
        )

    if not np.allclose(seg_nii.affine, reference_affine, atol=1e-5):
        raise ValueError(
            f"Segmentation affine mismatch for {patient_id}."
        )

    # BraTS labels used by this project are integer classes 0..3.
    if not np.all(np.isfinite(segmentation)):
        raise ValueError(f"Segmentation contains NaN/Inf values: {patient_id}")

    rounded_seg = np.rint(segmentation)
    if not np.allclose(segmentation, rounded_seg):
        raise ValueError(f"Segmentation contains non-integer labels: {patient_id}")

    unique_labels = set(np.unique(rounded_seg).astype(np.int64).tolist())
    if not unique_labels.issubset({0, 1, 2, 3}):
        raise ValueError(
            f"Unexpected segmentation labels for {patient_id}: {sorted(unique_labels)}"
        )

    patient["segmentation"] = rounded_seg.astype(np.int16)
    patient["spatial"] = {
        "original_shape": list(reference_shape),
        "affine": reference_affine.tolist(),
        "spacing": list(reference_spacing),
    }

    return patient


def get_patient_ids(dataset_root: str, percentage: int = 100):
    """Return a deterministic prefix of sorted patient IDs."""
    if percentage <= 0 or percentage > 100:
        raise ValueError("percentage must be between 1 and 100")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    patient_ids = sorted(
    name
    for name in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, name))
    and name.startswith("BraTS-GLI-")
)

    if not patient_ids:
        raise ValueError(f"No patient directories found in: {dataset_root}")

    number_of_patients = max(1, int(len(patient_ids) * percentage / 100))
    return patient_ids[:number_of_patients]
