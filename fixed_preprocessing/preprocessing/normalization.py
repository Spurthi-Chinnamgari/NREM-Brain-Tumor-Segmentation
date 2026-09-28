import numpy as np


def normalize_volume(image):
    """Z-score normalize non-zero voxels while keeping background at zero."""
    image = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError("MRI volume contains NaN/Inf values before normalization.")

    brain_mask = image > 0
    if not np.any(brain_mask):
        raise ValueError("MRI volume contains no non-zero voxels.")

    brain_voxels = image[brain_mask]
    mean = float(np.mean(brain_voxels))
    std = float(np.std(brain_voxels))

    normalized = np.zeros_like(image, dtype=np.float32)
    if std < 1e-8:
        # A constant non-zero image cannot be meaningfully z-scored.
        # Keep the non-zero region at zero rather than creating NaN/Inf.
        return normalized

    normalized[brain_mask] = (brain_voxels - mean) / std

    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalization produced NaN/Inf values.")

    return normalized


def normalize_patient(patient):
    for modality in patient["modalities"]:
        patient["modalities"][modality] = normalize_volume(
            patient["modalities"][modality]
        )
    return patient
