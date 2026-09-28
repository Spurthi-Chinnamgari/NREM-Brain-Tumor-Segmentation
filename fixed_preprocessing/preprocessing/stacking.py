import numpy as np

from ..config import MODALITIES


def stack_modalities(patient):
    stacked = np.stack(
        [patient["modalities"][modality] for modality in MODALITIES],
        axis=0,
    ).astype(np.float32)

    if stacked.shape[0] != 4:
        raise ValueError(f"Expected 4 MRI modalities, got {stacked.shape[0]}")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("Stacked MRI data contains NaN/Inf values.")

    return stacked
