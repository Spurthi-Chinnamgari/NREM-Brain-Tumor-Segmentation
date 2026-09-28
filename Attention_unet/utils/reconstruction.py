"""Coordinate-based reconstruction for processed 3D patches."""

from typing import Sequence

import numpy as np


def reconstruct_patches(
    patches: np.ndarray,
    coordinates: np.ndarray,
    padded_shape: Sequence[int],
    patch_size: Sequence[int] = (64, 64, 64),
    output_kind: str = "labels",
) -> np.ndarray:
    """Reconstruct patches using preprocessing coordinates in ``(x, y, z)`` order.

    Inputs may be label patches shaped ``(N, X, Y, Z)`` or channel-first
    logits/probabilities shaped ``(N, C, X, Y, Z)``. Overlapping patches are
    averaged rather than overwritten.
    """
    patches = np.asarray(patches)
    coordinates = np.asarray(coordinates)
    padded_shape = tuple(int(value) for value in padded_shape)
    patch_size = tuple(int(value) for value in patch_size)

    if patches.ndim not in (4, 5):
        raise ValueError(f"Expected 4D or 5D patches, got {patches.shape}")
    if coordinates.shape != (patches.shape[0], 3):
        raise ValueError("Patch and coordinate counts/shapes do not match")
    if patches.shape[-3:] != patch_size:
        raise ValueError(f"Patch shape {patches.shape[-3:]} does not match {patch_size}")
    if output_kind not in ("labels", "logits", "probabilities"):
        raise ValueError("output_kind must be 'labels', 'logits', or 'probabilities'")
    if np.any(coordinates < 0) or np.any(
        coordinates + np.asarray(patch_size) > np.asarray(padded_shape)
    ):
        raise ValueError("Patch coordinates exceed padded volume bounds")

    channels = patches.shape[1] if patches.ndim == 5 else None
    if output_kind == "labels" and channels is not None:
        raise ValueError("Label patches must have shape (N, X, Y, Z), not a channel dimension")
    output_shape = (channels, *padded_shape) if channels is not None else padded_shape
    prediction_sum = np.zeros(output_shape, dtype=np.float64)
    count_map = np.zeros(padded_shape, dtype=np.uint32)

    for patch, coordinate in zip(patches, coordinates):
        x, y, z = (int(value) for value in coordinate)
        target = (
            slice(x, x + patch_size[0]),
            slice(y, y + patch_size[1]),
            slice(z, z + patch_size[2]),
        )
        if channels is None:
            prediction_sum[target] += patch
        else:
            prediction_sum[(slice(None), *target)] += patch
        count_map[target] += 1

    if np.any(count_map == 0):
        raise ValueError("Patch coordinates do not cover the padded volume")

    if output_kind == "labels":
        label_votes = np.zeros((4, *padded_shape), dtype=np.uint32)
        for patch, coordinate in zip(patches.astype(np.int64), coordinates):
            x, y, z = (int(value) for value in coordinate)
            target = (
                slice(x, x + patch_size[0]),
                slice(y, y + patch_size[1]),
                slice(z, z + patch_size[2]),
            )
            for label in range(4):
                label_votes[(label, *target)] += patch == label
        reconstructed = np.argmax(label_votes, axis=0)
    elif channels is None:
        reconstructed = prediction_sum / count_map
    else:
        reconstructed = prediction_sum / count_map[np.newaxis, ...]

    if output_kind == "labels":
        return reconstructed.astype(np.uint8)
    return reconstructed.astype(np.float32)


def remove_padding(volume: np.ndarray, cropped_shape: Sequence[int]) -> np.ndarray:
    """Remove the high-end padding added by preprocessing."""
    cropped_shape = tuple(int(value) for value in cropped_shape)
    slices = tuple(slice(0, value) for value in cropped_shape)
    return np.asarray(volume)[(...,) + slices]


def restore_original_space(
    cropped_volume: np.ndarray,
    original_shape: Sequence[int],
    crop_bbox: Sequence[int],
) -> np.ndarray:
    """Place cropped ``(x, y, z)`` data back into original array space."""
    original_shape = tuple(int(value) for value in original_shape)
    bbox = tuple(int(value) for value in crop_bbox)
    if len(bbox) != 6:
        raise ValueError("crop_bbox must be (xmin, xmax, ymin, ymax, zmin, zmax)")

    target = (
        slice(bbox[0], bbox[1] + 1),
        slice(bbox[2], bbox[3] + 1),
        slice(bbox[4], bbox[5] + 1),
    )
    expected_shape = tuple(item.stop - item.start for item in target)
    if cropped_volume.shape[-3:] != expected_shape:
        raise ValueError(
            f"Cropped volume shape {cropped_volume.shape[-3:]} does not match "
            f"crop_bbox shape {expected_shape}"
        )

    restored = np.zeros(cropped_volume.shape[:-3] + original_shape, dtype=cropped_volume.dtype)
    restored[(..., *target)] = cropped_volume
    return restored


def reconstruct_from_metadata(
    patches: np.ndarray,
    coordinates: np.ndarray,
    metadata: dict,
    output_kind: str = "labels",
    restore_original: bool = False,
) -> np.ndarray:
    """Reconstruct and crop, optionally restoring the original patient shape."""
    patch_info = metadata["patch_info"]
    volume = reconstruct_patches(
        patches,
        coordinates,
        patch_info["padded_shape"],
        patch_info["patch_size"],
        output_kind=output_kind,
    )
    volume = remove_padding(volume, metadata["cropped_shape"])
    if restore_original:
        volume = restore_original_space(
            volume,
            metadata["original_shape"],
            metadata["crop_bbox"],
        )
    return volume