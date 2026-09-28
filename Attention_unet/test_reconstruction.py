import numpy as np

from utils.reconstruction import reconstruct_from_metadata, reconstruct_patches


def test_non_overlapping_label_reconstruction():
    patches = np.stack([
        np.ones((2, 2, 2), dtype=np.uint8),
        np.full((2, 2, 2), 2, dtype=np.uint8),
    ])
    coordinates = np.asarray([[0, 0, 0], [0, 0, 2]], dtype=np.int32)
    reconstructed = reconstruct_patches(
        patches, coordinates, padded_shape=(2, 2, 4), patch_size=(2, 2, 2)
    )
    assert reconstructed.shape == (2, 2, 4)
    assert np.all(reconstructed[:, :, :2] == 1)
    assert np.all(reconstructed[:, :, 2:] == 2)


def test_overlapping_labels_use_majority_vote():
    patches = np.stack([
        np.zeros((2, 2, 2), dtype=np.uint8),
        np.ones((2, 2, 2), dtype=np.uint8),
        np.ones((2, 2, 2), dtype=np.uint8),
    ])
    coordinates = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    reconstructed = reconstruct_patches(
        patches, coordinates, padded_shape=(2, 2, 2), patch_size=(2, 2, 2)
    )
    assert np.all(reconstructed == 1)


def test_metadata_crop_and_original_restoration():
    patches = np.ones((1, 2, 2, 2), dtype=np.uint8)
    metadata = {
        "patch_info": {"padded_shape": [2, 2, 2], "patch_size": [2, 2, 2]},
        "cropped_shape": [1, 2, 2],
        "original_shape": [3, 4, 4],
        "crop_bbox": [1, 1, 1, 2, 1, 2],
    }
    restored = reconstruct_from_metadata(
        patches,
        np.asarray([[0, 0, 0]], dtype=np.int32),
        metadata,
        restore_original=True,
    )
    assert restored.shape == (3, 4, 4)
    assert np.all(restored[1, 1:3, 1:3] == 1)
    assert np.count_nonzero(restored) == 4
