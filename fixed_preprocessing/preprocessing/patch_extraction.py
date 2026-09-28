import numpy as np


def _required_padded_size(length, patch, stride):
    if length <= patch:
        return patch
    steps = int(np.ceil((length - patch) / stride))
    return patch + steps * stride


def extract_patches(
    image,
    segmentation,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64),
):
    """Extract complete 3D patches and return their coordinates/metadata.

    Padding is added only at the end of each spatial axis, so every voxel of
    the cropped volume is covered by at least one patch. With the current
    baseline (stride == patch size), patches do not overlap.
    """
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError(f"Expected image shape (4, X, Y, Z), got {image.shape}")
    if segmentation.shape != image.shape[1:]:
        raise ValueError(
            f"Image/segmentation shape mismatch: {image.shape} vs {segmentation.shape}"
        )

    px, py, pz = patch_size
    sx, sy, sz = stride
    if min(*patch_size, *stride) <= 0:
        raise ValueError("patch_size and stride must contain positive values")

    _, x, y, z = image.shape
    padded_shape = (
        _required_padded_size(x, px, sx),
        _required_padded_size(y, py, sy),
        _required_padded_size(z, pz, sz),
    )
    pad_after = tuple(padded_shape[i] - (x, y, z)[i] for i in range(3))

    padded_image = np.pad(
        image,
        ((0, 0), (0, pad_after[0]), (0, pad_after[1]), (0, pad_after[2])),
        mode="constant",
        constant_values=0,
    )
    padded_segmentation = np.pad(
        segmentation,
        ((0, pad_after[0]), (0, pad_after[1]), (0, pad_after[2])),
        mode="constant",
        constant_values=0,
    )

    image_patches = []
    segmentation_patches = []
    patch_coordinates = []

    for i in range(0, padded_shape[0] - px + 1, sx):
        for j in range(0, padded_shape[1] - py + 1, sy):
            for k in range(0, padded_shape[2] - pz + 1, sz):
                image_patches.append(padded_image[:, i:i + px, j:j + py, k:k + pz])
                segmentation_patches.append(padded_segmentation[i:i + px, j:j + py, k:k + pz])
                patch_coordinates.append((i, j, k))

    return (
        np.asarray(image_patches, dtype=np.float32),
        np.asarray(segmentation_patches, dtype=np.int16),
        np.asarray(patch_coordinates, dtype=np.int32),
        {
            "original_cropped_shape": [x, y, z],
            "padded_shape": list(padded_shape),
            "pad_after": list(pad_after),
            "patch_size": list(patch_size),
            "stride": list(stride),
        },
    )
