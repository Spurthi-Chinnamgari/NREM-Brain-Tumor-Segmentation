import numpy as np


def get_bounding_box(image):
    """Return an inclusive XYZ bounding box around non-zero voxels."""
    brain_mask = image > 0
    if not np.any(brain_mask):
        raise ValueError("Cannot compute a brain bounding box: image contains no non-zero voxels.")

    x, y, z = np.where(brain_mask)
    return (
        int(x.min()), int(x.max()),
        int(y.min()), int(y.max()),
        int(z.min()), int(z.max()),
    )


def crop_volume(image, bbox, padding=5):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    xmin = max(0, xmin - padding)
    xmax = min(image.shape[0] - 1, xmax + padding)
    ymin = max(0, ymin - padding)
    ymax = min(image.shape[1] - 1, ymax + padding)
    zmin = max(0, zmin - padding)
    zmax = min(image.shape[2] - 1, zmax + padding)

    return image[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]


def padded_crop_bbox(bbox, image_shape, padding=5):
    """Return the actual inclusive crop bbox after clipping to image bounds."""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    return (
        max(0, xmin - padding), min(image_shape[0] - 1, xmax + padding),
        max(0, ymin - padding), min(image_shape[1] - 1, ymax + padding),
        max(0, zmin - padding), min(image_shape[2] - 1, zmax + padding),
    )


def crop_patient(patient, padding=5):
    reference = patient["modalities"]["t1n"]

    bbox = get_bounding_box(reference)

    # Calculate the actual crop bounding box first
    actual_bbox = padded_crop_bbox(
        bbox,
        reference.shape,
        padding=padding
    )

    # Crop all MRI modalities
    for modality in patient["modalities"]:
        patient["modalities"][modality] = crop_volume(
            patient["modalities"][modality],
            actual_bbox,
            padding=0
        )

    # Crop segmentation using the same bounding box
    patient["segmentation"] = crop_volume(
        patient["segmentation"],
        actual_bbox,
        padding=0
    )

    # Store crop information
    patient["crop_bbox"] = list(actual_bbox)
    patient["cropped_shape"] = list(patient["segmentation"].shape)

    return patient
