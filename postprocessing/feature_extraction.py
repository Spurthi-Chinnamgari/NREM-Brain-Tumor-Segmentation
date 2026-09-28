import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_PATH = PROJECT_ROOT / "postprocessing" / "postprocessed" / "cleaned_prediction_mask.nii.gz"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "postprocessing" / "results" / "features.json"

REGIONS = {
    "WT": {"name": "Whole Tumor", "labels": [1, 2, 3]},
    "TC": {"name": "Tumor Core", "labels": [1, 3]},
    "ET": {"name": "Enhancing Tumor", "labels": [3]},
}


def _finite_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _finite_vector(values: np.ndarray | tuple[float, ...]) -> list[float]:
    return [_finite_float(value) for value in values]


def _load_mask(mask_path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image, np.ndarray]:
    path = Path(mask_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction mask does not exist: {path}")

    image = nib.load(str(path))
    mask_data = np.asarray(image.get_fdata(dtype=np.float64), dtype=np.float64)
    if mask_data.ndim != 3:
        raise ValueError(f"Prediction mask must be 3D: {path}")
    if not np.all(np.isfinite(mask_data)):
        raise ValueError(f"Prediction mask contains NaN or infinite values: {path}")

    rounded = np.rint(mask_data)
    if not np.allclose(mask_data, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(f"Prediction mask contains non-integer labels: {path}")

    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Prediction mask has invalid voxel spacing: {spacing}")

    return rounded.astype(np.int16), image, spacing


def _surface_area(region_mask: np.ndarray, spacing: np.ndarray) -> float:
    if np.count_nonzero(region_mask) == 0:
        return 0.0

    padded_mask = np.pad(region_mask.astype(np.uint8), 1, mode="constant")
    try:
        vertices, faces, _, _ = marching_cubes(
            padded_mask,
            level=0.5,
            spacing=tuple(float(value) for value in spacing),
        )
        return _finite_float(mesh_surface_area(vertices, faces))
    except (RuntimeError, ValueError):
        return 0.0


def _elongation(region_mask: np.ndarray, spacing: np.ndarray) -> float:
    coordinates = np.argwhere(region_mask).astype(float) * spacing
    if coordinates.shape[0] < 2:
        return 1.0

    covariance = np.atleast_2d(np.cov(coordinates, rowvar=False))
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    smallest = max(float(eigenvalues[0]), np.finfo(float).eps)
    largest = max(float(eigenvalues[-1]), 0.0)
    return _finite_float(math.sqrt(largest / smallest))


def _empty_features() -> dict[str, object]:
    return {
        "tumor_present": False,
        "voxel_count": 0,
        "volume_mm3": 0.0,
        "volume_cm3": 0.0,
        "bounding_box_mm": {"length": 0.0, "width": 0.0, "height": 0.0},
        "centroid_voxel": None,
        "centroid_world_mm": None,
        "location": "Not detected",
        "elongation": 0.0,
        "sphericity": 0.0,
        "compactness": 0.0,
        "surface_area_mm2": 0.0,
    }


def extract_features(region_mask: np.ndarray, spacing: np.ndarray, affine: np.ndarray) -> dict[str, object]:
    region_mask = np.asarray(region_mask, dtype=bool)
    voxel_count = int(np.count_nonzero(region_mask))
    if voxel_count == 0:
        return _empty_features()

    coordinates = np.argwhere(region_mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    dimensions_mm = (maximum - minimum + 1).astype(float) * spacing

    centroid_voxel = np.asarray(ndimage.center_of_mass(region_mask), dtype=float)
    centroid_world_mm = np.asarray(nib.affines.apply_affine(affine, centroid_voxel), dtype=float)
    volume_mm3 = _finite_float(voxel_count * float(np.prod(spacing)))
    surface_area_mm2 = _surface_area(region_mask, spacing)
    sphericity = 0.0
    if surface_area_mm2 > 0.0:
        sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2

    bounding_box_volume_mm3 = float(np.prod(dimensions_mm))
    compactness = volume_mm3 / bounding_box_volume_mm3 if bounding_box_volume_mm3 > 0 else 0.0

    return {
        "tumor_present": True,
        "voxel_count": voxel_count,
        "volume_mm3": volume_mm3,
        "volume_cm3": _finite_float(volume_mm3 / 1000.0),
        "bounding_box_mm": {
            "length": _finite_float(dimensions_mm[0]),
            "width": _finite_float(dimensions_mm[1]),
            "height": _finite_float(dimensions_mm[2]),
        },
        "centroid_voxel": _finite_vector(centroid_voxel),
        "centroid_world_mm": _finite_vector(centroid_world_mm),
        "location": {"centroid_world_mm": _finite_vector(centroid_world_mm)},
        "elongation": _elongation(region_mask, spacing),
        "sphericity": min(max(_finite_float(sphericity), 0.0), 1.0),
        "compactness": _finite_float(compactness),
        "surface_area_mm2": surface_area_mm2,
    }


def extract_all_features(mask_path: str | Path = DEFAULT_MASK_PATH) -> dict[str, object]:
    mask, image, spacing = _load_mask(mask_path)
    return {
        "mask_path": str(Path(mask_path).expanduser().resolve()),
        "voxel_spacing_mm": _finite_vector(spacing),
        "tumor_present": bool(np.any(mask > 0)),
        "regions": {
            code: extract_features(np.isin(mask, region["labels"]), spacing, image.affine)
            for code, region in REGIONS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from the final cleaned prediction mask.")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK_PATH, help="Final cleaned prediction mask NIfTI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Feature report JSON path.")
    args = parser.parse_args()

    report = extract_all_features(args.mask)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)

    print("=" * 70)
    print("TUMOR FEATURE EXTRACTION COMPLETED")
    print(f"Mask: {report['mask_path']}")
    print(f"Saved JSON: {output_path}")
    for code, region in REGIONS.items():
        result = report["regions"][code]
        status = "present" if result["tumor_present"] else "not detected"
        print(f"{code} ({region['name']}): {status}")
        print(f"  Volume: {result['volume_mm3']:.2f} mm3 ({result['volume_cm3']:.4f} cm3)")
        print(f"  Surface area: {result['surface_area_mm2']:.2f} mm2")


if __name__ == "__main__":
    main()
