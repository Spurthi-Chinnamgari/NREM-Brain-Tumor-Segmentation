import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_UNET_DIR = PROJECT_ROOT / "Attention_unet"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "postprocessing" / "gradcam_results"

sys.path.insert(0, str(PROJECT_ROOT))
from Attention_unet.models.attention_unet_3d import AttentionUNet3D


def _load_checkpoint(model: torch.nn.Module, device: torch.device) -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {CHECKPOINT}")
    checkpoint = torch.load(str(CHECKPOINT), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)


def _select_patch_and_class(patient_id: str, target_class: int | None, patch_index: int | None) -> tuple[int, int]:
    prediction_path = ATTENTION_UNET_DIR / "results" / "predictions" / f"{patient_id}_predictions.npy"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Inference patch predictions do not exist: {prediction_path}")
    predictions = np.asarray(np.load(prediction_path), dtype=np.uint8)
    if predictions.ndim != 4 or predictions.shape[1:] != (64, 64, 64):
        raise ValueError(f"Unexpected inference patch prediction shape: {predictions.shape}")

    if target_class is None:
        class_counts = [int(np.count_nonzero(predictions == label)) for label in (1, 2, 3)]
        target_class = (1, 2, 3)[int(np.argmax(class_counts))] if max(class_counts) else 0
    if target_class not in range(4):
        raise ValueError("target class must be one of 0, 1, 2, or 3")

    if patch_index is None:
        patch_index = int(np.argmax([np.count_nonzero(patch == target_class) for patch in predictions]))
    if not 0 <= patch_index < predictions.shape[0]:
        raise IndexError(f"patch index {patch_index} is outside [0, {predictions.shape[0] - 1}]")
    return patch_index, target_class


def generate_gradcam(patient_id: str, target_class: int | None = None, patch_index: int | None = None) -> list[Path]:
    patient_dir = PROCESSED_ROOT / "test" / patient_id
    images_path = patient_dir / "images.npy"
    coordinates_path = patient_dir / "patch_coordinates.npy"
    metadata_path = patient_dir / "metadata.json"
    for path in (images_path, coordinates_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Grad-CAM input: {path}")

    images = np.load(images_path, mmap_mode="r")
    coordinates = np.load(coordinates_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if images.ndim != 5 or images.shape[1:] != (4, 64, 64, 64):
        raise ValueError(f"Unexpected patient image shape: {images.shape}")

    patch_index, target_class = _select_patch_and_class(patient_id, target_class, patch_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet3D(
        in_channels=4,
        out_channels=4,
        features=[16, 32, 64, 128, 256],
        dropout=0.1,
        use_transpose=True,
    ).to(device)
    _load_checkpoint(model, device)
    model.eval()

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.decoder_blocks[-1].conv
    forward_handle = target_layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_handle = target_layer.register_full_backward_hook(lambda _module, _inputs, output: gradients.append(output[0]))

    try:
        patch_array = np.array(images[patch_index], dtype=np.float32, copy=True)
        patch_tensor = torch.as_tensor(patch_array, device=device).unsqueeze(0)
        model.zero_grad(set_to_none=True)
        output = model(patch_tensor)
        if tuple(output.shape) != (1, 4, 64, 64, 64):
            raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")
        prediction = torch.argmax(output, dim=1)
        target_voxels = prediction == target_class
        if not torch.any(target_voxels):
            raise RuntimeError(f"Selected target class {target_class} is absent from the model prediction for patch {patch_index}")
        score = output[:, target_class][target_voxels].mean()
        score.backward()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=patch_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy().astype(np.float32)
    finally:
        forward_handle.remove()
        backward_handle.remove()

    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{patient_id}_patch{patch_index}_class{target_class}_gradcam"
    npy_path = OUTPUT_DIR / f"{stem}.npy"
    png_path = OUTPUT_DIR / f"{stem}.png"
    metadata_output_path = OUTPUT_DIR / f"{stem}.json"
    np.save(npy_path, cam)

    slice_index = cam.shape[0] // 2
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(cam[slice_index], cmap="jet", vmin=0.0, vmax=1.0)
    axis.set_title(f"3D Grad-CAM | {patient_id} | patch {patch_index} | class {target_class}")
    axis.axis("off")
    figure.colorbar(axis.images[0], ax=axis, label="Grad-CAM intensity")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metadata_output_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "patch_index": patch_index,
                "target_class": target_class,
                "patch_coordinate": np.asarray(coordinates[patch_index]).astype(int).tolist(),
                "patch_size": metadata["patch_info"]["patch_size"],
                "coordinate_convention": "(x, y, z) in cropped padded space",
                "source_images": str(images_path.resolve()),
                "source_checkpoint": str(CHECKPOINT.resolve()),
                "space": "processed patch space; not an original-space heatmap",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [npy_path, png_path, metadata_output_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate genuine model-based 3D Grad-CAM for a patient patch.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--patch-index", type=int, default=None, help="Patch to explain; defaults to the patch with most target voxels.")
    parser.add_argument("--target-class", type=int, default=None, help="Class to explain; defaults to the most prevalent predicted tumor class.")
    args = parser.parse_args()
    for output in generate_gradcam(args.patient_id, args.target_class, args.patch_index):
        print(f"Saved Grad-CAM artifact: {output.resolve()}")


if __name__ == "__main__":
    main()
