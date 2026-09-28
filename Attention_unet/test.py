"""
3D Attention U-Net Test / Evaluation Pipeline
----------------------------------------------

Uses the VERIFIED processed BraTS preprocessing output.

Expected:

data/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt

Processed sample:

    image -> (4, 64, 64, 64)
    mask  -> (64, 64, 64)

Model:

    input  -> (B, 4, 64, 64, 64)
    output -> (B, 4, 64, 64, 64)

Prediction:

    argmax(logits, dim=1)
        ->
    (B, 64, 64, 64)

IMPORTANT:

    - No raw NIfTI loading
    - No random patient split
    - No Z-score normalization
    - No random crop
    - No synthetic data
    - No 2D testing
    - Missing checkpoint = hard error
"""


import os
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_default_config

from dataset import (
    BraTSDataset3D,
    read_split_file,
)

from models import AttentionUNet3D

from metrics import (
    compute_dice_score,
    compute_iou_score,
    compute_precision_recall_f1,
    compute_brats_regions_metrics,
    compute_hausdorff_distance_95,
)

from utils import reconstruct_from_metadata

from utils import (
    setup_logger,
    load_checkpoint,
    save_prediction_comparison,
)


# ============================================================
# CONSTANTS
# ============================================================

PATCH_SIZE = (64, 64, 64)

NUM_MODALITIES = 4
NUM_CLASSES = 4


# ============================================================
# FIND DATA PATHS
# ============================================================

def find_data_paths(
    processed_root=None,
    split_root=None,
):
    """
    Locate processed data and split directories.

    Supported structures:

        project/
        ├── preprocessing/
        │   └── data/
        │       ├── processed/
        │       └── splits/
        │
        └── Attention_unet/

    or:

        project/
        ├── data/
        │   ├── processed/
        │   └── splits/
        │
        └── Attention_unet/
    """

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(
        current_dir
    )

    if processed_root is not None:
        processed_root = os.path.abspath(
            processed_root
        )

    if split_root is not None:
        split_root = os.path.abspath(
            split_root
        )

    processed_candidates = [
        os.path.join(
            project_root,
            "preprocessing",
            "data",
            "processed",
        ),

        os.path.join(
            project_root,
            "data",
            "processed",
        ),

        os.path.join(
            current_dir,
            "data",
            "processed",
        ),
    ]

    split_candidates = [
        os.path.join(
            project_root,
            "preprocessing",
            "data",
            "splits",
        ),

        os.path.join(
            project_root,
            "data",
            "splits",
        ),

        os.path.join(
            current_dir,
            "data",
            "splits",
        ),
    ]

    # --------------------------------------------------------
    # Processed root
    # --------------------------------------------------------

    if processed_root is None:

        for path in processed_candidates:

            if os.path.isdir(path):

                processed_root = path
                break

    # --------------------------------------------------------
    # Split root
    # --------------------------------------------------------

    if split_root is None:

        for path in split_candidates:

            if os.path.isdir(path):

                split_root = path
                break

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if processed_root is None:

        raise FileNotFoundError(
            "\nProcessed data directory not found.\n\n"
            "Expected data/processed containing:\n"
            "    train/\n"
            "    val/\n"
            "    test/\n\n"
            "Use --processed-root to specify it manually."
        )

    if split_root is None:

        raise FileNotFoundError(
            "\nSplit directory not found.\n\n"
            "Expected data/splits containing:\n"
            "    train.txt\n"
            "    val.txt\n"
            "    test.txt\n\n"
            "Use --split-root to specify it manually."
        )

    return (
        processed_root,
        split_root,
    )


# ============================================================
# VERIFY TEST SPLIT
# ============================================================

def verify_test_split(
    split_root,
):
    """
    Read and verify the exact test split.
    """

    test_file = os.path.join(
        split_root,
        "test.txt",
    )

    test_ids = read_split_file(
        test_file
    )

    print()
    print("=" * 70)
    print("EXACT TEST SPLIT")
    print("=" * 70)

    print(
        f"Test patients: {len(test_ids)}"
    )

    for patient_id in test_ids:

        print(
            f"  {patient_id}"
        )

    print("=" * 70)

    return test_ids


# ============================================================
# VERIFY CHECKPOINT
# ============================================================

def verify_checkpoint(
    checkpoint_path,
):
    """
    Real evaluation MUST use a trained checkpoint.
    """

    checkpoint_path = os.path.abspath(
        checkpoint_path
    )

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nTRAINED CHECKPOINT NOT FOUND.\n\n"
            f"Expected:\n{checkpoint_path}\n\n"
            "Testing has been stopped intentionally.\n"
            "A randomly initialized model must NOT be "
            "used for real evaluation."
        )

    return checkpoint_path


# ============================================================
# EVALUATE
# ============================================================

def evaluate_test_set(
    model,
    dataloader,
    device,
    results_dir,
    logger,
):
    """
    Evaluate the complete processed test patch set.

    NOTE:
        This evaluates processed 64^3 patches directly.

        True patient-volume reconstruction requires patch
        coordinates / cropped-volume metadata from preprocessing.
    """

    model.eval()

    all_dices = []
    all_ious = []

    all_wt_dices = []
    all_tc_dices = []
    all_et_dices = []

    all_hd95s = []

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    with torch.no_grad():

        for batch_index, batch in enumerate(
            tqdm(
                dataloader,
                desc="Evaluating Test Set",
            )
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            # =================================================
            # INPUT VALIDATION
            # =================================================

            if images.ndim != 5:

                raise RuntimeError(
                    "\nInvalid test input shape.\n"
                    f"Expected: (B, 4, 64, 64, 64)\n"
                    f"Got:      {tuple(images.shape)}"
                )

            if (
                images.shape[1:] !=
                (
                    NUM_MODALITIES,
                    *PATCH_SIZE
                )
            ):

                raise RuntimeError(
                    "\nInvalid test patch shape.\n"
                    f"Expected: "
                    f"(B, 4, 64, 64, 64)\n"
                    f"Got:      {tuple(images.shape)}"
                )

            # =================================================
            # MODEL
            # =================================================

            if device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    logits = model(
                        images
                    )

            else:

                logits = model(
                    images
                )

            # =================================================
            # OUTPUT VALIDATION
            # =================================================

            expected_output = (
                images.shape[0],
                NUM_CLASSES,
                *PATCH_SIZE
            )

            if tuple(
                logits.shape
            ) != expected_output:

                raise RuntimeError(
                    "\nInvalid model output shape.\n"
                    f"Expected: {expected_output}\n"
                    f"Got:      {tuple(logits.shape)}"
                )

            # =================================================
            # ARGMAX
            # =================================================

            preds = torch.argmax(
                logits,
                dim=1,
            )

            # =================================================
            # LABEL CHECK
            # =================================================

            unique_predictions = torch.unique(
                preds
            )

            if torch.any(
                (unique_predictions < 0)
                |
                (unique_predictions >= NUM_CLASSES)
            ):

                raise RuntimeError(
                    "\nInvalid prediction labels:\n"
                    f"{unique_predictions.cpu().tolist()}"
                )

            # =================================================
            # METRICS
            # =================================================

            mean_dice, class_dice = (
                compute_dice_score(
                    preds,
                    masks,
                )
            )

            mean_iou, class_iou = (
                compute_iou_score(
                    preds,
                    masks,
                )
            )

            brats_metrics = (
                compute_brats_regions_metrics(
                    preds,
                    masks,
                )
            )

            # =================================================
            # HD95
            #
            # Compute ET HD95 for each sample in batch.
            # =================================================

            for sample_index in range(
                preds.shape[0]
            ):

                prediction_np = (
                    preds[
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_np = (
                    masks[
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                prediction_et = (
                    prediction_np == 3
                )

                target_et = (
                    target_np == 3
                )

                hd95 = (
                    compute_hausdorff_distance_95(
                        prediction_et,
                        target_et,
                    )
                )

                if np.isfinite(hd95):

                    all_hd95s.append(
                        float(hd95)
                    )

            # =================================================
            # STORE METRICS
            # =================================================

            all_dices.append(
                float(mean_dice)
            )

            all_ious.append(
                float(mean_iou)
            )

            all_wt_dices.append(
                float(
                    brats_metrics[
                        "dice_wt"
                    ]
                )
            )

            all_tc_dices.append(
                float(
                    brats_metrics[
                        "dice_tc"
                    ]
                )
            )

            all_et_dices.append(
                float(
                    brats_metrics[
                        "dice_et"
                    ]
                )
            )

            # =================================================
            # VISUALIZATION
            #
            # Save first five processed test patches.
            # =================================================

            if batch_index < 5:

                image_np = (
                    images[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_np = (
                    masks[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                prediction_np = (
                    preds[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                patient_id = batch[
                    "patient_id"
                ][0]

                patch_index = batch[
                    "patch_index"
                ][0]

                save_path = os.path.join(
                    results_dir,
                    (
                        f"test_"
                        f"{patient_id}_"
                        f"patch_{patch_index}.png"
                    ),
                )

                try:

                    save_prediction_comparison(
                        image_np,
                        target_np,
                        prediction_np,
                        save_path=save_path,
                        title=(
                            f"Test: "
                            f"{patient_id} "
                            f"Patch {patch_index}"
                        ),
                    )

                except Exception as e:

                    logger.warning(
                        "Could not save "
                        f"visualization: {e}"
                    )

    # ========================================================
    # SUMMARY
    # ========================================================

    def safe_mean(values):

        if len(values) == 0:
            return 0.0

        return float(
            np.mean(values)
        )

    metrics_summary = {

        "test_mean_dice":
            safe_mean(all_dices),

        "test_mean_iou":
            safe_mean(all_ious),

        "test_dice_wt":
            safe_mean(all_wt_dices),

        "test_dice_tc":
            safe_mean(all_tc_dices),

        "test_dice_et":
            safe_mean(all_et_dices),

        "test_mean_hd95_et":
            safe_mean(all_hd95s),

        "evaluation_level":
            "processed_64x64x64_patches",

        "num_test_patches":
            len(dataloader.dataset),

    }

    return metrics_summary


# ============================================================
# PATIENT-LEVEL RECONSTRUCTED EVALUATION
# ============================================================

def evaluate_reconstructed_test_set(model, dataset, device, results_dir, logger):
    """Evaluate each patient after coordinate-based patch reconstruction."""
    model.eval()
    patient_indices = {}
    for index, sample in enumerate(dataset.samples):
        patient_indices.setdefault(sample[3], []).append(index)

    metric_values = {name: [] for name in ("dice", "iou", "wt", "tc", "et", "hd95")}
    precision_values = []
    recall_values = []
    f1_values = []

    for patient_id, indices in patient_indices.items():
        predictions = []
        masks = []
        coordinates = []

        with torch.inference_mode():
            for dataset_index in indices:
                sample = dataset[dataset_index]
                logits = model(sample["image"].unsqueeze(0).to(device))
                predictions.append(torch.argmax(logits, dim=1).squeeze(0).cpu().numpy())
                masks.append(sample["mask"].numpy())
                coordinates.append(sample["patch_coordinate"].numpy())

        metadata_path = os.path.join(
            dataset.processed_root, dataset.split, patient_id, "metadata.json"
        )
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        prediction_volume = reconstruct_from_metadata(
            np.stack(predictions), np.stack(coordinates), metadata
        )
        mask_volume = reconstruct_from_metadata(
            np.stack(masks), np.stack(coordinates), metadata
        )
        prediction_tensor = torch.from_numpy(prediction_volume).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_volume).unsqueeze(0)

        dice, _ = compute_dice_score(prediction_tensor, mask_tensor)
        iou, _ = compute_iou_score(prediction_tensor, mask_tensor)
        prf = compute_precision_recall_f1(prediction_tensor, mask_tensor)
        regions = compute_brats_regions_metrics(prediction_tensor, mask_tensor)
        hd95 = compute_hausdorff_distance_95(
            prediction_volume == 3, mask_volume == 3,
            voxel_spacing=tuple(metadata.get("spacing", (1.0, 1.0, 1.0))),
        )

        metric_values["dice"].append(dice)
        metric_values["iou"].append(iou)
        metric_values["wt"].append(regions["dice_wt"])
        metric_values["tc"].append(regions["dice_tc"])
        metric_values["et"].append(regions["dice_et"])
        metric_values["hd95"].append(hd95)
        precision_values.append(prf["mean_precision"])
        recall_values.append(prf["mean_recall"])
        f1_values.append(prf["mean_f1"])

        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, f"{patient_id}_prediction_cropped.npy"), prediction_volume)
        np.save(os.path.join(results_dir, f"{patient_id}_ground_truth_cropped.npy"), mask_volume)
        logger.info(f"Reconstructed and evaluated {patient_id} ({len(indices)} patches)")

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "test_mean_dice": mean(metric_values["dice"]),
        "test_mean_iou": mean(metric_values["iou"]),
        "test_mean_precision": mean(precision_values),
        "test_mean_recall": mean(recall_values),
        "test_mean_f1": mean(f1_values),
        "test_dice_wt": mean(metric_values["wt"]),
        "test_dice_tc": mean(metric_values["tc"]),
        "test_dice_et": mean(metric_values["et"]),
        "test_mean_hd95_et": mean(metric_values["hd95"]),
        "evaluation_level": "patient_cropped_volumes",
        "num_test_patients": len(patient_indices),
        "num_test_patches": len(dataset),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 3D Attention U-Net "
            "on the verified BraTS test split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to trained checkpoint."
        ),
    )

    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help=(
            "Path to data/processed."
        ),
    )

    parser.add_argument(
        "--split-root",
        type=str,
        default=None,
        help=(
            "Path to data/splits."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    cfg = get_default_config()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # PATHS
    # ========================================================

    processed_root, split_root = (
        find_data_paths(
            processed_root=args.processed_root,
            split_root=args.split_root,
        )
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = args.checkpoint

    if checkpoint is None:

        checkpoint = os.path.join(
            cfg.train.checkpoint_dir,
            "nrem_best.pth",
        )

    checkpoint = verify_checkpoint(
        checkpoint
    )

    # ========================================================
    # LOGGER
    # ========================================================

    logger = setup_logger(
        log_file=os.path.join(
            cfg.train.log_dir,
            "test.log",
        )
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Attention U-Net 3D TEST EVALUATION"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        f"Checkpoint: {checkpoint}"
    )

    logger.info(
        f"Processed root: {processed_root}"
    )

    logger.info(
        f"Split root: {split_root}"
    )

    # ========================================================
    # VERIFY TEST SPLIT
    # ========================================================

    test_ids = verify_test_split(
        split_root
    )

    # ========================================================
    # DATASET
    # ========================================================

    test_ds = BraTSDataset3D(
        processed_root=processed_root,
        split="test",
        split_file=os.path.join(
            split_root,
            "test.txt",
        ),
        patch_size=PATCH_SIZE,
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # PRINT DATASET INFO
    # ========================================================

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print(
        f"Patients: {len(test_ids)}"
    )

    print(
        f"Patches : {len(test_ds)}"
    )

    print(
        "Patch size:",
        PATCH_SIZE,
    )

    print(
        "Input channels:",
        NUM_MODALITIES,
    )

    print("=" * 70)

    # ========================================================
    # MODEL
    # ========================================================

    model = AttentionUNet3D(
        in_channels=NUM_MODALITIES,
        out_channels=NUM_CLASSES,
        features=[
            16,
            32,
            64,
            128,
            256,
        ],
        dropout=0.1,
        use_transpose=True,
        use_nrem=True,
    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    logger.info(
        "Loading trained checkpoint..."
    )

    load_checkpoint(
        checkpoint,
        model=model,
        device=device,
    )

    logger.info(
        "Successfully loaded trained checkpoint."
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate_reconstructed_test_set(
        model=model,
        dataset=test_ds,
        device=device,
        results_dir=cfg.train.results_dir,
        logger=logger,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Mean Dice : "
        f"{results['test_mean_dice']:.4f}"
    )

    print(
        f"Mean IoU  : "
        f"{results['test_mean_iou']:.4f}"
    )

    print(
        f"WT Dice   : "
        f"{results['test_dice_wt']:.4f}"
    )

    print(
        f"TC Dice   : "
        f"{results['test_dice_tc']:.4f}"
    )

    print(
        f"ET Dice   : "
        f"{results['test_dice_et']:.4f}"
    )

    print(
        f"ET HD95   : "
        f"{results['test_mean_hd95_et']:.2f}"
    )

    print(
        f"Test patches: "
        f"{results['num_test_patches']}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE JSON
    # ========================================================

    os.makedirs(
        cfg.train.results_dir,
        exist_ok=True,
    )

    results_json_path = os.path.join(
        cfg.train.results_dir,
        "test_results.json",
    )

    with open(
        results_json_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    logger.info(
        f"Saved test results to: "
        f"{results_json_path}"
    )

    print()
    print(
        "Test evaluation completed."
    )

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()