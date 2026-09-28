import os


# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "brats2023",
)

PROCESSED_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
)


# ======================================================
# Preprocessing
# ======================================================

SPLIT_SEED = 42
DEFAULT_PATIENT_PERCENTAGE = 100

PATCH_SIZE = (64, 64, 64)


# ======================================================
# Dataset Split
# ======================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ======================================================
# MRI
# ======================================================

MODALITIES = [
    "t1n",
    "t1c",
    "t2w",
    "t2f",
]

SEGMENTATION = "seg"