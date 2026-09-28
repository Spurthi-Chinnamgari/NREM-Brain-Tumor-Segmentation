import numpy as np

from .preprocessing.patch_extraction import extract_patches
from .preprocessing.cropping import get_bounding_box, crop_patient
from .preprocessing.normalization import normalize_volume
from .preprocessing.run_preprocessing import build_patient_split

# Patch coverage test on a non-multiple shape.
image = np.ones((4, 70, 130, 65), dtype=np.float32)
seg = np.zeros((70, 130, 65), dtype=np.int16)
seg[60:70, 120:130, 55:65] = 1
patches, masks, coords, info = extract_patches(image, seg)
assert patches.shape[1:] == (4, 64, 64, 64)
assert masks.shape[1:] == (64, 64, 64)
assert len(coords) == patches.shape[0]
assert info["padded_shape"] == [128, 192, 128]
assert info["pad_after"] == [58, 62, 63]
assert np.all(np.isfinite(patches))

# Normalization should never create NaN/Inf.
vol = np.zeros((8, 8, 8), dtype=np.float32)
vol[2:6, 2:6, 2:6] = 10
norm = normalize_volume(vol)
assert np.all(np.isfinite(norm))
assert np.all(norm[vol == 0] == 0)

# Split count for the project's 37-patient case.
ids = [f"P{i:02d}" for i in range(37)]
tr, va, te = build_patient_split(ids)
assert (len(tr), len(va), len(te)) == (26, 6, 5)
assert len(set(tr) | set(va) | set(te)) == 37
assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))

print("PREPROCESSING CONTRACT TEST PASSED")
print("patches:", patches.shape)
print("coordinates:", coords.shape)
print("padded shape:", info["padded_shape"])
print("split:", len(tr), len(va), len(te))
