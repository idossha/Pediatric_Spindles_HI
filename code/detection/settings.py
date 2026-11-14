"""Detector parameters (the paper's) and the thresholds of the per-file checks."""

DETECT_SF = 100.0  # the paper detected after resampling to 100 Hz
YASA_KWARGS = {"freq_sp": (10.0, 16.0), "duration": (0.5, 2.0), "min_distance": 500,
                   "thresh": {"rel_pow": 0.2, "corr": 0.65, "rms": 1.5}}
BLOCK = 8  # channels per YASA call; YASA treats channels independently, so blocking does not change the output

ALIGN_N, ALIGN_PAD, ALIGN_SEED = 240, 600, 20260927
MIN_ALIGN_RHO, MIN_WITHIN_5PCT = 0.9, 0.8
MAX_END_GAP_S = 300.0
