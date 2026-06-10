"""Reader-facing constants for MindHier Stage 2.

Update these values when using a different NSD preprocessing pipeline.
"""

SUBJECT_NUM_VOXELS = {
    1: 15724,
    2: 14278,
    3: 15226,
    4: 13153,
    5: 13039,
    6: 17907,
    7: 12682,
    8: 14386,
}

DEFAULT_MULTI_SUBJECTS = [1, 2, 5, 7]
NSD_TRIALS_PER_SESSION = 750
DEFAULT_TEST_SAMPLES = 3000
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
