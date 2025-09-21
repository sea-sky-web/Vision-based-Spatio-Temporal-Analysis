"""Configuration for Wildtrack v1 dry-run"""

ROOT = '.'
DATA_ROOT = 'Data/Wildtrack'
BATCH_SIZE = 2
NUM_CAMERAS = 7
IMAGE_SIZE = (3, 256, 256)
BEV_SIZE = (1, 128, 128)
DEVICE = 'cpu'


def __repr__():
    return f"WildtrackConfig(DATA_ROOT={DATA_ROOT}, BATCH_SIZE={BATCH_SIZE})"
