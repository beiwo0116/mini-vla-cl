from pathlib import Path

SEED = 42
REPLAY_KEEP_FRAC = 0.02
N_EVAL_EPISODES = 50
N_ACTIONS = 7
EMBED_DIM = 512
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
CKPT_DIR = RESULTS_DIR / "ckpts"
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
MAX_EPISODE_STEPS = 240
