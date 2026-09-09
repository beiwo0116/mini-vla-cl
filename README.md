# Mini-VLA Continual Skills

Language-conditioned policy (RGB + English instruction → discrete MiniGrid actions) trained with behavior cloning, then evaluated under sequential skill learning. Compare naive fine-tuning vs keeping 2% of old demos. This is a small, trainable stand-in for continual VLA protocols — not a reproduction of a specific VLA paper, and not LIBERO.

## Setup (isolated venv)

```powershell
cd C:\Users\被窝\CursorProject\mini_vla_cl
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not reuse another project's virtualenv.

## Local smoke (fake encoder)

```powershell
.\.venv\Scripts\python.exe -m mini_vla_cl collect --skill goto --n 4 --encoder fake --out results/cache/goto.npz
.\.venv\Scripts\python.exe -m mini_vla_cl collect --skill pickup --n 4 --encoder fake --out results/cache/pickup.npz
.\.venv\Scripts\python.exe -m mini_vla_cl train --method naive --skills goto,pickup --epochs 1 --encoder fake --device cpu --eval-episodes 2
```

## Metrics

- **Current skill success:** success rate of the skill just learned.
- **Average success:** mean success over all skills seen so far.
- **Forgetting:** historical max success of an old skill minus its current success.

## Kaggle (official numbers)

1. Open Internet. Install on CPU, then switch to GPU T4.
2. Put this repo on the machine (`git clone` or Dataset zip).
3. Run `notebooks/kaggle_train.ipynb`.
4. Download `results/` (json, png, checkpoints) before the session ends.

Seed is `42`.

## Boundary

Grid-world language-conditioned control with a frozen CLIP (or fake) encoder. Do not describe this as reproducing OpenVLA, LIBERO, or CLARE.
