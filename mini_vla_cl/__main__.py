from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_vla_cl.config import (
    BATCH_SIZE,
    CACHE_DIR,
    CKPT_DIR,
    N_EPOCHS,
    N_EVAL_EPISODES,
    RESULTS_DIR,
    SEED,
)
from mini_vla_cl.data.cache import embed_transitions, load_embedded, save_embedded
from mini_vla_cl.env.expert import collect_skill_demos
from mini_vla_cl.env.skills import SKILLS
from mini_vla_cl.eval.plot import plot_week4, write_week4_table
from mini_vla_cl.model.encoder import ClipEncoder, FakeEncoder


def _encoder(name: str, device: str):
    if name == "fake":
        return FakeEncoder()
    if name == "clip":
        return ClipEncoder(device=device)
    raise ValueError(f"unknown encoder: {name}")


def cmd_collect(args: argparse.Namespace) -> None:
    n = args.n if args.n is not None else SKILLS[args.skill].n_train_episodes
    raw = collect_skill_demos(args.skill, n_episodes=n, start_seed=SEED)
    enc = _encoder(args.encoder, args.device)
    rows = embed_transitions(raw, enc)
    out = Path(args.out) if args.out else CACHE_DIR / f"{args.skill}.npz"
    save_embedded(out, rows)
    print(f"wrote {len(rows)} transitions to {out}")


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from mini_vla_cl.train.sequential import run_method

    skill_ids = [s.strip() for s in args.skills.split(",") if s.strip()]
    enc = _encoder(args.encoder, args.device)
    demos = {}
    for sid in skill_ids:
        path = CACHE_DIR / f"{sid}.npz"
        demos[sid] = load_embedded(path)
    result = run_method(
        method=args.method,
        skill_ids=skill_ids,
        demos=demos,
        encoder=enc,
        n_epochs=args.epochs,
        batch_size=BATCH_SIZE,
        device=args.device,
        eval_episodes=args.eval_episodes,
        seed=SEED,
    )
    policy = result.pop("policy")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{args.method}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    ckpt = CKPT_DIR / f"{args.method}.pt"
    torch.save(policy.state_dict(), ckpt)
    print(f"wrote {json_path}")


def cmd_plot(args: argparse.Namespace) -> None:
    naive = json.loads(Path(args.naive).read_text(encoding="utf-8"))
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    data = {"naive": naive, "replay": replay}
    out = Path(args.out)
    plot_week4(data, out)
    write_week4_table(data, out.with_suffix(".csv"))
    print(f"wrote {out}")


def cmd_eval(_args: argparse.Namespace) -> None:
    print("eval is performed inside train; use plot to render week-4 figures")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mini_vla_cl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--skill", required=True)
    p_collect.add_argument("--n", type=int, default=None)
    p_collect.add_argument("--out", default=None)
    p_collect.add_argument("--encoder", choices=("fake", "clip"), default="fake")
    p_collect.add_argument("--device", default="cpu")
    p_collect.set_defaults(func=cmd_collect)

    p_train = sub.add_parser("train")
    p_train.add_argument("--method", choices=("naive", "replay"), required=True)
    p_train.add_argument("--skills", default="goto,pickup")
    p_train.add_argument("--epochs", type=int, default=N_EPOCHS)
    p_train.add_argument("--encoder", choices=("fake", "clip"), default="fake")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--eval-episodes", type=int, default=N_EVAL_EPISODES)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval")
    p_eval.set_defaults(func=cmd_eval)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--naive", required=True)
    p_plot.add_argument("--replay", required=True)
    p_plot.add_argument("--out", default=str(RESULTS_DIR / "week4.png"))
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
