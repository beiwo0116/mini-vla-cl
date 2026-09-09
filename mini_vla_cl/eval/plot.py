import csv
from pathlib import Path


def _goto_curve(result: dict) -> list[float]:
    return [row["success"]["goto"] for row in result["after_skill"]]


def plot_week4(results_by_method: dict[str, dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs = ["goto", "pickup"]
    fig, ax = plt.subplots()
    for name, result in results_by_method.items():
        ys = _goto_curve(result)
        ax.plot(xs[: len(ys)], ys, marker="o", label=name)
    ax.set_title("GoTo success after each skill (naive vs 2% replay)")
    ax.set_xlabel("After learning")
    ax.set_ylabel("GoTo success rate")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_week4_table(results_by_method: dict[str, dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "avg_after_task2",
        "goto_forget_after_task2",
        "goto_success_after_task2",
        "pickup_success_after_task2",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, result in results_by_method.items():
            last = result["after_skill"][-1]
            writer.writerow(
                {
                    "method": method,
                    "avg_after_task2": last["average"],
                    "goto_forget_after_task2": last["forgetting_goto"],
                    "goto_success_after_task2": last["success"].get("goto"),
                    "pickup_success_after_task2": last["success"].get("pickup"),
                }
            )
