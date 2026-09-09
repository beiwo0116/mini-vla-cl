from pathlib import Path

from mini_vla_cl.eval.plot import plot_week4, write_week4_table


def _fake():
    def pack(g1, g2, p2):
        return {
            "method": "x",
            "after_skill": [
                {
                    "just_learned": "goto",
                    "success": {"goto": g1},
                    "average": g1,
                    "forgetting_goto": 0.0,
                },
                {
                    "just_learned": "pickup",
                    "success": {"goto": g2, "pickup": p2},
                    "average": (g2 + p2) / 2,
                    "forgetting_goto": g1 - g2,
                },
            ],
        }

    return {"naive": pack(0.9, 0.3, 0.8), "replay": pack(0.9, 0.75, 0.8)}


def test_plot_and_csv(tmp_path: Path):
    data = _fake()
    png = tmp_path / "week4.png"
    csv = tmp_path / "week4.csv"
    plot_week4(data, png)
    write_week4_table(data, csv)
    assert png.exists() and png.stat().st_size > 0
    text = csv.read_text(encoding="utf-8")
    assert "naive" in text and "replay" in text
    assert "goto_forget_after_task2" in text
