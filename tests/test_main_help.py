import subprocess
import sys


def test_module_help():
    r = subprocess.run(
        [sys.executable, "-m", "mini_vla_cl", "--help"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "collect" in r.stdout and "train" in r.stdout
