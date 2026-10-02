"""Launch selected diagnostic ranks under py-spy (parent attach needs no ptrace capability)."""
import os
from pathlib import Path
import shutil
import sys


def main():
    rank = os.environ.get("RANK", "0")
    command = [sys.executable, "train.py", *sys.argv[1:]]
    selected = os.environ.get("ROBOMETER_PROFILE_NATIVE_RANKS", "0").split(",")
    if rank in selected:
        executable = os.environ.get("PY_SPY_BIN") or shutil.which("py-spy")
        if not executable:
            raise RuntimeError("Set PY_SPY_BIN to a py-spy executable for native CPU profiling")
        folder = Path(os.environ.get("ROBOMETER_NATIVE_PROFILE_DIR", "logs/native_profile"))
        folder.mkdir(parents=True, exist_ok=True)
        command = [executable, "record", "--native", "--rate",
                   os.environ.get("ROBOMETER_PROFILE_NATIVE_RATE", "5"), "--format", "speedscope",
                   "--output", str(folder / f"native_rank_{rank}.json"), "--", *command]
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
