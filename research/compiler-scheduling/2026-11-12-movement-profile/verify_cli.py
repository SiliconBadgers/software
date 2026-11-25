"""Check that one movement CLI invocation per shape matches the sweep table."""

import json
from pathlib import Path
import subprocess
import sys
import time

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
CAPTURES = ROOT / "results/2026-09-27_1839_windows-2/graphs/pp128-fa-on"
OUTPUT = ROOT / "build/compiler/movement-profile/2026-11-12"
RAW = OUTPUT / "cli"


def main():
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    verified = []
    for row in summary["rows"]:
        phase, tensor_id = row["phase"], row["sample_tensor_id"]
        capture = CAPTURES / f"{phase}.json.gz"
        command = [sys.executable, "-m", "sbengine.codegen.movement", str(capture),
                   "--phase", phase, "--tensor-id", str(tensor_id)]
        print(f"{phase} {row['m']}x{row['n']}x{row['k']} tensor {tensor_id}",
              file=sys.stderr, flush=True)
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT / "engine", text=True,
                                capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"CLI failed for {phase} tensor {tensor_id}: {result.stderr}")
        output = json.loads(result.stdout)
        expected = {
            "sram_used_bytes": row["sram_used_bytes"],
            "matrix_commands": row["matrix_commands"],
            "transfer_commands": row["transfer_commands"],
            "hbm_to_sram_bytes": row["hbm_to_sram_bytes"],
            "sram_to_hbm_bytes": row["sram_to_hbm_bytes"],
        }
        actual = {
            "sram_used_bytes": output["sram_used_bytes"],
            "matrix_commands": output["counts"]["matrix"],
            "transfer_commands": output["counts"]["transfers"],
            "hbm_to_sram_bytes": output["bytes"]["hbm_to_sram"],
            "sram_to_hbm_bytes": output["bytes"]["sram_to_hbm"],
        }
        if actual != expected or not output["check"]["passed"]:
            raise RuntimeError(f"CLI disagrees for {phase} tensor {tensor_id}: "
                               f"expected={expected}, actual={actual}, check={output['check']}")
        (RAW / f"{phase}-tensor-{tensor_id}.json").write_text(result.stdout, encoding="utf-8")
        verified.append({"phase": phase, "tensor_id": tensor_id,
                         "shape": [row["m"], row["n"], row["k"]],
                         "matches_sweep": True,
                         "checker_passed": True,
                         "elapsed_seconds": round(time.monotonic() - start, 2)})
        print(f"  matched in {verified[-1]['elapsed_seconds']:.2f}s",
              file=sys.stderr, flush=True)
    (OUTPUT / "cli_verification.json").write_text(
        json.dumps({"verified_shapes": len(verified), "rows": verified}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Verified {len(verified)} CLI runs against the sweep table")


if __name__ == "__main__":
    main()
