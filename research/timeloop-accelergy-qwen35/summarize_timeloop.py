"""Check profiled matrix MACs and minimum tensor sizes against Timeloop stats."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "qwen-ffn-2026-10-12"
CASES = {
    "ffn-gate-decode-a-stationary": (6144, 2048, 1),
    "ffn-gate-prefill-a-stationary": (6144, 2048, 128),
    "ffn-gate-prefill-weight-reuse": (6144, 2048, 128),
    "ffn-out-decode-a-stationary": (2048, 6144, 1),
    "ffn-out-prefill-a-stationary": (2048, 6144, 128),
    "ffn-out-prefill-weight-reuse": (2048, 6144, 128),
    "output-head-a-stationary": (248320, 2048, 1),
}


def number(section: str, label: str) -> int:
    found = re.search(rf"(?m)^\s*{re.escape(label)}\s*[:=]\s*(\d+)", section)
    if not found:
        raise ValueError(f"missing {label}")
    return int(found.group(1))


def dataspace(section: str, name: str) -> str:
    found = re.search(rf"(?ms)^    {name}:\n(.*?)(?=^    [ABC]:|^Networks|\Z)", section)
    if not found:
        raise ValueError(f"missing dataspace {name}")
    return found.group(1)


print("case,MACs,cycles,utilization_pct,A_reads_8bit_words,B_reads_8bit_words,C_writes_model_words,GB_peak_model_words")
for name, (m, k, n) in CASES.items():
    stats = (RESULTS / name / "timeloop-model.stats.txt").read_text()
    dram = stats.split("=== MainMemory ===", 1)[1].split("Networks", 1)[0]
    gb = stats.split("=== GlobalBuffer ===", 1)[1].split("=== MainMemory ===", 1)[0]
    summary = stats.split("Summary Stats", 1)[1]
    macs = number(summary, "Computes")
    cycles = number(summary, "Cycles")
    util = re.search(r"Utilization:\s+([\d.]+)%", summary)
    if not util:
        raise ValueError(f"missing utilization in {name}")
    a = number(dataspace(dram, "A"), "Scalar reads (per-instance)")
    b = number(dataspace(dram, "B"), "Scalar reads (per-instance)")
    c = number(dataspace(dram, "C"), "Scalar updates (per-instance)")
    used = sum(int(x) for x in re.findall(r"Utilized capacity\s*:\s*(\d+)", gb))
    assert macs == m * k * n, name
    assert a >= m * k and b >= k * n and c >= m * n, name
    assert used <= 2_097_152, name  # provisional 2 MiB, 8-bit model words
    assert cycles * 64 == macs, name  # Timeloop's ideal 64-MAC/cycle result
    print(f"{name},{macs},{cycles},{util.group(1)},{a},{b},{c},{used}")
