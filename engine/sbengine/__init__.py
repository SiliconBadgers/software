"""sbengine: analytical accelerator model driven by captured llama.cpp graphs.

Successor to the 2026-09-24 explorer model (experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/engine.js,
Zeb Taylor, llama.cpp commit 308883b). It keeps that model's structure (units, shared L1, HBM, cost proxy)
and changes its equations where our captured data allow a better one. See docs/EQUATIONS.md.

Every figure this package produces is an analytical estimate under stated assumptions
(docs/ASSUMPTIONS.md), not an ASIC/FPGA measurement and not a model-quality evaluation.
"""

VERSION = "0.1.0"
