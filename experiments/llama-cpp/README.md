# experiments/llama-cpp

Recorded baseline in 2026-09-22/. Put each new run in a separate dated or named directory with a manifest, commands, source/model hashes, host/backend, inputs, results and limitations.

See [the current assignment](../../docs/START-HERE.md).

## Qwen3.5-2B graph capture and accelerator explorer

[2026-09-24-qwen35-2b/](2026-09-24-qwen35-2b/README.md) contains BF16 CPU
prefill/decode graph captures for 128- and 512-token prompts, reproduction
scripts, saved design configurations, and an interactive accelerator explorer.
Open its `index.html` locally to use the explorer. Its accelerator timing and
resource costs are uncalibrated analytical estimates; see the experiment's
limitations before using them in compute-unit recommendations.
