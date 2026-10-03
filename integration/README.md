# Device integration probes

`firmware/` contains the RV32IM mailbox/MMIO polling probe used with both Ibex
and CV32E40P. `llama_probe.cpp` contains the ggml numerical comparison and
Qwen3.5-2B post-execution metadata observer. Tensor math remains on the host CPU;
this is not an accelerator backend or a hardware inference benchmark.

The [SoC workspace](https://github.com/SiliconBadgers/soc) pins
this repository, supplies the simulated device and builds these sources:

```sh
./scripts/workspace.sh cores
./scripts/workspace.sh graph-check
./scripts/workspace.sh model-check
```

Run those commands in SoC after following its SETUP.md. The model check
requires an explicit model download. Shared wire/register behavior remains a
proposal; this firmware does not establish a final ABI or CPU choice.
