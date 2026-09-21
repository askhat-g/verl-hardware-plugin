# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin`.

The plugin registers:
- **`PlatformTPU`**: Device metadata, Ray resource configuration, and PJRT worker slice environment.
- **`TorchTitanTPUEngineWithLMHead`**: TorchTitan training engine (`engine=torchtitan`) with static 256-token sequence bucketing (`VERL_TPU_SEQ_BUCKET_SIZE=256`) and pure FSDP2 support on `torch_tpu`.

## Directory Structure

```text
verl_hardware_plugin/
├── engines
│   ├── torchtitan_tpu.py             # TPU TorchTitan training engine
│   └── torchtitan_tpu_utils.py       # Static sequence bucketing & TPU attention helpers
├── platforms
│   ├── platform_tpu.py               # TPU platform settings
│   └── platform_tpu_workarounds.py   # Multi-host metric aggregation & Ray worker helpers
└── utils
    └── tpu_sft_hooks.py              # TPU SFT loss unpacking & runtime hooks
```

```text
user_guide_tpu/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
└── quick_start.md                    # Selecting the platform and running SFT training
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — verify the platform/engine and run an SFT training job

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
| Training engine | `TorchTitanTPUEngineWithLMHead` (`engine=torchtitan`) |
| Communication backend | `tpu_dist` (registered by `torch_tpu`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` (see note below) |
| Ray resource name | `TPU` |
| IPC support | No |
| Colocated worker groups | Not supported — a chip belongs to a single process |

### Why the visibility env var is `CUDA_VISIBLE_DEVICES`

This is intentional. verl assigns `os.environ[<this key>]` when launching vLLM servers. Pointing it
at `TPU_VISIBLE_CHIPS` would overwrite the per-worker chip index that the platform sets through
`get_worker_env_vars()` and reads back through `ray_local_rank_override()`, breaking rank mapping.
The two variables serve different purposes and must not be merged.

## Chip Support

| Generation | HBM per chip | Topologies with a built-in mapping |
|------------|--------------|------------------------------------|
| v6e | 32 GB | `v6e-4`, `v6e-8`, `v6e-32` |

## Related Documentation

- [verl plugin system](../development.md)
