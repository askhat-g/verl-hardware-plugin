# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin`.

The plugin registers both the TPU platform (`PlatformTPU`: device metadata, Ray resource
configuration, and the PJRT worker environment) and the TorchTitan TPU training engine
(`TorchTitanTPUEngineWithLMHead` under `verl_hardware_plugin/engines/torchtitan_tpu.py`).

## Directory Structure

```text
verl_hardware_plugin/
├── engines
│   ├── torchtitan_tpu.py             # TorchTitan TPU training engine (FSDP2 / SPMD)
│   └── tpu_utils.py                  # Sequence bucketing and TorchTitan TPU config/input helpers
└── platforms
    └── platform_tpu.py               # TPU platform settings
```

```text
scripts/
└── run_sft_qwen3_0_6b_tpu.sh         # SFT launch script for Qwen3-0.6B on TPU v6e
```

```text
user_guide_tpu/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
└── quick_start.md                    # Selecting the platform and running SFT
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — select the TPU platform, verify registration, and run SFT

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
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
| v6e | 32 GB | `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32` |

v6e is the primary generation this plugin targets. When the generation cannot be determined from
Ray node labels or environment variables, the platform falls back to the v6e HBM figure.

## Related Documentation

- [verl plugin system](../development.md)
