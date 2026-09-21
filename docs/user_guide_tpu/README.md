# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin` for both **Supervised Fine-Tuning (SFT)** and **Group Relative Policy Optimization (GRPO) RL** workloads on Cloud TPU v6e slices.

The plugin registers:
- **`PlatformTPU`**: Device metadata, multi-slice Ray placement group isolation (`ROLLOUT_BASE_PORT=8070` vs `TRAINER_BASE_PORT=8471`), and PJRT worker slice environment.
- **`TorchTitanTPUEngineWithLMHead`**: TorchTitan training engine (`engine=torchtitan` / `actor_rollout_ref.actor.strategy=torchtitan`) with static 256-token sequence bucketing (`VERL_TPU_SEQ_BUCKET_SIZE=256`) and pure FSDP2 support on `torch_tpu`.
- **`TPUCheckpointEngine`**: High-speed weight synchronization (`actor_rollout_ref.rollout.checkpoint_engine.backend=tpu`) via `TPUWeightRegistry` and host-side TP sharding/QKV fusion for vLLM TPU rollout workers.

## Directory Structure

```text
verl_hardware_plugin/
├── engines
│   ├── torchtitan_tpu.py             # TPU TorchTitan training engine
│   ├── torchtitan_tpu_utils.py       # Static sequence bucketing & TPU attention helpers
│   ├── tpu_checkpoint_engine.py      # TPU weight sync checkpoint engine
│   └── tpu_weight_registry.py        # Detached Ray actor holding weight references
├── platforms
│   ├── platform_tpu.py               # TPU platform & multi-slice mesh settings
│   └── platform_tpu_workarounds.py   # Multi-host metric aggregation & Ray worker helpers
└── utils
    ├── tpu_grpo_hooks.py             # vLLM rollout server helpers & PPO loss hooks
    ├── tpu_sft_hooks.py              # TPU SFT loss unpacking & metric hooks
    └── tpu_vllm_patch.py             # Concat-free vLLM RoPE & compile cache patches
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — verify the platform/engines and run SFT or GRPO RL training

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
| Training engine | `TorchTitanTPUEngineWithLMHead` (`torchtitan`) |
| Checkpoint engine | `TPUCheckpointEngine` (`tpu`) |
| Rollout engine | `vLLM` (`vllm-torchtpu`) |
| Communication backend | `tpu_dist` (registered by `torch_tpu`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` |
| Ray resource name | `TPU` |
| IPC support | No (uses Ray Object Store via `TPUWeightRegistry`) |
| Colocated worker groups | Not supported — separate slices for Trainer and Rollout |
