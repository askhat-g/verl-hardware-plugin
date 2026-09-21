# Google TPU Quick Start

This page covers verifying the TPU platform and TorchTitan engine registration and running Supervised Fine-Tuning (SFT) on Cloud TPU v6e.

## 1. Select the Platform

Always select TPU explicitly:

```bash
export VERL_PLATFORM=tpu
export PJRT_DEVICE=TPU
```

## 2. Verify Platform and Engine Resolution

```bash
VERL_PLATFORM=tpu python3 -c '
from verl.plugin.platform.platform_manager import get_platform
from verl.workers.engine.base import EngineRegistry
import verl_hardware_plugin

p = get_platform()
print("device:   ", p.device_name)
print("vendor:   ", p.vendor_name)
print("backend:  ", p.communication_backend_name())
print("ray res:  ", p.ray_resource_name())
print("engine:   ", EngineRegistry.get_engine_cls("language_model", "torchtitan").__name__)
'
```

Expected output:

```text
device:    tpu
vendor:    google
backend:   tpu_dist
ray res:   TPU
engine:    TorchTitanTPUEngineWithLMHead
```

## 3. Run SFT Training (`Qwen3-0.6B` on `GSM8K`)

Use [`scripts/run_sft_qwen3_0_6b_tpu.sh`](../../scripts/run_sft_qwen3_0_6b_tpu.sh) to run Multi-Turn / Instruction SFT with the TorchTitan engine (`engine=torchtitan`) on a TPU v6e-8 slice (2 hosts $\times$ 4 chips/host) or TPU v6e-4 slice:

```bash
# Quick 8-step smoke test on a single-host TPU v6e-4 (1 host x 4 chips)
SMOKE_TEST=1 NNODES_TRAINER=1 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=4 \
  bash scripts/run_sft_qwen3_0_6b_tpu.sh

# Full SFT run on a 2-host TPU v6e-8 slice (2 hosts x 4 chips)
NNODES_TRAINER=2 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=8 \
  bash scripts/run_sft_qwen3_0_6b_tpu.sh
```

## 4. Run the Plugin Test Suite

```bash
pytest tests/test_plugin_registration.py tests/test_tpu_sft_hooks.py -v
```
