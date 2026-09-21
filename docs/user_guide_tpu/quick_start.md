# Google TPU Quick Start

This page covers verifying the TPU platform, TorchTitan engine, and TPU checkpoint engine, and launching SFT and GRPO RL training jobs on Cloud TPU v6e.

## 1. Select the Platform

```bash
export VERL_PLATFORM=tpu
export PJRT_DEVICE=TPU
```

## 2. Verify Platform, Training Engine, and Checkpoint Engine Resolution

```bash
VERL_PLATFORM=tpu python3 -c '
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.plugin.platform.platform_manager import get_platform
from verl.workers.engine.base import EngineRegistry
import verl_hardware_plugin

p = get_platform()
print("device:      ", p.device_name)
print("vendor:      ", p.vendor_name)
print("backend:     ", p.communication_backend_name())
print("train engine:", EngineRegistry.get_engine_cls("language_model", "torchtitan").__name__)
print("ckpt engine: ", CheckpointEngineRegistry.get("tpu").__name__)
'
```

Expected output:

```text
device:       tpu
vendor:       google
backend:      tpu_dist
train engine: TorchTitanTPUEngineWithLMHead
ckpt engine:  TPUCheckpointEngine
```

## 3. Run SFT Training (`Qwen3-0.6B` on `GSM8K`)

```bash
SMOKE_TEST=1 NNODES_TRAINER=2 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=8 \
  bash scripts/run_sft_qwen3_0_6b_tpu.sh
```

## 4. Run GRPO RL Training (`Qwen3-0.6B` on `GSM8K` across 2 `v6e-8` Slices)

Use [`scripts/run_grpo_qwen3_0_6b_tpu.sh`](../../scripts/run_grpo_qwen3_0_6b_tpu.sh) with Slice 0 running the TorchTitan Actor (`NNODES_TRAINER=2`, `N_CHIPS_TRAINER=4`) and Slice 1 running the vLLM Rollout (`NNODES_ROLLOUT=2`, `N_CHIPS_ROLLOUT=4`):

```bash
# Quick 5-step smoke test
SMOKE_TEST=1 bash scripts/run_grpo_qwen3_0_6b_tpu.sh

# Full 100-step GRPO training run
bash scripts/run_grpo_qwen3_0_6b_tpu.sh
```

## 5. Run the Plugin Test Suite

```bash
pytest tests/ -v
```
