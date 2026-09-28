# Google TPU Quick Start

## 1. Select the Platform

Always select TPU explicitly:

```bash
export VERL_PLATFORM=tpu
export PJRT_DEVICE=TPU
```

Auto-detection also works on a TPU host, because each registered platform is probed in turn and only
the TPU probe succeeds. But that outcome depends on registration order, which is not a stable
contract. `VERL_PLATFORM=tpu` is the supported way to select TPU.

## 2. Verify Platform Resolution

```bash
python3 -c '
from verl.plugin.platform.platform_manager import get_platform
p = get_platform()
print("device:   ", p.device_name)
print("vendor:   ", p.vendor_name)
print("backend:  ", p.communication_backend_name())
print("ray res:  ", p.ray_resource_name())
print("ipc:      ", p.is_ipc_supported())
print("colocate: ", p.supports_colocated_worker_groups())
'
```

Expected output:

```text
device:    tpu
vendor:    google
backend:   tpu_dist
ray res:   TPU
ipc:       False
colocate:  False
```

## 3. Verify Ray Resource Requests

The platform requests chips as a custom Ray resource named `TPU`, not as `num_gpus`:

```bash
python3 -c '
from verl.plugin.platform.platform_manager import get_platform
print(get_platform().ray_resource_options(4))
'
```

Expected output: `{'resources': {'TPU': 4}}`

Your Ray cluster must advertise a `TPU` resource for scheduling to succeed. On KubeRay this comes
from the TPU node pool's resource annotations.

## 4. Verify TorchTitan TPU Engine

```bash
pytest tests/test_plugin_registration.py -k tpu -v
pytest tests/test_tpu_engine.py -v
```

Expected: all TPU registration and engine utility cases pass on any host, with or without a TPU attached.

## 5. Run Supervised Fine-Tuning (SFT) on TPU v6e

With a KubeRay TPU v6e-8 cluster running (2 hosts $\times$ 4 chips = 8 TPU chips), launch the GSM8K SFT script:

```bash
# Quick 8-step smoke test with validation at steps 4 and 8
SMOKE_TEST=1 bash scripts/run_sft_qwen3_0_6b_tpu.sh

# Full 20-step training run
bash scripts/run_sft_qwen3_0_6b_tpu.sh
```

Or submit via `ray job submit` to a remote KubeRay head service:

```bash
export RAY_ADDRESS="http://localhost:23333"

ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash -c 'SMOKE_TEST=1 bash scripts/run_sft_qwen3_0_6b_tpu.sh'
```

### Key SFT Configuration Notes

- **Sequence Bucketing (`model.use_remove_padding=True`, `data.pad_mode=no_padding`)**: Packed 1D sequences are padded on CPU to multiples of `VERL_TPU_SEQ_BUCKET_SIZE` (default `256`) before H2D transfer, preventing dynamic-shape XLA recompilation while isolating packed documents via a 4D block-diagonal causal mask in `TPUVarlenAttention`.
- **Pure FSDP2 (`engine.tensor_parallel_size=1`, `engine.data_parallel_shard_size=<total_chips>`)**: Shard across all TPU chips in the slice using FSDP2 (`data_parallel_shard_size=8` on `v6e-8`, or `NNODES_TRAINER=1 N_CHIPS_TRAINER=4 DATA_PARALLEL_SHARD_SIZE=4` on `v6e-4`).
