# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU GRPO RL and vLLM rollout server helpers, stale engine census, and PPO/value loss hooks."""

import asyncio
import logging
import os
import socket
from typing import Any

import ray
from tensordict import TensorDict

from verl.utils.device import get_device_name, get_resource_name
from verl_hardware_plugin.engines.torchtitan_tpu_utils import select_and_to_padded_tensor, tpu_no_padding_2_padding
from verl_hardware_plugin.utils.tpu_vllm_patch import apply_tpu_vllm_patches

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource or with V1 explicitly disabled."""
    return get_resource_name() == "TPU" or os.environ.get("VLLM_USE_V1") == "0"


def override_vllm_configs_for_tpu(args_or_config: Any) -> None:
    """Enforces VLLM_USE_V1=0 and use_v1=False across dictionary and namespace objects on TPU."""
    os.environ["VLLM_USE_V1"] = "0"
    try:
        import vllm.envs as vllm_envs

        vllm_envs.VLLM_USE_V1 = False
    except Exception:
        pass

    if isinstance(args_or_config, dict):
        args_or_config["use_v1"] = False
        return

    for obj in [args_or_config, getattr(args_or_config, "model_config", None)]:
        if obj is None:
            continue
        for attr in ("use_v1", "_use_v1"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, False)
                except Exception:
                    pass
            if hasattr(obj, "__dict__") and attr in obj.__dict__:
                try:
                    obj.__dict__[attr] = False
                except Exception:
                    pass


def prepare_tpu_server_args(args: dict) -> None:
    """Configures TPU-specific CLI/server arguments and environment variables."""
    if not is_tpu_vllm_run():
        return

    args["enable_sleep_mode"] = False
    args["distributed_executor_backend"] = "external_launcher"
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "0"


def parse_proc_stat_for_zombie_engine(stat_line: str, engine_marker: str = "EngineCor") -> tuple[bool, bool]:
    """Parses a single /proc/<pid>/stat line and returns (is_zombie, is_engine_zombie)."""
    close = stat_line.rfind(")")
    if close == -1:
        return False, False

    fields = stat_line[close + 1 :].split()
    if not fields or fields[0] != "Z":
        return False, False

    opened = stat_line.find("(")
    comm = stat_line[opened + 1 : close] if opened != -1 else ""
    return True, engine_marker in comm


def probe_stale_tpu_engines(self=None) -> dict[str, Any]:
    """Return a zombie-process census for this host (`hostname`, `zombies`, `engines`)."""
    census = {"hostname": socket.gethostname(), "zombies": 0, "engines": 0}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return census

    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                stat_line = handle.read().decode(errors="replace")
        except OSError:
            continue

        is_zombie, is_engine = parse_proc_stat_for_zombie_engine(stat_line)
        if is_zombie:
            census["zombies"] += 1
            if is_engine:
                census["engines"] += 1

    return census


async def get_tpu_server_launch_config(workers):
    """Queries node ID, visible chips, and TPU environment variables from all TPU workers."""
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    os.environ.get("TPU_VISIBLE_CHIPS", "0"),
                )
            )
            for worker in workers
        ]
    )

    worker_tpu_envs = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("TPU_")
                    or k.startswith("TORCH_TPU_")
                    or k
                    in (
                        "CLOUD_TPU_TASK_ID",
                        "CHIPS_PER_HOST",
                        "LIBTPU_INIT_ARGS",
                        "SKIP_JAX_PRECOMPILE",
                        "VLLM_ENABLE_V1_MULTIPROCESSING",
                        "VLLM_DISABLE_COMPILE_CACHE",
                        "VERL_PLATFORM",
                        "XLA_FLAGS",
                    )
                }
            )
            for worker in workers
        ]
    )

    node_id = worker_infos[0][0]
    visible_chips = ",".join([info[1] for info in worker_infos])
    tpu_env_vars = worker_tpu_envs[0] if worker_tpu_envs else {}
    return node_id, visible_chips, tpu_env_vars


def apply_tpu_grpo_hooks() -> None:
    """Applies TPU GRPO hooks (concat-free vLLM RoPE and static-shape ppo_loss/value_loss padding)."""
    apply_tpu_vllm_patches()

    try:
        import verl.workers.utils.losses as verl_losses

        if not getattr(verl_losses, "_verl_tpu_ppo_loss_patched", False):
            _orig_no_padding_2_padding = verl_losses.no_padding_2_padding

            def _patched_no_padding_2_padding(tensor, data: TensorDict):
                if get_device_name() == "tpu" or getattr(tensor, "_tpu_padded_values", None) is not None:
                    return tpu_no_padding_2_padding(tensor, data)
                return _orig_no_padding_2_padding(tensor, data)

            verl_losses.no_padding_2_padding = _patched_no_padding_2_padding
            verl_losses.select_and_to_padded_tensor = select_and_to_padded_tensor
            verl_losses._verl_tpu_ppo_loss_patched = True
    except Exception as e:
        logger.debug("Failed to patch ppo_loss/value_loss for TPU: %s", e)
