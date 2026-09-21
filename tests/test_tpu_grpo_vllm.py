# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for TPU GRPO vLLM patches, multi-slice PlatformTPU isolation, and RoPE equivalence."""

import os
from unittest import mock

import torch

from verl_hardware_plugin.platforms.platform_tpu import (
    ROLLOUT_BASE_PORT,
    TRAINER_BASE_PORT,
    PlatformTPU,
)
from verl_hardware_plugin.utils.tpu_grpo_hooks import (
    override_vllm_configs_for_tpu,
    parse_proc_stat_for_zombie_engine,
    prepare_tpu_server_args,
)
from verl_hardware_plugin.utils.tpu_vllm_patch import _tpu_rotate_neox, _tpu_widen


def test_tpu_rotate_neox_and_widen_bitwise_identical():
    """Verify concat-free _tpu_rotate_neox and _tpu_widen match standard torch.cat implementations."""
    x = torch.randn(2, 4, 8, 64, dtype=torch.bfloat16)
    x1 = x[..., :32]
    x2 = x[..., 32:]
    expected_rot = torch.cat((-x2, x1), dim=-1)
    actual_rot = _tpu_rotate_neox(x)
    assert torch.equal(actual_rot, expected_rot)

    cos = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    expected_widen = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    actual_widen = _tpu_widen(cos, torch.bfloat16)
    assert torch.equal(actual_widen, expected_widen)


def test_platform_tpu_rollout_and_trainer_port_separation(monkeypatch):
    """Rollout and trainer worker groups must use non-overlapping slicebuilder port ranges."""
    import ray

    platform = PlatformTPU()
    monkeypatch.setattr(ray, "nodes", lambda: [{"NodeID": "n1", "NodeManagerAddress": "10.0.0.1", "Alive": True}])
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.0.0.1")

    class _FakePG:
        id = "pg1"

    monkeypatch.setattr(
        ray._private.state.state,
        "placement_group_table",
        lambda pg_id: {"state": "CREATED", "bundles_to_node_id": {0: "n1", 1: "n1", 2: "n1", 3: "n1"}},
        raising=False,
    )

    trainer_env = platform.get_tpu_env_vars(
        rank=1, world_size=4, local_rank=1, local_world_size=4, name_prefix="actor_wg", pgs=[_FakePG()]
    )
    rollout_env = platform.get_tpu_env_vars(
        rank=1, world_size=4, local_rank=1, local_world_size=4, name_prefix="rollout_wg", pgs=[_FakePG()]
    )

    assert trainer_env["TPU_PROCESS_PORT"] == str(TRAINER_BASE_PORT + 1)
    assert rollout_env["TPU_PROCESS_PORT"] == str(ROLLOUT_BASE_PORT + 1)
    assert rollout_env["SKIP_JAX_PRECOMPILE"] == "1"
    assert rollout_env["TPU_MULTIHOST_BACKEND"] == "ray"


def test_platform_tpu_placement_bundle_and_rollout_env_vars():
    platform = PlatformTPU()

    trainer_bundle = {"CPU": 1}
    platform.configure_placement_group_bundle(trainer_bundle, True, "TPU", "actor_pool", "tpu-group-0")
    assert trainer_bundle["TPU"] == 1
    assert trainer_bundle["tpu-group-0"] == 1e-4

    rollout_bundle = {"CPU": 1}
    platform.configure_placement_group_bundle(rollout_bundle, True, "TPU", "rollout_pool", "tpu-group-1")
    assert "TPU" not in rollout_bundle
    assert rollout_bundle["tpu-group-1"] == 1e-4

    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false"}, clear=True):
        assert platform.rollout_env_vars() == {"LIBTPU_INIT_ARGS": "--xla_tpu_use_enhanced_launch_barrier=false"}


def test_zombie_engine_parser_and_vllm_overrides():
    is_z, is_eng = parse_proc_stat_for_zombie_engine("12345 (VLLM::EngineCor) Z 1 12345 12345 0 -1")
    assert is_z is True
    assert is_eng is True

    is_z2, is_eng2 = parse_proc_stat_for_zombie_engine("12346 (python3) S 1 12346 12346 0 -1")
    assert is_z2 is False
    assert is_eng2 is False

    cfg = {"use_v1": True}
    override_vllm_configs_for_tpu(cfg)
    assert cfg["use_v1"] is False

    server_args = {}
    with mock.patch.dict(os.environ, {"VLLM_USE_V1": "0"}):
        prepare_tpu_server_args(server_args)
        assert server_args["enable_sleep_mode"] is False
        assert server_args["distributed_executor_backend"] == "external_launcher"
