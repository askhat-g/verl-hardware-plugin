# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for TPUCheckpointEngine, TPUWeightRegistry, and worker weight sharding/fusion."""

import torch
import torch.nn as nn

from verl_hardware_plugin.engines.tpu_checkpoint_engine import (
    TPUCheckpointEngine,
    get_clean_name,
    get_layer_group,
    load_weights_on_worker,
)
from verl_hardware_plugin.engines.tpu_weight_registry import TPUWeightRegistryState


def test_tpu_checkpoint_engine_registered():
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("tpu") is TPUCheckpointEngine


def test_tpu_weight_registry_write_eviction():
    """Verify write-based eviction: step 0 of a new job evicts step 5 from a previous job."""
    reg = TPUWeightRegistryState()
    reg.set_weights(5, ["old_ref"])
    assert reg.get_weights(5) == ["old_ref"]

    # New job writes step 0 -> step 5 must be evicted
    reg.set_weights(0, ["new_ref"])
    assert reg.get_weights(0) == ["new_ref"]
    assert reg.get_weights(5) is None

    reg.clear()
    assert reg.get_weights(0) is None


def test_clean_name_and_layer_group():
    raw = "_fsdp_wrapped_module._checkpoint_wrapped_module.module.model.layers.12.self_attn.q_proj.weight"
    assert get_clean_name(raw) == "model.layers.12.self_attn.q_proj.weight"
    assert get_layer_group(raw) == "layers.12"
    assert get_layer_group("model.tok_embeddings.weight") == "embeddings"
    assert get_layer_group("model.norm.weight") == "output"


class _DummyRolloutAttention(nn.Module):
    def __init__(self, tp_size: int = 2, flipped: bool = False):
        super().__init__()
        # In vLLM, qkv_proj is a QKVParallelLinear module whose weight key is `...self_attn.qkv_proj.weight`
        self.qkv_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.qkv_proj.weight.data.zero_()
        self.qkv_proj.tp_size = tp_size
        self.qkv_proj.num_kv_head_replicas = 1
        self.qkv_proj._tpu_weight_flipped = flipped


class _DummyRolloutModel(nn.Module):
    def __init__(self, tp_size: int = 2, flipped: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"self_attn": _DummyRolloutAttention(tp_size, flipped)})])
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)


def test_pack_and_load_weights_with_qkv_fusion_and_tp_sharding():
    q_w = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    k_w = (torch.arange(8, dtype=torch.bfloat16) + 100).reshape(2, 4)
    v_w = (torch.arange(8, dtype=torch.bfloat16) + 200).reshape(2, 4)
    emb_w = (torch.arange(32, dtype=torch.bfloat16) + 300).reshape(8, 4)

    def weight_gen():
        yield ("_fsdp_wrapped_module.layers.0.self_attn.q_proj.weight", q_w)
        yield ("_fsdp_wrapped_module.layers.0.self_attn.k_proj.weight", k_w)
        yield ("_fsdp_wrapped_module.layers.0.self_attn.v_proj.weight", v_w)
        yield ("_fsdp_wrapped_module.tok_embeddings.weight", emb_w)

    state_dict = TPUCheckpointEngine.pack_weights_to_grouped_dict(weight_gen())
    assert "grouped" in state_dict

    # Load onto rank 0 and rank 1 of a 2-way TP rollout model
    model_r0 = _DummyRolloutModel(tp_size=2, flipped=False)
    model_r1 = _DummyRolloutModel(tp_size=2, flipped=False)

    keys_r0 = load_weights_on_worker(model_r0, state_dict, rank=0, target_device="cpu")
    keys_r1 = load_weights_on_worker(model_r1, state_dict, rank=1, target_device="cpu")
    assert keys_r0 > 0
    assert keys_r1 > 0

    # Rank 0 should have q_w[:2], k_w[:1], v_w[:1] concatenated along dim 0
    expected_r0_qkv = torch.cat([q_w[:2], k_w[:1], v_w[:1]], dim=0)
    expected_r1_qkv = torch.cat([q_w[2:], k_w[1:], v_w[1:]], dim=0)

    assert torch.equal(model_r0.layers[0]["self_attn"].qkv_proj.weight.data, expected_r0_qkv)
    assert torch.equal(model_r1.layers[0]["self_attn"].qkv_proj.weight.data, expected_r1_qkv)
    # tok_embeddings should also populate lm_head.weight
    assert torch.equal(model_r0.lm_head.weight.data, emb_w)


def test_streaming_weight_loader_cross_bucket_qkv_fusion():
    from verl_hardware_plugin.engines.tpu_checkpoint_engine import TPUStreamingWeightLoader

    q_w = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    k_w = (torch.arange(8, dtype=torch.bfloat16) + 100).reshape(2, 4)
    v_w = (torch.arange(8, dtype=torch.bfloat16) + 200).reshape(2, 4)
    emb_w = (torch.arange(32, dtype=torch.bfloat16) + 300).reshape(8, 4)

    model_r0 = _DummyRolloutModel(tp_size=2, flipped=False)
    loader = TPUStreamingWeightLoader(model_r0, rank=0, target_device="cpu")

    # Bucket 0 contains only q_proj and k_proj (v_proj is in Bucket 1)
    loaded_b0 = loader.load_bucket(
        [
            ("_fsdp_wrapped_module.layers.0.self_attn.q_proj.weight", q_w),
            ("_fsdp_wrapped_module.layers.0.self_attn.k_proj.weight", k_w),
        ]
    )
    assert loaded_b0 == 0
    assert len(loader.pending_raw_tensors) == 2

    # Bucket 1 delivers v_proj and tok_embeddings -> completes qkv_proj fusion
    loaded_b1 = loader.load_bucket(
        [
            ("_fsdp_wrapped_module.layers.0.self_attn.v_proj.weight", v_w),
            ("_fsdp_wrapped_module.tok_embeddings.weight", emb_w),
        ]
    )
    assert loaded_b1 == 2
    assert len(loader.pending_raw_tensors) == 0
    loader.finalize()

    expected_r0_qkv = torch.cat([q_w[:2], k_w[:1], v_w[:1]], dim=0)
    assert torch.equal(model_r0.layers[0]["self_attn"].qkv_proj.weight.data, expected_r0_qkv)
    assert torch.equal(model_r0.lm_head.weight.data, emb_w)


def test_tpu_weight_registry_bucket_acks_and_topology():
    import asyncio

    async def _run():
        reg = TPUWeightRegistryState()
        await reg.set_bucket(
            step=1, bucket_idx=0, bucket_ref_list=["ref0"], bucket_meta={"k": 1}, is_last=True, num_receivers=2
        )
        entry = await reg.get_bucket(step=1, bucket_idx=0)
        assert entry == (["ref0"], {"k": 1}, True)

        await reg.ack_bucket(step=1, bucket_idx=0)
        assert (1, 0) in reg.buckets
        await reg.ack_bucket(step=1, bucket_idx=0)
        await reg.wait_bucket_acks(step=1, bucket_idx=0)
        assert (1, 0) not in reg.buckets

    asyncio.run(_run())

    actor_kw, rollout_kw = TPUCheckpointEngine.build_topology(
        actor_wg_world_size=4,
        rollout_world_size=2,
        metadata=[
            {"is_master": True, "sync_round": 3},
            {"is_master": False},
            {"is_master": False},
            {"is_master": False},
        ],
    )
    assert actor_kw["rank"] == [0, None, None, None]
    assert rollout_kw["rank"] == [1, 2]
    assert rollout_kw["world_size"] == [3, 3]
