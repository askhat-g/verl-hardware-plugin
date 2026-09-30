# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import os
from unittest import mock

import torch


def test_bucket_length_and_env_override():
    from verl_hardware_plugin.engines.tpu_utils import bucket_length, get_tpu_seq_bucket_size

    assert get_tpu_seq_bucket_size() == 256
    assert bucket_length(1) == 256
    assert bucket_length(256) == 256
    assert bucket_length(257) == 512
    assert bucket_length(70, bucket_size=64) == 128

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "64"}):
        assert get_tpu_seq_bucket_size() == 64
        assert bucket_length(65) == 128


def test_unwrap_metadata():
    from verl_hardware_plugin.engines.tpu_utils import unwrap_metadata

    assert unwrap_metadata([torch.tensor(3.5)]) == 3.5
    assert unwrap_metadata((True, False)) is True
    assert unwrap_metadata("flex") == "flex"


def test_pad_packed_inputs_for_tpu_builds_4d_document_causal_mask():
    from tensordict import TensorDict

    from verl_hardware_plugin.engines.tpu_utils import pad_packed_inputs_for_tpu

    # Two packed documents of lengths 3 and 2 -> orig_seq_len = 5
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        _, _, _, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
        )

    assert orig_seq_len == 5
    assert attention_masks.shape == (1, 1, 8, 8)
    assert attention_masks.dtype == torch.bool
    # Document 0 (tokens 0..2) attends causally within [0..2] and not to document 1 (tokens 3..4)
    assert attention_masks[0, 0, 2, 0].item() is True
    assert attention_masks[0, 0, 0, 2].item() is False
    assert attention_masks[0, 0, 3, 2].item() is False
    assert attention_masks[0, 0, 4, 3].item() is True
    # Padded tail (tokens 5..7) has self-attention only (no cross-token attention)
    assert attention_masks[0, 0, 6, 6].item() is True
    assert attention_masks[0, 0, 6, 5].item() is False


def test_replace_varlen_attention_with_tpu_attention():
    from verl_hardware_plugin.engines.tpu_utils import (
        TPUVarlenAttention,
        replace_varlen_attention_with_tpu_attention,
    )

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = torch.nn.Identity()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert replace_varlen_attention_with_tpu_attention([model]) == 2
    assert isinstance(model[0].inner_attention, TPUVarlenAttention)
    assert isinstance(model[1].inner_attention, TPUVarlenAttention)
    # Idempotent when called a second time
    assert replace_varlen_attention_with_tpu_attention([model]) == 0


def test_resolve_tpu_topology_bounds_multi_host():
    from verl_hardware_plugin.platforms.platform_tpu import resolve_tpu_topology_bounds

    # v6e-8 across two hosts: the mesh spans hosts, so the bounds are the host bounds.
    topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
        total_chips=8, num_nodes=2
    )
    assert (topology, host_bounds, chips_per_host_bounds, chips_per_host) == ("2,4,1", "2,4,1", "1,1,1", "4")

    # v6e-16 used to fall through to "1,1,1", which silently trains on one chip.
    assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=4)[:2] == ("4,4,1", "4,4,1")


def test_resolve_tpu_topology_bounds_single_host():
    from verl_hardware_plugin.platforms.platform_tpu import resolve_tpu_topology_bounds

    # A 4-chip slice on one host is addressed inside the host, not across hosts.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=1) == ("2,2,1", "1,1,1", "2,2,1", "4")
    # Same chip count spread over two hosts cannot use in-host bounds.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=2) == ("2,2,1", "1,1,1", "1,1,1", "2")


def test_resolve_tpu_topology_bounds_pod_type_and_env_override():
    from verl_hardware_plugin.platforms.platform_tpu import resolve_tpu_topology_bounds

    # Pod type wins over the chip count: this job holds 8 of a 16-chip slice.
    assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2, pod_type="v6e-16")[0] == "4,4,1"

    # The env var wins over everything, including an unknown chip count.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,3,1"}):
        assert resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)[0] == "2,3,1"

    with mock.patch.dict(os.environ, {"VERL_TPU_CHIPS_PER_HOST": "2"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2)[3] == "2"


def test_resolve_tpu_topology_bounds_raises_on_unknown_slice():
    import pytest

    from verl_hardware_plugin.platforms.platform_tpu import resolve_tpu_topology_bounds

    # Guessing "1,1,1" here would train on a subset of the slice without any error.
    with pytest.raises(ValueError, match="TORCH_TPU_TOPOLOGY"):
        resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)


def test_auto_assign_accelerator_type_matches_slice_size_and_availability():
    import ray

    from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

    # Cluster with a 16-chip slice named "tpu-group-0" (alphabetically first!)
    # and two 8-chip slices named "tpu-group-1" and "custom-slice-2".
    fake_nodes = [
        *[
            {
                "NodeID": f"n16-{i}",
                "Alive": True,
                "Resources": {"TPU": 4.0, "tpu-group-0": 1.0},
                "Labels": {"ray.io/tpu-slice-name": "tpu-group-0", "ray.io/tpu-pod-type": "v6e-16"},
            }
            for i in range(4)
        ],
        *[
            {
                "NodeID": f"n8a-{i}",
                "Alive": True,
                "Resources": {"TPU": 4.0, "tpu-group-1": 1.0},
                "Labels": {"ray.io/tpu-slice-name": "tpu-group-1", "ray.io/tpu-pod-type": "v6e-8"},
            }
            for i in range(2)
        ],
        *[
            {
                "NodeID": f"n8b-{i}",
                "Alive": True,
                "Resources": {"TPU": 4.0, "custom-slice-2": 1.0},
                "Labels": {"ray.io/tpu-slice-name": "custom-slice-2", "ray.io/tpu-pod-type": "v6e-8"},
            }
            for i in range(2)
        ],
    ]
    avail = {n["NodeID"]: {"TPU": 4.0} for n in fake_nodes}

    platform = PlatformTPU()
    with (
        mock.patch.object(ray, "is_initialized", return_value=True),
        mock.patch.object(ray, "nodes", return_value=fake_nodes),
        mock.patch.object(ray._private.state, "available_resources_per_node", return_value=avail, create=True),
    ):
        # 1. An 8-TPU job ([4, 4]) skips "tpu-group-0" (16 chips) and picks "custom-slice-2" / "tpu-group-1"
        slice_8_first = platform.auto_assign_accelerator_type("actor", None, [4, 4])
        assert slice_8_first == "custom-slice-2"

        # 2. A second 8-TPU pool ([4, 4]) picks the remaining 8-TPU slice ("tpu-group-1")
        slice_8_second = platform.auto_assign_accelerator_type("rollout", None, [4, 4])
        assert slice_8_second == "tpu-group-1"

        # 3. A 16-TPU pool ([4, 4, 4, 4]) picks "tpu-group-0"
        slice_16 = platform.auto_assign_accelerator_type("trainer16", None, [4, 4, 4, 4])
        assert slice_16 == "tpu-group-0"

