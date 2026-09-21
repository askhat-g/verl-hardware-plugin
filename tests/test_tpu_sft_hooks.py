# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for TPU SFT runtime hooks, metric aggregation, and bucketed loss unpacking."""

import os
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from verl_hardware_plugin.platforms.platform_tpu_workarounds import (
    aggregate_sft_metrics_tpu,
    convert_tensors_to_scalars,
    extract_validation_loss,
)
from verl_hardware_plugin.utils.tpu_sft_hooks import apply_tpu_sft_hooks


def test_convert_tensors_to_scalars():
    raw = {
        "loss": torch.tensor([1.25]),
        "grad_norm": [torch.tensor(0.5), torch.tensor(0.75)],
        "vector": torch.tensor([1.0, 2.0]),
    }
    converted = convert_tensors_to_scalars(raw)
    assert converted["loss"] == pytest.approx(1.25)
    assert converted["grad_norm"] == [pytest.approx(0.5), pytest.approx(0.75)]
    assert isinstance(converted["vector"], torch.Tensor)
    assert converted["vector"].device.type == "cpu"


def test_aggregate_sft_metrics_tpu():
    rank_metrics = [
        {"loss": 2.0, "grad_norm": 1.0, "lr": 1e-5},
        {"loss": 4.0, "grad_norm": 3.0, "lr": 1e-5},
    ]
    agg = aggregate_sft_metrics_tpu(rank_metrics)
    assert agg["loss"] == pytest.approx(3.0)
    assert agg["grad_norm"] == pytest.approx(2.0)
    assert agg["lr"] == pytest.approx(1e-5)
    assert agg["mfu"] == pytest.approx(0.0)


def test_extract_validation_loss():
    assert extract_validation_loss([{"loss": 1.5}, {"loss": 2.5}]) == pytest.approx(2.0)
    assert extract_validation_loss({"loss": [1.0, 3.0]}) == pytest.approx(2.0)
    assert extract_validation_loss(1.75) == pytest.approx(1.75)


def test_tpu_sft_loss_unpacks_bucketed_padded_values():
    import verl.workers.utils.losses as verl_losses
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    apply_tpu_sft_hooks()

    # 2 packed sequences of length 3 and 2 -> orig_seq_len=5, padded to bucket 8
    unpadded = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float32)
    padded = torch.cat([unpadded, torch.zeros(3, dtype=torch.float32)])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    log_probs_nt = torch.nested.nested_tensor_from_jagged(unpadded, cu_seqlens)
    log_probs_nt._tpu_padded_values = padded

    loss_mask_nt = torch.nested.as_nested_tensor(
        [torch.tensor([0, 1, 1], dtype=torch.int64), torch.tensor([0, 1], dtype=torch.int64)],
        layout=torch.jagged,
    )
    data = TensorDict({"loss_mask": loss_mask_nt}, batch_size=[2])
    tu.assign_non_tensor_data(data, "pad_mode", DatasetPadMode.NO_PADDING)
    tu.assign_non_tensor_data(data, "dp_size", 1)
    tu.assign_non_tensor_data(data, "batch_num_tokens", 3)

    loss, metrics = verl_losses.sft_loss(config=None, model_output={"log_probs": log_probs_nt}, data=data)
    assert torch.isfinite(loss)
    assert "loss" in metrics


def test_concat_tensordict_merges_metrics_on_tpu():
    import verl.plugin.platform.platform_manager as pm
    from verl.utils import tensordict_utils as tu
    from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU  # noqa: F401

    apply_tpu_sft_hooks()
    old_platform = pm._current_platform
    pm._current_platform = None
    try:
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "tpu"}):
            td1 = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 1.0}})
            td2 = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": 3.0}})
            merged = tu.concat_tensordict_with_none_bsz([td1, td2])
            assert tu.get(merged, "metrics")["loss"] == [1.0, 3.0]
    finally:
        pm._current_platform = old_platform
