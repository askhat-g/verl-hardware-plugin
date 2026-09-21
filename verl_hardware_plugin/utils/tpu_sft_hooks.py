# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Runtime hooks for TPU SFT training (bucketed sft_loss unpacking, logprobs, and multi-rank metric merging)."""

import logging
import os

import torch
import torch.nn.functional as F
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _tpu_sft_loss_with_padded_values(config, model_output, data: TensorDict, dp_group=None):
    """Computes SFT loss on TPU when model_output['log_probs'] carries static bucketed `_tpu_padded_values`."""
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.metric import AggregationType, Metric
    from verl.utils.torch_functional import masked_mean, masked_sum
    from verl.workers.utils import losses as verl_losses

    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    log_prob = model_output["log_probs"]
    log_prob_flatten = getattr(log_prob, "_tpu_padded_values", None)

    if pad_mode != DatasetPadMode.NO_PADDING or log_prob_flatten is None:
        return verl_losses._orig_sft_loss(config, model_output, data, dp_group=dp_group)

    dp_size = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)
    loss_mask = data["loss_mask"]

    loss_mask_flatten = torch.roll(loss_mask.values().detach().cpu(), shifts=-1, dims=0)
    pad_len = int(log_prob_flatten.shape[0]) - int(loss_mask_flatten.shape[0])
    if pad_len > 0:
        loss_mask_flatten = F.pad(loss_mask_flatten, (0, pad_len), value=0)
    loss_mask_flatten = loss_mask_flatten.to(device=log_prob_flatten.device)

    if batch_num_tokens is not None:
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        loss = -masked_mean(log_prob_flatten, loss_mask_flatten)

    metrics = {
        "loss": Metric(value=loss, aggregation=AggregationType.MEAN),
    }
    return loss, metrics


def apply_tpu_sft_hooks() -> None:
    """Applies TPU SFT runtime hooks to verl's loss, logprobs, and tensordict utilities."""
    # 1. Patch logprobs_from_logits to use logprobs_from_logits_naive on TPU tensors
    try:
        import verl.utils.torch_functional as tf

        if not getattr(tf.logprobs_from_logits, "_verl_tpu_patched", False):
            _orig_logprobs = tf.logprobs_from_logits

            def _patched_logprobs_from_logits(logits, labels, inplace_backward=True):
                if isinstance(logits, torch.Tensor) and logits.device.type == "tpu":
                    return tf.logprobs_from_logits_naive(logits, labels)
                return _orig_logprobs(logits, labels, inplace_backward=inplace_backward)

            _patched_logprobs_from_logits._verl_tpu_patched = True
            tf.logprobs_from_logits = _patched_logprobs_from_logits
    except Exception as e:
        logger.debug("Failed to patch logprobs_from_logits for TPU: %s", e)

    # 2. Patch sft_loss to unpack `_tpu_padded_values` when present
    try:
        import verl.workers.utils.losses as verl_losses

        if not getattr(verl_losses, "_verl_tpu_sft_loss_patched", False):
            verl_losses._orig_sft_loss = verl_losses.sft_loss
            verl_losses.sft_loss = _tpu_sft_loss_with_padded_values
            verl_losses._verl_tpu_sft_loss_patched = True
    except Exception as e:
        logger.debug("Failed to patch sft_loss for TPU: %s", e)

    # 3. Patch tensordict_utils.concat_tensordict_with_none_bsz to merge per-rank metrics on TPU
    try:
        import verl.utils.tensordict_utils as tu
        from verl.plugin.platform import get_platform

        if not getattr(tu, "_verl_tpu_concat_patched", False):
            _orig_concat = tu.concat_tensordict_with_none_bsz

            def _patched_concat_tensordict_with_none_bsz(data: list[TensorDict]):
                for d in data:
                    assert len(d.batch_size) == 0
                if len(data) > 1 and "metrics" in data[0] and get_platform().device_name == "tpu":
                    all_metrics = [tu.get(d, "metrics") for d in data]
                    if all(isinstance(m, dict) for m in all_metrics):
                        merged_metrics = {}
                        for k in all_metrics[0].keys():
                            vals = []
                            for m in all_metrics:
                                if k not in m:
                                    continue
                                v = m[k]
                                if isinstance(v, list):
                                    vals.extend(v)
                                else:
                                    vals.append(v)
                            merged_metrics[k] = vals
                        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": merged_metrics})
                return _orig_concat(data)

            tu.concat_tensordict_with_none_bsz = _patched_concat_tensordict_with_none_bsz
            tu._verl_tpu_concat_patched = True
    except Exception as e:
        logger.debug("Failed to patch concat_tensordict_with_none_bsz for TPU: %s", e)
