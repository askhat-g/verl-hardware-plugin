# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TorchTitan training engine implementation for Google TPU devices."""

import logging
import os
from contextlib import nullcontext
from typing import Callable

import torch
import torch.distributed
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngineWithLMHead
from verl_hardware_plugin.engines.torchtitan_tpu_utils import (
    bucket_length,
    compute_global_batch_num_tokens,
    pad_packed_inputs_for_tpu,
    replace_varlen_attention_with_tpu_attention,
    synchronize_tpu_loss,
    unwrap_metadata,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",
    backend=["torchtitan"],
    device="tpu",
    vendor="google",
)
class TorchTitanTPUEngineWithLMHead(TorchTitanEngineWithLMHead):
    """TorchTitan engine implementation for language models with LM head on Google TPU."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        from verl_hardware_plugin.utils.tpu_grpo_hooks import apply_tpu_grpo_hooks
        from verl_hardware_plugin.utils.tpu_sft_hooks import apply_tpu_sft_hooks

        apply_tpu_sft_hooks()
        apply_tpu_grpo_hooks()

        if engine_config.tensor_parallel_size > 1:
            logger.warning(
                "tensor_parallel_size=%d is not supported on TPU: it produces non-finite "
                "gradients, which optimizer_step() silently skips, so the policy will not "
                "train. Set tensor_parallel_size=1 and use data_parallel_shard_size instead.",
                engine_config.tensor_parallel_size,
            )

        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        # Apply TPU-specific TorchTitan Trainer.Config adjustments before initialize() creates Trainer:
        # 1. Use foreach optimizer implementation instead of CUDA fused
        self.config.optimizer.implementation = "foreach"
        # 2. Use default SPMD backend and 'tpu' compile backend
        self.config.parallelism.spmd_backend = "default"
        self.config.compile.backend = "tpu"
        # 3. Disable implicit CPU offload on TPU when forward_only is enabled unless explicitly requested
        if not self.engine_config.offload_policy:
            self.config.training.enable_cpu_offload = False

        if get_device_name() == "tpu" and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def initialize(self):
        super().initialize()
        replace_varlen_attention_with_tpu_attention(self.module)
        if self.engine_config.forward_only:
            self.trainer.optimizers = None
            self.trainer.lr_schedulers = None

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False):
        """Perform forward and optionally backward pass on a batch with TPU loss synchronization."""
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)
        dp_group = self.get_data_parallel_group()

        if get_device_name() == "tpu":
            batch_num_tokens = compute_global_batch_num_tokens(data, dp_group, self.engine_config.tensor_parallel_size)
        else:
            batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
            if dp_group is not None:
                torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            batch_num_tokens = batch_num_tokens.item()

        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens)
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=dp_group,
            same_micro_num_in_dp=True,
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self.trainer.train_context(), ctx, torch.profiler.record_function(f"micro_batch{micro_batch_idx}"):
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    if get_device_name() == "tpu":
                        synchronize_tpu_loss(loss)
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def optimizer_step(self):
        """Perform optimizer step with foreach=False gradient clipping on TPU."""
        from torchtitan.distributed import utils as dist_utils

        # torch._foreach_norm (foreach=True) returns inf on the TPU backend even when gradients are finite.
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=get_device_name() != "tpu",
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite (%s); skipping this optimizer step", grad_norm)
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        return grad_norm.item()

    def prepare_model_inputs(self, micro_batch: TensorDict):
        from verl.workers.engine.torchtitan.utils import get_attention_masks

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        if use_remove_padding:
            if get_device_name() == "tpu":
                input_ids, position_ids, labels, attention_mask, orig_seq_len = pad_packed_inputs_for_tpu(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    micro_batch=micro_batch,
                    device=get_device_id(),
                )
                output_args["orig_seq_len"] = orig_seq_len
            else:
                input_ids = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids = position_ids.values().unsqueeze(1)
                else:
                    position_ids = position_ids.values().unsqueeze(0)

                labels = torch.roll(input_ids, shifts=-1, dims=1)
                attn_type = self.engine_config.attn_type
                attention_mask = get_attention_masks(
                    input_batch=input_ids,
                    positions=position_ids,
                    attn_type=attn_type,
                )
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = int(max(input_ids.offsets().diff()))
            if get_device_name() == "tpu":
                max_seq_len = bucket_length(max_seq_len)

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0)
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )

            if position_ids.dim() == 3:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, position_ids.size(1), max_seq_len)
                ).transpose(0, 1)
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

            attention_mask_list = [torch.ones_like(t, dtype=torch.uint8) for t in loss_mask.unbind()]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(
                attention_mask, padding=0, output_size=(batch_size, max_seq_len)
            )
            if get_device_name() == "tpu":
                seq_idx = torch.arange(max_seq_len, device=attention_mask.device)
                causal = (seq_idx.unsqueeze(1) >= seq_idx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                attention_mask = causal & attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            else:
                attn_type = self.engine_config.attn_type
                attention_mask = get_attention_masks(
                    input_batch=input_ids,
                    positions=position_ids,
                    attn_type=attn_type,
                    attention_mask=attention_mask,
                )

        extra_inputs = {"positions": position_ids}
        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        extra_inputs.update(multi_modal_inputs)
        extra_kwargs = {"attention_masks": attention_mask}
        output_args["labels"] = labels

        if input_ids.device.type == "tpu":
            input_ids = input_ids.contiguous()
            extra_inputs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in extra_inputs.items()}
            extra_kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in extra_kwargs.items()}

        return input_ids, extra_inputs, extra_kwargs, output_args

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )
        pad_mode = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = unwrap_metadata(micro_batch["temperature"])
        calculate_entropy = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        )

        labels = output_args["labels"]
        model_output = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()

        if isinstance(logits, torch.distributed.tensor.DTensor):
            logits = logits.full_tensor()

        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0)
            logits_rmpad = logits_rmpad / temperature
            inplace_backward = not calculate_entropy
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            padded_log_probs = log_probs.squeeze(0)
            orig_seq_len = output_args.get("orig_seq_len")
            if orig_seq_len is not None:
                cu_seqlens_cpu = cu_seqlens.detach().cpu()
                unpadded_log_probs = padded_log_probs.detach().cpu()[:orig_seq_len]
                log_probs = torch.nested.nested_tensor_from_jagged(unpadded_log_probs, cu_seqlens_cpu)
                log_probs._tpu_padded_values = padded_log_probs
                if calculate_entropy:
                    unpadded_entropy = entropy_rmpad.detach().cpu()[:orig_seq_len]
                    entropy = torch.nested.nested_tensor_from_jagged(unpadded_entropy, cu_seqlens_cpu)
                    entropy._tpu_padded_values = entropy_rmpad
            else:
                log_probs = torch.nested.nested_tensor_from_jagged(padded_log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits = logits / temperature
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            lengths = cu_seqlens.diff()
            starts = torch.zeros_like(lengths, device=logits.device)
            logits = torch.nested.narrow(logits, 1, starts, lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=labels)
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)

            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        from verl.workers.engine.utils import detach_tree

        device_name = get_device_name()
        if device_name != "tpu":
            micro_batch = micro_batch.to(get_device_id())

        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            assert len(self.module) == 1, "TorchtitanEngine only supports single model module for now"
            raw_output = self.module[0](input_ids, **extra_inputs, **extra_kwargs)
            model_output = self.prepare_model_outputs(raw_output, output_args, micro_batch)

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "loss_function must be provided when forward_only is False"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            for val in model_output.values():
                if hasattr(val, "_tpu_padded_values"):
                    delattr(val, "_tpu_padded_values")

            output = {
                "model_output": detach_tree(model_output),
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

        return loss, output
