# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU-specific vLLM patches for rotary positional embeddings and compile cache."""

import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _tpu_sign(x: torch.Tensor) -> torch.Tensor:
    """Builds [[-1], [1]] with the dtype/device of ``x`` without Python globals that break AOT serialization."""
    return (torch.arange(2, dtype=x.dtype, device=x.device) * 2.0 - 1.0).unsqueeze(-1)


def _tpu_rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """cat((-x2, x1), -1) without a concat. Bitwise identical under IEEE754."""
    half = x.shape[-1] // 2
    swapped = x.unflatten(-1, (2, half)).flip(-2)
    return (swapped * _tpu_sign(x)).flatten(-2)


def _tpu_widen(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., half] -> [..., 1, 2 * half], i.e. cat((t, t), -1).unsqueeze(-2)."""
    lead = t.shape[:-1]
    half = t.shape[-1]
    return t.unsqueeze(-2).expand(*lead, 2, half).reshape(*lead, 2 * half).unsqueeze(-2).to(dtype)


def patch_tpu_rotary_emb() -> None:
    """Patches vLLM ApplyRotaryEmb and rotate_neox with concat-free implementations.

    On Google TPU, torch.cat along the last dimension lowers to a strided DynamicUpdateSlice (DUS).
    When fused with elementwise ops, it fails the unaligned DUS predicate inside the XLA:TPU
    fusion emitter (b/501165531). This patch replaces the concatenation with an unflatten + flip
    formulation that is bitwise identical and emits no DUS instructions.
    """
    try:
        import vllm.model_executor.layers.rotary_embedding.common as rotary_common
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_verl_tpu_rotary_patched", False):
            return

        def patched_forward_static(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> torch.Tensor:
            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()

            if is_neox_style:
                cos_f = _tpu_widen(cos, x.dtype)
                sin_f = _tpu_widen(sin, x.dtype)
                output = x * cos_f + _tpu_rotate_neox(x) * sin_f
            else:
                cos = cos.unsqueeze(-2).to(x.dtype)
                sin = sin.unsqueeze(-2).to(x.dtype)
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                output = torch.stack((o1, o2), dim=-1).flatten(-2)

            if enable_fp32_compute:
                output = output.to(origin_dtype)
            return output

        ApplyRotaryEmb.forward_static = staticmethod(patched_forward_static)
        rotary_common.rotate_neox = _tpu_rotate_neox
        ApplyRotaryEmb._verl_tpu_rotary_patched = True
        logger.info("Successfully applied TPU concat-free RoPE patch to vLLM.")
    except Exception as e:
        logger.debug("Failed to apply TPU rotary embedding patch to vLLM: %s", e)


def _tpu_runtime_present() -> bool:
    if os.environ.get("VERL_PLATFORM") == "tpu":
        return True
    return importlib.util.find_spec("torch_tpu") is not None


def apply_tpu_vllm_patches() -> None:
    """Apply TPU-specific vLLM patches when running on TPU."""
    if not _tpu_runtime_present():
        return

    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    patch_tpu_rotary_emb()
