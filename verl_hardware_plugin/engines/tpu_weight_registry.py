# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lightweight CPU-only Ray actor registry for tracking TPU weight checkpoint references."""

import os

import ray


class TPUWeightRegistryState:
    """In-memory state container for holding references to synchronized TPU model weights across steps."""

    def __init__(self):
        self.weights = {}

    def set_weights(self, step, ref):
        self.weights[step] = ref
        # Keep only the entry just written, to bound memory and disk.
        # Evict by write, not by step order: this actor is detached, so a step
        # left by a previous job (e.g. step 5) would otherwise outrank step 0
        # of a fresh job.
        for old_step in [s for s in self.weights if s != step]:
            old_ref = self.weights[old_step]
            if isinstance(old_ref, str):
                try:
                    if os.path.exists(old_ref):
                        os.remove(old_ref)
                except Exception:
                    pass
            del self.weights[old_step]

    def get_weights(self, step):
        return self.weights.get(step, None)

    def clear(self):
        """Drops every cached entry. Used to reset state left by a previous job."""
        self.weights.clear()


TPUWeightRegistry = ray.remote(num_cpus=0)(TPUWeightRegistryState)
