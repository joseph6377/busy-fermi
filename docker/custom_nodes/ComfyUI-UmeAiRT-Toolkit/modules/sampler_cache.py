"""
UmeAiRT Toolkit - Sampler Prompt Cache
---------------------------------------
Caches CLIP-encoded prompt conditioning to avoid redundant encoding
when the prompt text, CLIP model, LoRAs, and ControlNets are unchanged.
"""

import copy
import weakref
import torch
from .common import log_node


def _check_controlnets_equal(c1, c2):
    """Compare two ControlNet stacks without using == on tensors.

    Tensor equality via ``==`` returns a tensor (not a bool) and raises
    ``RuntimeError`` in boolean contexts.  This function compares
    stack structure and object identity instead.

    Args:
        c1: First ControlNet stack (list of tuples).
        c2: Second ControlNet stack (list of tuples).

    Returns:
        bool: True if both stacks are structurally equivalent.
    """
    if c1 is c2:
        return True
    if not c1 and not c2:
        return True
    if not c1 or not c2:
        return False
    if len(c1) != len(c2):
        return False
    for t1, t2 in zip(c1, c2):
        if len(t1) != len(t2):
            return False
        if t1[0] != t2[0]:
            return False
        if t1[1] is not t2[1]:
            return False
        if t1[2:] != t2[2:]:
            return False
    return True


def build_zero_cond(positive_cond):
    """Build a zero-filled conditioning tensor matching the positive prompt shape.

    For FLUX/guidance-based models, encoding an empty string still produces
    non-zero embeddings that interfere with sampling (causes blurry output).
    A true zero conditioning effectively disables the negative guidance path.

    Args:
        positive_cond: The positive conditioning to match in structure/shape.

    Returns:
        A deep copy of positive_cond with all tensor values zeroed.
    """
    zero_cond = copy.deepcopy(positive_cond)
    for item in zero_cond:
        if isinstance(item, dict):
            for key, val in item.items():
                if isinstance(val, torch.Tensor):
                    item[key] = torch.zeros_like(val)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                    if isinstance(sub[0], torch.Tensor):
                        sub[0] = torch.zeros_like(sub[0])
                    if isinstance(sub[1], dict):
                        for k, v in sub[1].items():
                            if isinstance(v, torch.Tensor):
                                sub[1][k] = torch.zeros_like(v)
    return zero_cond


class PromptCache:
    """Caches CLIP-encoded conditioning to skip redundant tokenize+encode cycles.

    Cache is invalidated when any of the following change:
    - Positive or negative prompt text
    - CLIP model object (tracked via weakref)
    - Active LoRA stack configuration
    - Active ControlNet stack configuration
    """

    def __init__(self):
        self._last_pos_text = None
        self._last_neg_text = None
        self._last_clip_ref = None
        self._cached_positive = None
        self._cached_negative = None
        self._last_loras = None
        self._last_controlnets = None

    def try_get_cached(self, pos_text, neg_text, clip, loras, controlnets):
        """Attempt to retrieve cached conditioning.

        Args:
            pos_text (str): Current positive prompt text.
            neg_text (str): Current negative prompt text.
            clip: Current CLIP model object.
            loras: Current LoRA stack (list of tuples or None).
            controlnets: Current ControlNet stack (list of tuples or None).

        Returns:
            tuple or None: (positive_cond, negative_cond) deep copies if cache
            is valid, or None if cache is stale.
        """
        last_clip = self._last_clip_ref() if self._last_clip_ref is not None else None
        can_use = (
            self._last_pos_text == pos_text
            and self._last_neg_text == neg_text
            and last_clip is clip
            and self._last_loras == loras
            and _check_controlnets_equal(self._last_controlnets, controlnets)
        )
        if can_use:
            log_node("Image Generator: Using cached Prompts (Fast Start)", color="GREEN")
            return copy.deepcopy(self._cached_positive), copy.deepcopy(self._cached_negative)
        return None

    def update(self, pos_text, neg_text, clip, loras, controlnets, positive_cond, negative_cond):
        """Store freshly encoded conditioning in the cache.

        Args:
            pos_text (str): Positive prompt text.
            neg_text (str): Negative prompt text.
            clip: CLIP model object (stored as weakref).
            loras: LoRA stack configuration.
            controlnets: ControlNet stack configuration.
            positive_cond: Encoded positive conditioning.
            negative_cond: Encoded negative conditioning.
        """
        self._last_pos_text = pos_text
        self._last_neg_text = neg_text
        self._last_clip_ref = weakref.ref(clip)
        self._last_loras = loras
        self._last_controlnets = controlnets
        self._cached_positive = copy.deepcopy(positive_cond)
        self._cached_negative = copy.deepcopy(negative_cond)
