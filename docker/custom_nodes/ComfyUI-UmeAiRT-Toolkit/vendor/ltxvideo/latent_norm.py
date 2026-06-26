"""
Vendored from ComfyUI-LTXVideo/latent_norm.py (Apache 2.0)
Only LTXVAdainLatent is needed for the LoopingSampler's normalization.
"""

import copy
import torch


class LTXVAdainLatent:
    """Adaptive Instance Normalization for video latents.

    Normalizes each channel of the target latent to match the statistics
    (mean, std) of the reference latent, then linearly interpolates
    by the given factor.
    """

    def batch_normalize(self, latents, reference, factor, per_frame=False):
        """Normalize latents to match reference statistics.

        Args:
            latents: Dict with "samples" key, shape [B, C, F, H, W].
            reference: Dict with "samples" key, same format.
            factor: Interpolation factor (0=no change, 1=full normalization).
            per_frame: If True, normalize per-frame instead of globally.

        Returns:
            Tuple of (normalized_latents_dict,).
        """
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"]  # B x C x F x H x W

        if per_frame:
            if reference["samples"].size(2) == 1:
                reference["samples"] = reference["samples"].repeat(
                    1, 1, t.size(2), 1, 1
                )
            elif t.size(2) > reference["samples"].size(2):
                raise ValueError("Latents have more frames than reference")

        for i in range(t.size(0)):  # batch
            for c in range(t.size(1)):  # channel
                if not per_frame:
                    r_sd, r_mean = torch.std_mean(
                        reference["samples"][i, c], dim=None
                    )
                    i_sd, i_mean = torch.std_mean(t[i, c], dim=None)
                    t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean
                else:
                    for f in range(t.size(2)):
                        r_sd, r_mean = torch.std_mean(
                            reference["samples"][i, c, f], dim=None
                        )
                        i_sd, i_mean = torch.std_mean(t[i, c, f], dim=None)
                        t[i, c, f] = ((t[i, c, f] - i_mean) / i_sd) * r_sd + r_mean

        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)
