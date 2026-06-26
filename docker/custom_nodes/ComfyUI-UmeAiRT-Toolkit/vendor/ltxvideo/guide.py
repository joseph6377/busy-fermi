"""
Vendored from ComfyUI-LTXVideo/guide.py (Apache 2.0)
Only the blur_internal helper is needed for the samplers.
"""

import comfy_extras.nodes_post_processing as post_processing


def blur_internal(image, blur_radius):
    """Apply Gaussian blur to an image tensor.

    Uses OpenCV-style sigma calculation: sigma = 0.3 * blur_radius
    (slightly weaker than the OpenCV default of 0.3 * r + 0.5).

    Args:
        image: Image tensor [B, H, W, C].
        blur_radius: Kernel radius. 0 = no blur.

    Returns:
        Blurred image tensor.
    """
    if blur_radius > 0:
        sigma = 0.3 * blur_radius
        image = post_processing.Blur.execute(image, blur_radius, sigma)[0]
    return image
