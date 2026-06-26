"""
UmeAiRT Toolkit - LTX Utilities
---------------------------------
Spatio-temporal tiled VAE decode for LTX-2.3 video generation.

The tiled decode logic is adapted from ComfyUI-LTXVideo by Lightricks Ltd.
(Apache License 2.0) — https://github.com/Lightricks/ComfyUI-LTXVideo

Only the minimal decode utilities are included; no external dependency required.
"""

import logging
import torch

logger = logging.getLogger("UmeAiRT")


# ---------------------------------------------------------------------------
# Temporal chunk boundary helpers
# ---------------------------------------------------------------------------

def _compute_chunk_boundaries(chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames):
    """Compute chunk boundaries for temporal tiling.

    Args:
        chunk_start: Starting frame index for the current chunk.
        temporal_tile_length: Length of each temporal tile.
        temporal_overlap: Number of frames to overlap between chunks.
        total_latent_frames: Total number of latent frames.

    Returns:
        Tuple of (overlap_start, chunk_end).
    """
    if chunk_start == 0:
        chunk_end = min(chunk_start + temporal_tile_length, total_latent_frames)
        overlap_start = chunk_start
    else:
        overlap_start = max(1, chunk_start - temporal_overlap - 1)
        extra_frames = chunk_start - overlap_start
        chunk_end = min(
            chunk_start + temporal_tile_length - extra_frames,
            total_latent_frames,
        )
    return overlap_start, chunk_end


def _calculate_temporal_output_boundaries(overlap_start, time_scale_factor, tile_out_frames):
    """Calculate temporal output boundaries for the decoded tile.

    Args:
        overlap_start: Starting frame index including overlap.
        time_scale_factor: Time scaling factor from VAE.
        tile_out_frames: Number of frames in the decoded tile.

    Returns:
        Tuple of (out_t_start, out_t_end).
    """
    out_t_start = 1 + overlap_start * time_scale_factor
    out_t_end = out_t_start + tile_out_frames
    return out_t_start, out_t_end


# ---------------------------------------------------------------------------
# Spatial-only tiled decode (base)
# ---------------------------------------------------------------------------

def _spatial_tiled_decode(vae, samples, horizontal_tiles, vertical_tiles,
                          overlap, working_device="auto", working_dtype="auto"):
    """Decode latents with spatial tiling for VRAM efficiency.

    Decodes the full temporal extent at once but splits spatially into tiles
    with linear-blend overlap for seamless joins.

    Args:
        vae: ComfyUI VAE object.
        samples: 5-D latent tensor [B, C, F, H, W].
        horizontal_tiles: Number of horizontal tiles.
        vertical_tiles: Number of vertical tiles.
        overlap: Overlap in latent pixels between tiles.
        working_device: Target device for accumulation ('auto' = same as input).
        working_dtype: Target dtype for accumulation ('auto' = same as input).

    Returns:
        Decoded image tensor [B, T, H_out, W_out, 3].
    """
    batch, channels, frames, height, width = samples.shape
    time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula
    image_frames = 1 + (frames - 1) * time_scale_factor

    output_height = height * height_scale_factor
    output_width = width * width_scale_factor

    base_tile_height = (height + (vertical_tiles - 1) * overlap) // vertical_tiles
    base_tile_width = (width + (horizontal_tiles - 1) * overlap) // horizontal_tiles

    target_device = samples.device if working_device == "auto" else working_device
    if working_dtype == "auto":
        target_dtype = samples.dtype
    elif working_dtype == "float16":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    output = torch.zeros(
        (batch, image_frames, output_height, output_width, 3),
        device=target_device, dtype=target_dtype,
    )
    weights = torch.zeros(
        (batch, image_frames, output_height, output_width, 1),
        device=target_device, dtype=target_dtype,
    )

    for v in range(vertical_tiles):
        for h in range(horizontal_tiles):
            h_start = h * (base_tile_width - overlap)
            v_start = v * (base_tile_height - overlap)
            h_end = min(h_start + base_tile_width, width) if h < horizontal_tiles - 1 else width
            v_end = min(v_start + base_tile_height, height) if v < vertical_tiles - 1 else height

            tile = samples[:, :, :, v_start:v_end, h_start:h_end]
            decoded_tile = vae.decode(tile)

            out_h_start = v_start * height_scale_factor
            out_h_end = v_end * height_scale_factor
            out_w_start = h_start * width_scale_factor
            out_w_end = h_end * width_scale_factor

            tile_out_height = out_h_end - out_h_start
            tile_out_width = out_w_end - out_w_start
            tile_weights = torch.ones(
                (batch, image_frames, tile_out_height, tile_out_width, 1),
                device=decoded_tile.device, dtype=decoded_tile.dtype,
            )

            overlap_out_h = overlap * height_scale_factor
            overlap_out_w = overlap * width_scale_factor

            if h > 0:
                h_blend = torch.linspace(0, 1, overlap_out_w, device=decoded_tile.device)
                tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(1, 1, 1, -1, 1)
            if h < horizontal_tiles - 1:
                h_blend = torch.linspace(1, 0, overlap_out_w, device=decoded_tile.device)
                tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(1, 1, 1, -1, 1)
            if v > 0:
                v_blend = torch.linspace(0, 1, overlap_out_h, device=decoded_tile.device)
                tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(1, 1, -1, 1, 1)
            if v < vertical_tiles - 1:
                v_blend = torch.linspace(1, 0, overlap_out_h, device=decoded_tile.device)
                tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(1, 1, -1, 1, 1)

            output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += (
                decoded_tile * tile_weights
            ).to(target_device, target_dtype)
            weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights.to(
                target_device, target_dtype
            )

    output /= weights + 1e-8
    return output


# ---------------------------------------------------------------------------
# Public API: Spatio-temporal tiled decode
# ---------------------------------------------------------------------------

def ltx_spatio_temporal_tiled_decode(vae, latent_samples,
                                     spatial_tiles=4, spatial_overlap=1,
                                     temporal_tile_length=16, temporal_overlap=1):
    """Decode LTX video latents with combined spatial + temporal tiling.

    Splits the decode into temporal chunks (with blending) and within each
    chunk, spatial tiling (with blending).  This keeps peak VRAM low even
    for long, high-resolution videos.

    Args:
        vae: ComfyUI VAE object with ``downscale_index_formula`` attribute.
        latent_samples: 5-D tensor [B, C, F, H, W] (latent video).
        spatial_tiles: Number of spatial tiles in each direction.
        spatial_overlap: Overlap between spatial tiles (in latent pixels).
        temporal_tile_length: Max latent frames per temporal chunk.
        temporal_overlap: Overlap between temporal chunks (in latent frames).

    Returns:
        Decoded frames tensor [N, H, W, 3] ready for ComfyUI preview.
    """
    if temporal_tile_length < temporal_overlap + 1:
        raise ValueError("Temporal tile length must be greater than temporal overlap + 1")

    samples = latent_samples
    batch, channels, frames, height, width = samples.shape
    time_scale_factor, width_scale_factor, height_scale_factor = vae.downscale_index_formula
    image_frames = 1 + (frames - 1) * time_scale_factor

    output_height = height * height_scale_factor
    output_width = width * width_scale_factor

    target_device = samples.device
    target_dtype = samples.dtype

    output = torch.empty(
        (batch, image_frames, output_height, output_width, 3),
        device=target_device, dtype=target_dtype,
    )

    total_latent_frames = frames
    chunk_start = 0

    while chunk_start < total_latent_frames:
        overlap_start, chunk_end = _compute_chunk_boundaries(
            chunk_start, temporal_tile_length, temporal_overlap, total_latent_frames
        )

        chunk_frames = chunk_end - overlap_start
        logger.info(
            f"LTX Tiled Decode: temporal chunk {overlap_start}:{chunk_end} ({chunk_frames} latent frames)"
        )

        tile = samples[:, :, overlap_start:chunk_end]
        decoded_tile = _spatial_tiled_decode(
            vae=vae,
            samples=tile,
            vertical_tiles=spatial_tiles,
            horizontal_tiles=spatial_tiles,
            overlap=spatial_overlap,
        )

        if chunk_start == 0:
            output[:, :decoded_tile.shape[1]] = decoded_tile
        else:
            if decoded_tile.shape[1] == 1:
                raise ValueError("Dropping first frame but tile has only 1 frame")
            decoded_tile = decoded_tile[:, 1:]  # Drop first frame (overlap)

            out_t_start, out_t_end = _calculate_temporal_output_boundaries(
                overlap_start, time_scale_factor, decoded_tile.shape[1]
            )

            overlap_frames = temporal_overlap * time_scale_factor
            frame_weights = torch.linspace(
                0, 1, overlap_frames + 2,
                device=decoded_tile.device, dtype=decoded_tile.dtype,
            )[1:-1]
            tile_weights = frame_weights.view(1, -1, 1, 1, 1)
            after_overlap_start = out_t_start + overlap_frames

            overlap_output = decoded_tile[:, :overlap_frames]
            output[:, out_t_start:after_overlap_start] *= 1 - tile_weights
            output[:, out_t_start:after_overlap_start] += tile_weights * overlap_output
            output[:, after_overlap_start:out_t_end] = decoded_tile[:, overlap_frames:]

        chunk_start = chunk_end

    # Reshape to ComfyUI format: [B*T, H, W, 3]
    output = output.view(batch * image_frames, output_height, output_width, output.shape[-1])
    return output
