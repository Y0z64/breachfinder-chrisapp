from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from data.constants import LEFT_CP, LEFT_INNER, RIGHT_CP, RIGHT_INNER


_TITLE_HEIGHT = 44
_PADDING = 16
_BACKGROUND = (8, 8, 8, 255)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)

    minimum = float(data[finite].min())
    maximum = float(data[finite].max())
    if maximum <= minimum:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = (data - minimum) / (maximum - minimum)
    scaled[~finite] = 0.0
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def _border_seed_mask(fillable: np.ndarray) -> np.ndarray:
    seeds = np.zeros_like(fillable, dtype=bool)
    seeds[0, :] = fillable[0, :]
    seeds[-1, :] = fillable[-1, :]
    seeds[:, 0] |= fillable[:, 0]
    seeds[:, -1] |= fillable[:, -1]
    return seeds


def _cross_structure() -> np.ndarray:
    return np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=bool,
    )


def _flood_fill_final(fillable: np.ndarray) -> np.ndarray:
    seeds = _border_seed_mask(fillable)
    if not seeds.any():
        return np.zeros_like(fillable, dtype=bool)
    return ndimage.binary_propagation(seeds, mask=fillable, structure=_cross_structure())


def _flood_fill_preview_states(fillable: np.ndarray, max_frames: int = 5) -> list[np.ndarray]:
    """Generate bounded preview states for animation without exhaustive iteration."""
    seeds = _border_seed_mask(fillable)
    if not seeds.any():
        return [np.zeros_like(fillable, dtype=bool)]

    if max_frames <= 1:
        return [_flood_fill_final(fillable)]

    structure = _cross_structure()
    max_iter = max(fillable.shape)
    preview_iters = np.unique(
        np.linspace(0, max_iter, num=max_frames - 1, dtype=int)
    )

    states: list[np.ndarray] = []
    for step in preview_iters:
        if step <= 0:
            state = seeds.copy()
        else:
            state = ndimage.binary_dilation(
                seeds,
                structure=structure,
                iterations=int(step),
                mask=fillable,
            )
        states.append(state)

    states.append(_flood_fill_final(fillable))

    # Remove duplicate consecutive states to keep exported GIF compact.
    compact: list[np.ndarray] = []
    for state in states:
        if not compact or not np.array_equal(compact[-1], state):
            compact.append(state)
    return compact


def _flood_fill_states(fillable: np.ndarray) -> list[np.ndarray]:
    # Legacy API kept for readability in callers.
    return _flood_fill_preview_states(fillable, max_frames=6)


def _sample_states(states: list[np.ndarray], max_frames: int = 5) -> list[np.ndarray]:
    if len(states) <= max_frames:
        return states

    indexes = np.unique(np.linspace(0, len(states) - 1, num=max_frames).astype(int))
    return [states[index] for index in indexes]


def _overlay_mask(
    base: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    color_layer = Image.new("RGBA", base.size, color + (0,))
    color_layer.putalpha(Image.fromarray(mask.astype(np.uint8) * alpha, mode="L"))
    return Image.alpha_composite(base, color_layer)


def _render_frame(
    seg_slice: np.ndarray,
    title: str,
    overlays: list[tuple[np.ndarray, tuple[int, int, int], int]] | None = None,
) -> Image.Image:
    overlays = overlays or []
    base = Image.fromarray(_normalize_to_uint8(seg_slice), mode="L").convert("RGBA")
    content_size = max(base.size)
    content = ImageOps.pad(
        base,
        (content_size, content_size),
        method=Image.Resampling.NEAREST,
        color=_BACKGROUND,
    )

    for mask, color, alpha in overlays:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = ImageOps.pad(
            mask_image,
            (content_size, content_size),
            method=Image.Resampling.NEAREST,
            color=0,
        )
        content = _overlay_mask(content, np.asarray(mask_image) > 0, color, alpha)

    canvas_size = content_size + _TITLE_HEIGHT + _PADDING * 2
    canvas = Image.new("RGBA", (canvas_size, canvas_size), _BACKGROUND)
    content_x = (canvas_size - content_size) // 2
    canvas.alpha_composite(content, (content_x, _TITLE_HEIGHT + _PADDING))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((_PADDING, 10), title, fill=(240, 240, 240, 255), font=font)

    return canvas.convert("RGB")


def _hemisphere_animation_frames(
    seg_slice: np.ndarray,
    side_name: str,
    cp_mask: np.ndarray,
    inner_mask: np.ndarray,
    contra_inner_mask: np.ndarray,
    detect_weakpoints: bool,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    cp_overlay = cp_mask.astype(bool)
    inner_overlay = inner_mask.astype(bool)

    frames.append(
        _render_frame(
            seg_slice,
            f"{side_name} hemisphere: input labels",
            overlays=[
                (cp_overlay, (0, 200, 0), 140),
                (inner_overlay, (0, 120, 255), 120),
            ],
        )
    )

    fillable = ~cp_overlay
    flood_states = _sample_states(_flood_fill_states(fillable), max_frames=5)
    for step, state in enumerate(flood_states, start=1):
        frames.append(
            _render_frame(
                seg_slice,
                f"{side_name} hemisphere: flood fill step {step}/{len(flood_states)}",
                overlays=[
                    (cp_overlay, (0, 200, 0), 140),
                    (state, (0, 220, 220), 120),
                ],
            )
        )

    outside_reachable = flood_states[-1]
    frames.append(
        _render_frame(
            seg_slice,
            f"{side_name} hemisphere: flood fill complete",
            overlays=[
                (cp_overlay, (0, 200, 0), 140),
                (outside_reachable, (0, 220, 220), 120),
            ],
        )
    )

    outside_reachable &= ~contra_inner_mask.astype(bool)
    frames.append(
        _render_frame(
            seg_slice,
            f"{side_name} hemisphere: exclude contralateral inner tissue",
            overlays=[
                (cp_overlay, (0, 200, 0), 140),
                (contra_inner_mask.astype(bool), (255, 170, 0), 110),
                (outside_reachable, (0, 220, 220), 120),
            ],
        )
    )

    labeled_inner, n_regions = ndimage.label(inner_overlay)
    breach_mask = np.zeros_like(cp_overlay, dtype=bool)
    exposed_regions = np.zeros_like(cp_overlay, dtype=bool)

    for region_id in range(1, n_regions + 1):
        region = labeled_inner == region_id
        if (outside_reachable & region).any():
            exposed_regions |= region
            breach_mask |= ndimage.binary_dilation(region, iterations=1) & outside_reachable & ~region

    if exposed_regions.any():
        frames.append(
            _render_frame(
                seg_slice,
                f"{side_name} hemisphere: exposed inner region detected",
                overlays=[
                    (cp_overlay, (0, 200, 0), 140),
                    (exposed_regions, (255, 140, 0), 120),
                    (breach_mask, (255, 0, 0), 220),
                ],
            )
        )
    else:
        frames.append(
            _render_frame(
                seg_slice,
                f"{side_name} hemisphere: no breach detected",
                overlays=[
                    (cp_overlay, (0, 200, 0), 140),
                    (inner_overlay, (0, 120, 255), 120),
                ],
            )
        )

    if detect_weakpoints:
        eroded_cp = ndimage.binary_erosion(cp_overlay, iterations=1)
        eroded_fillable = ~eroded_cp
        eroded_states = _sample_states(_flood_fill_states(eroded_fillable), max_frames=4)
        for step, state in enumerate(eroded_states, start=1):
            frames.append(
                _render_frame(
                    seg_slice,
                    f"{side_name} hemisphere: weakpoint check step {step}/{len(eroded_states)}",
                    overlays=[
                        (eroded_cp, (0, 200, 0), 140),
                        (state, (255, 220, 0), 120),
                    ],
                )
            )

    return frames


def build_detection_animation_frames(
    seg_slice: np.ndarray,
    label_values: tuple[int, int] = (LEFT_CP, RIGHT_CP),
    detect_weakpoints: bool = False,
) -> list[Image.Image]:
    """Render a step-by-step GIF frame sequence for one displayed slice."""
    left_cp = seg_slice == label_values[0]
    right_cp = seg_slice == label_values[1]
    left_inner = seg_slice == LEFT_INNER
    right_inner = seg_slice == RIGHT_INNER

    frames = [
        _render_frame(
            seg_slice,
            "Input slice overview",
            overlays=[
                ((left_cp | right_cp), (0, 200, 0), 140),
                ((left_inner | right_inner), (0, 120, 255), 120),
            ],
        )
    ]

    if right_cp.any() or right_inner.any():
        frames.extend(
            _hemisphere_animation_frames(
                seg_slice,
                "Right",
                right_cp,
                right_inner,
                left_inner,
                detect_weakpoints,
            )
        )

    if left_cp.any() or left_inner.any():
        frames.extend(
            _hemisphere_animation_frames(
                seg_slice,
                "Left",
                left_cp,
                left_inner,
                right_inner,
                detect_weakpoints,
            )
        )

    return frames


def save_detection_animation(
    seg_slice: np.ndarray,
    output_path: str | Path,
    label_values: tuple[int, int] = (LEFT_CP, RIGHT_CP),
    detect_weakpoints: bool = False,
    duration_ms: int = 450,
) -> Path:
    """Save a square GIF that walks through the breach detection steps."""
    frames = build_detection_animation_frames(
        seg_slice,
        label_values=label_values,
        detect_weakpoints=detect_weakpoints,
    )
    if not frames:
        raise ValueError("No animation frames could be generated")

    output = Path(output_path)
    if output.suffix.lower() != ".gif":
        output = output.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)

    paletted = [frame.convert("P", palette=Image.ADAPTIVE, colors=256) for frame in frames]
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output
