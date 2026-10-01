#!/usr/bin/env python3
"""CPU-only source/retake comparison for H3 retake validation videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


TEMPORAL = (34, 107)
SPATIAL = (96, 112, 640, 416)  # x0,y0,x1,y1, half-open
FPS = 24
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # H3 native temporal grid from comfy/ldm/minimax/model.py
SPATIAL_PATCH_PIXELS = 32  # 2x2 groups of H3 latent cells; each cell covers 16 source pixels here.


def native_support(count: int, start: int, end: int, feather: int) -> tuple[int, int]:
    """Mirror H3RetakePrepare's profile pooling to latent token intervals."""
    t = np.arange(count, dtype=np.float32) + 0.5
    profile = ((t >= start) & (t < end)).astype(np.float32)
    if feather > 0:
        ramp = np.minimum((t - start) / feather, (end - t) / feather)
        ramp = np.clip(ramp, 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
        profile *= ramp
    pos = 0
    active = []
    k = 0
    while pos < count:
        stop = min(count, pos + FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)])
        if profile[pos:stop].max(initial=0) > 0:
            active.extend((pos, stop))
        pos = stop
        k += 1
    return (min(active), max(active)) if active else (0, 0)


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def stats(source: list[np.ndarray], result: list[np.ndarray],
          frame_indices: np.ndarray, spatial: np.ndarray) -> dict:
    if frame_indices.size == 0 or not np.any(spatial):
        return {"frames": int(frame_indices.size), "pixels_per_frame": 0}
    diffs = []
    for i in frame_indices:
        a = source[int(i)].astype(np.float32)[spatial]
        b = result[int(i)].astype(np.float32)[spatial]
        diffs.append(np.abs(a - b))
    d = np.concatenate(diffs, axis=0)
    return {
        "frames": int(frame_indices.size),
        "pixels_per_frame": int(np.count_nonzero(spatial)),
        "mae_0_255": float(d.mean()),
        "rmse_0_255": float(np.sqrt(np.mean(d * d))),
        "p95_abs_0_255": float(np.percentile(d, 95)),
        "fraction_channel_values_over_2": float(np.mean(d > 2)),
        "fraction_channel_values_over_8": float(np.mean(d > 8)),
    }


def make_sheet(source: list[np.ndarray], result: list[np.ndarray], path: Path) -> list[int]:
    n = min(len(source), len(result))
    indices = [s * FPS for s in range(6) if s * FPS < n]
    h, w = source[0].shape[:2]
    tw = 256
    th = round(tw * h / w)
    lh = 30
    sheet = Image.new("RGB", (len(indices) * tw, 2 * (th + lh)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for col, idx in enumerate(indices):
        for row, frames, label in ((0, source, "source"), (1, result, "retake")):
            im = Image.fromarray(frames[idx]).resize((tw, th), Image.Resampling.LANCZOS)
            y = row * (th + lh)
            sheet.paste(im, (col * tw, y))
            draw.text((col * tw + 5, y + th + 7), f"{label}, {idx}f / {idx/FPS:.0f}s", fill="white")
    sheet.save(path, quality=90)
    return indices


def analyze(source_path: Path, result_path: Path, out: Path) -> dict:
    source, result = read_video(source_path), read_video(result_path)
    if not source or not result:
        raise RuntimeError(f"Could not decode source or retake: {source_path}, {result_path}")
    if source[0].shape != result[0].shape:
        raise ValueError(f"Resolution mismatch: {source[0].shape} vs {result[0].shape}")
    n = min(len(source), len(result))
    source, result = source[:n], result[:n]
    h, w = source[0].shape[:2]
    x0, y0, x1, y1 = SPATIAL
    x1, y1 = min(x1, w), min(y1, h)
    spatial_requested = np.zeros((h, w), dtype=bool)
    spatial_requested[y0:y1, x0:x1] = True
    patch = SPATIAL_PATCH_PIXELS
    sx0, sy0 = (x0 // patch) * patch, (y0 // patch) * patch
    sx1, sy1 = min(w, ((x1 + patch - 1) // patch) * patch), min(h, ((y1 + patch - 1) // patch) * patch)
    spatial_snapped = np.zeros((h, w), dtype=bool)
    spatial_snapped[sy0:sy1, sx0:sx1] = True
    all_pixels = np.ones((h, w), dtype=bool)
    inside_frames = np.arange(max(0, TEMPORAL[0]), min(TEMPORAL[1], n))
    support_start, support_end = native_support(n, *TEMPORAL, feather=8)
    support_inside_frames = np.arange(support_start, min(support_end, n))
    outside_frames = np.array([i for i in range(n) if i < support_start or i >= support_end])
    out.mkdir(parents=True, exist_ok=True)
    stem = result_path.stem
    previews = out / "previews"
    metrics = out / "metrics"
    previews.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    contact = previews / f"{stem}_source_vs_retake_0-5s.jpg"
    indices = make_sheet(source, result, contact)

    # Framewise copy-out checks around temporal interval and both edit edges.
    framewise = {}
    for i in sorted({0, TEMPORAL[0] - 1, TEMPORAL[0], TEMPORAL[1] - 1, TEMPORAL[1], n - 1}):
        if not 0 <= i < n:
            continue
        d = np.abs(source[i].astype(np.float32) - result[i].astype(np.float32))
        framewise[str(i)] = {
            "global_mae_0_255": float(d.mean()),
            "spatial_requested_inside_mae": float(d[spatial_requested].mean()),
            "spatial_snapped_inside_mae": float(d[spatial_snapped].mean()),
            "spatial_snapped_outside_mae": float(d[~spatial_snapped].mean()),
        }

    temporal_range = np.arange(n)
    # Spatial and temporal overlap is reported separately to reveal leakage.
    temporal_inside_spatial_inside = stats(source, result, support_inside_frames, spatial_snapped)
    temporal_inside_spatial_outside = stats(source, result, support_inside_frames, ~spatial_snapped)
    report = {
        "source": str(source_path),
        "retake": str(result_path),
        "frames": n,
        "dimensions": [w, h],
        "fps": FPS,
        "temporal_requested_half_open": list(TEMPORAL),
        "temporal_native_latent_support_half_open": [support_start, support_end],
        "temporal_support_derivation": "H3RetakePrepare profile (smoothstep feather 8) max-pooled over FRAME_PER_TOKEN=(1,4,4,4,4); first/last active latent intervals snap [34,107) outward.",
        "spatial_rect_requested_half_open_xyxy": [x0, y0, x1, y1],
        "spatial_rect_native_snapped_half_open_xyxy": [sx0, sy0, sx1, sy1],
        "spatial_snap_patch_pixels": patch,
        "regions_vs_source": {
            "outside_temporal_support_all_pixels": stats(source, result, outside_frames, all_pixels),
            "inside_temporal_support_all_pixels": stats(source, result, support_inside_frames, all_pixels),
            "inside_requested_spatial_rectangle_all_frames": stats(source, result, temporal_range, spatial_requested),
            "outside_requested_spatial_rectangle_all_frames": stats(source, result, temporal_range, ~spatial_requested),
            "inside_native_snapped_spatial_rectangle_all_frames": stats(source, result, temporal_range, spatial_snapped),
            "outside_native_snapped_spatial_rectangle_all_frames": stats(source, result, temporal_range, ~spatial_snapped),
            "inside_temporal_and_native_snapped_spatial_support": temporal_inside_spatial_inside,
            "inside_temporal_but_outside_native_snapped_spatial_support": temporal_inside_spatial_outside,
        },
        "selected_frame_differences": framewise,
        "contact_sheet_frames": indices,
        "contact_sheet": str(contact),
        "limits": {
            "decoded_pixels": "Source and retake are decoded video frames; H.264 re-encoding can create small differences even outside the edit mask.",
            "scores": "These are preservation/change measurements, not a score for visual quality or edit correctness. Review source/retake contact sheets and motion.",
            "mask_grid": "Spatial rectangle is evaluated at requested half-open pixel bounds; the node's internal grid snapping and latent support can expand effective influence.",
        },
    }
    report_path = metrics / f"{stem}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("retakes", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for retake in args.retakes:
        if retake.is_file():
            print(json.dumps(analyze(args.source, retake, args.out_dir), indent=2))
        else:
            print(f"skip missing retake: {retake}")


if __name__ == "__main__":
    main()
