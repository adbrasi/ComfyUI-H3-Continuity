#!/usr/bin/env python3
"""CPU-only visual metrics and previews for H3 action/adventure chains.

Metrics are deliberately descriptive: decoded RGB seam delta, upper-wall
brightness proxy, and Laplacian variance. None proves identity or motion quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


FPS = 24
JOIN_FRAMES = (124, 226, 328)
FEATHER_HANDOVERS = (107, 209, 311)


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def pair_metric(frames: list[np.ndarray], index: int) -> dict[str, float | int] | None:
    if index <= 0 or index >= len(frames):
        return None
    delta = frames[index].astype(np.float32) - frames[index - 1].astype(np.float32)
    return {
        "mae_0_255": float(np.abs(delta).mean()),
        "rmse_0_255": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_0_255": int(np.abs(delta).max()),
    }


def flow_pair(frames: list[np.ndarray], index: int) -> dict[str, float | int] | None:
    """Flow magnitude and backward-flow-warped RGB residual for pair (i-1,i)."""
    if index <= 0 or index >= len(frames):
        return None
    prev = cv2.resize(frames[index - 1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    curr = cv2.resize(frames[index], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    # Backward flow maps each current-frame pixel to its matching previous-frame pixel.
    flow = cv2.calcOpticalFlowFarneback(
        curr_gray, prev_gray, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(np.sum(flow * flow, axis=2))
    yy, xx = np.mgrid[0:curr.shape[0], 0:curr.shape[1]].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = ((map_x >= 1) & (map_x < prev.shape[1] - 1) &
             (map_y >= 1) & (map_y < prev.shape[0] - 1))
    residual = np.abs(curr.astype(np.float32) - warped.astype(np.float32)).mean(axis=2)
    if not np.any(valid):
        return None
    return {
        "flow_magnitude_median_px_per_frame_halfres": float(np.median(mag[valid])),
        "flow_magnitude_p90_px_per_frame_halfres": float(np.percentile(mag[valid], 90)),
        "flow_warped_rgb_mae_0_255_halfres": float(residual[valid].mean()),
        "valid_warp_fraction": float(valid.mean()),
    }


def frame_set(frames: list[np.ndarray], indices: list[int], path: Path,
              labels: list[str], thumb_width: int = 320) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    thumb_height = round(thumb_width * h / w)
    label_h = 34
    sheet = Image.new("RGB", (len(indices) * thumb_width, thumb_height + label_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for col, (idx, label) in enumerate(zip(indices, labels)):
        if not 0 <= idx < len(frames):
            continue
        im = Image.fromarray(frames[idx]).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(im, (col * thumb_width, 0))
        draw.text((col * thumb_width + 5, thumb_height + 8), label, fill="white")
    sheet.save(path, quality=88)


def analyze(path: Path, out_dir: Path, strip_radius: int,
            join_frames: tuple[int, ...], handover_frames: tuple[int, ...],
            segment_ends: tuple[int, ...]) -> dict:
    frames = decode(path)
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    count = len(frames)
    metrics_dir = out_dir / "metrics"
    previews_dir = out_dir / "previews"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    # Per-video sample every three seconds, plus the final frame when needed.
    regular = list(range(0, count, 3 * FPS))
    if not regular or regular[-1] != count - 1:
        regular.append(count - 1)
    frame_set(frames, regular, previews_dir / f"{stem}_3sec.jpg", [f"{i}f / {i/FPS:.1f}s" for i in regular])

    # Three rows: each row is a seam, with pre/post samples on a common time axis.
    seam_indices = list(join_frames) + list(handover_frames)
    cols = sorted({max(0, min(count - 1, s + off)) for s in seam_indices
                   for off in range(-strip_radius, strip_radius + 1, max(1, strip_radius // 2))})
    # Include every boundary's local neighborhood as a separate row, padding missing frames.
    tw = 144
    h, w = frames[0].shape[:2]
    th = round(tw * h / w)
    label_h = 28
    valid_seams = [("payload join", s) for s in join_frames] + [("feather handover", s) for s in handover_frames]
    strip = Image.new("RGB", (len(range(-strip_radius, strip_radius + 1, max(1, strip_radius // 2))) * tw,
                               len(valid_seams) * (th + label_h)), (22, 22, 22))
    draw = ImageDraw.Draw(strip)
    offsets = list(range(-strip_radius, strip_radius + 1, max(1, strip_radius // 2)))
    for row, (label, seam) in enumerate(valid_seams):
        y0 = row * (th + label_h)
        draw.text((3, y0 + th + 8), f"{label} @ {seam}", fill="white")
        for col, off in enumerate(offsets):
            idx = seam + off
            if 0 <= idx < count:
                im = Image.fromarray(frames[idx]).resize((tw, th), Image.Resampling.LANCZOS)
                strip.paste(im, (col * tw, y0))
                draw.text((col * tw + 3, y0 + 3), f"{idx} ({off:+})", fill="yellow")
    strip.save(previews_dir / f"{stem}_seam_strips.jpg", quality=90)

    # Generated segments are 102 delivered frames each; source contributes 124.
    bounds = [0] + [min(end, count) for end in segment_ends if end > 0]
    if not bounds or bounds[-1] < count:
        bounds.append(count)
    segments = {}
    for ix, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        end = min(end, count)
        if end <= start:
            continue
        sharpness = []
        brightness = []
        for rgb in frames[start:end]:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            # Top ~40% is mostly sky/canopy/building in these planned shots.
            # Treat only as a broad background illumination/composition proxy.
            top = gray[:max(1, round(gray.shape[0] * 0.40)), :]
            brightness.append(float(top.mean()))
        segments["source" if ix == 0 else f"extension_{ix}"] = {
            "frame_range_half_open": [start, end],
            "mean_laplacian_variance": float(np.mean(sharpness)),
            "median_laplacian_variance": float(np.median(sharpness)),
            "mean_upper_40pct_luma_0_255": float(np.mean(brightness)),
        }

    seams = {str(s): pair_metric(frames, s) for s in join_frames if s < count}
    handovers = {str(s): pair_metric(frames, s) for s in handover_frames if s < count}
    def flow_window(center: int) -> dict[str, dict | None]:
        return {str(offset): flow_pair(frames, center + offset)
                for offset in range(-3, 4)}

    flow_seams = {str(s): flow_window(s) for s in join_frames if s < count}
    flow_handovers = {str(s): flow_window(s) for s in handover_frames if s < count}
    report = {
        "video": str(path),
        "frames": count,
        "dimensions": [frames[0].shape[1], frames[0].shape[0]],
        "fps": FPS,
        "expected_duration_seconds": count / FPS,
        "payload_join_rgb_delta": seams,
        "feather_handover_rgb_delta": handovers,
        "farneback_flow_and_warped_residual_near_payload_joins": flow_seams,
        "farneback_flow_and_warped_residual_near_feather_handovers": flow_handovers,
        "segments": segments,
        "previews": {
            "three_second_contact_sheet": str(previews_dir / f"{stem}_3sec.jpg"),
            "seam_strips": str(previews_dir / f"{stem}_seam_strips.jpg"),
        },
        "limits": {
            "seam_delta": "Decoded adjacent-frame RGB delta only; camera motion, pose change, exposure and codec affect it. It does not measure perceptual discontinuity.",
            "laplacian": "Blur/sharpness proxy; changes with subject scale, movement, texture, focus and framing. Do not treat as a monotonic degradation score.",
            "brightness": "Mean luma in the upper 40% is a rough background-light/composition proxy; it may include subject or foreground depending on scene layout.",
            "identity": "No automated identity or semantic continuity metric is applied; review contact sheets and video in motion.",
            "optical_flow": "Half-resolution Farneback median/p90 magnitude is a rough apparent-motion proxy. No crossfade is applied to delivered frames. Lower flow-warped RGB residual can reflect less motion, blur, different choreography, or flow-estimator limitations; it also depends on occlusion, disocclusion, parallax and lighting. Compare it with neighboring pairs, not as a standalone score.",
        },
    }
    (metrics_dir / f"{stem}_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path, help="Video paths (glob expansion is done by the shell).")
    parser.add_argument("--out-dir", type=Path, default=Path("milestones/action_validation"))
    parser.add_argument("--strip-radius", type=int, default=8, help="Frames on either side of each boundary.")
    parser.add_argument("--join-frames", type=int, nargs="+", default=list(JOIN_FRAMES), help="Nominal payload join frame indices.")
    parser.add_argument("--handover-frames", type=int, nargs="*", default=list(FEATHER_HANDOVERS), help="Mask/feather handover frame indices, if relevant.")
    parser.add_argument("--segment-ends", type=int, nargs="+", default=[124, 226, 328, 430], help="Exclusive segment endpoints, followed by a final segment if needed.")
    args = parser.parse_args()
    for video in args.videos:
        if video.is_file():
            print(json.dumps(analyze(video, args.out_dir, args.strip_radius,
                                     tuple(args.join_frames), tuple(args.handover_frames),
                                     tuple(args.segment_ends)), indent=2))
        else:
            print(f"skip missing video: {video}")


if __name__ == "__main__":
    main()
