#!/usr/bin/env python3
"""Reproducible ComfyUI API runs and H3 video/latent seam analysis.

The runner never starts ComfyUI or selects a device. It submits saved API-format
graphs to an already-running server, snapshots every submitted graph and result,
and downloads workflow outputs into an isolated run directory.

Examples:
  python scripts/validate_gpu.py run --graph workflows/00_gerar_fonte.api.json --name source
  python scripts/validate_gpu.py manifest --file scripts/gpu_validation.json
  python scripts/validate_gpu.py analyze-video out.mp4 --seam-frame 124 --contact-sheet
  python scripts/validate_gpu.py compare-latents source.safetensors generated.safetensors \
      --preserve 0:34 --preserve 90:124
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUT = REPO / "milestones" / "gpu_validation"


class ValidationError(RuntimeError):
    pass


def _json_request(url: str, *, method: str = "GET", payload: Any = None,
                  timeout: float = 30.0) -> tuple[Any, bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ValidationError(f"HTTP {exc.code} at {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValidationError(f"Could not reach ComfyUI at {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, raw


def _get_bytes(url: str, timeout: float = 60.0) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ValidationError(f"HTTP {exc.code} at {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValidationError(f"Request failed at {url}: {exc}") from exc


def _normalise_graph(value: Any) -> dict[str, Any]:
    # Workflow API export is a flat {node_id: {class_type, inputs}} mapping.
    # Accept Comfy's {prompt: ...} response wrapper as a convenience.
    if isinstance(value, dict) and "prompt" in value and isinstance(value["prompt"], dict):
        value = value["prompt"]
    if not isinstance(value, dict) or not value:
        raise ValidationError("Graph must be a non-empty ComfyUI API prompt mapping.")
    for node_id, node in value.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ValidationError(f"Node {node_id!r} must contain an inputs object.")
        if "class_type" not in node:
            raise ValidationError(f"Node {node_id!r} has no class_type.")
    return value


def _apply_overrides(graph: dict[str, Any], overrides: dict[str, Any] | None) -> None:
    for node_id, inputs in (overrides or {}).items():
        if str(node_id) not in graph:
            raise ValidationError(f"Override refers to missing node {node_id!r}.")
        if not isinstance(inputs, dict):
            raise ValidationError(f"Overrides for node {node_id!r} must be an object.")
        graph[str(node_id)]["inputs"].update(copy.deepcopy(inputs))


def _expand_variables(value: Any, variables: dict[str, str]) -> Any:
    """Expand {{NAME}} string tokens in API graphs and override values."""
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{{" + key + "}}", str(replacement))
        return value
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_variables(item, variables) for key, item in value.items()}
    return value


def _graph_path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _node_times(history: dict[str, Any]) -> dict[str, Any]:
    prompt = history.get("prompt") or []
    if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
        return prompt[2].get("execution_time", {}) or {}
    return history.get("execution_time", {}) or {}


def _history_payload(response: dict[str, Any], prompt_id: str) -> dict[str, Any] | None:
    item = response.get(prompt_id)
    if item is not None:
        return item
    if "outputs" in response or "status" in response:
        return response
    return None


def _download_history_outputs(base_url: str, history: dict[str, Any], folder: Path) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    outputs = history.get("outputs") or {}
    for node_id, node_outputs in outputs.items():
        if not isinstance(node_outputs, dict):
            continue
        for output_type, entries in node_outputs.items():
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict) or not entry.get("filename"):
                    continue
                filename = str(entry["filename"])
                subfolder = str(entry.get("subfolder", ""))
                typ = str(entry.get("type", "output"))
                query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": typ})
                url = base_url.rstrip("/") + "/view?" + query
                data = _get_bytes(url)
                name = Path(filename).name
                target = folder / f"node_{node_id}_{output_type}_{index}_{name}"
                target.write_bytes(data)
                saved.append({"node_id": str(node_id), "output_type": output_type,
                              "filename": filename, "subfolder": subfolder,
                              "type": typ, "saved_as": str(target), "bytes": len(data)})
    return saved


def run_graph(graph_file: Path, *, name: str, base_url: str, out_root: Path,
              overrides: dict[str, Any] | None = None, timeout: float = 1800,
              poll_interval: float = 1.0, client_id: str | None = None,
              download_outputs: bool = True, variables: dict[str, str] | None = None) -> dict[str, Any]:
    if not graph_file.is_file():
        raise ValidationError(f"Graph file does not exist: {graph_file}")
    with graph_file.open(encoding="utf-8") as stream:
        graph = _normalise_graph(json.load(stream))
    variables = {"RUN_ID": f"{_timestamp()}_{uuid.uuid4().hex[:6]}", **(variables or {})}
    graph = _expand_variables(graph, variables)
    overrides = _expand_variables(overrides or {}, variables)
    _apply_overrides(graph, overrides)
    run_id = f"{_timestamp()}_{name}_{uuid.uuid4().hex[:8]}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "submitted_api.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {"run_id": run_id, "name": name, "graph_source": str(graph_file.resolve()),
                "base_url": base_url.rstrip("/"), "overrides": overrides or {}, "variables": variables,
                "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "client_id": client_id or str(uuid.uuid4())}
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base = base_url.rstrip("/")
    _json_request(base + "/system_stats", timeout=10)
    queued, raw = _json_request(base + "/prompt", method="POST",
                                payload={"prompt": graph, "client_id": metadata["client_id"]}, timeout=60)
    (run_dir / "queue_response.json").write_bytes(raw + b"\n")
    if not isinstance(queued, dict) or not queued.get("prompt_id"):
        raise ValidationError(f"ComfyUI did not return prompt_id. Response saved in {run_dir}")
    prompt_id = str(queued["prompt_id"])
    metadata["prompt_id"] = prompt_id
    metadata["queued_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    start = time.monotonic()
    history = None
    latest_history = None
    while time.monotonic() - start < timeout:
        result, _ = _json_request(base + "/history/" + urllib.parse.quote(prompt_id), timeout=20)
        latest_history = _history_payload(result, prompt_id) if isinstance(result, dict) else None
        if latest_history is not None:
            status = latest_history.get("status", {}) or {}
            if status.get("completed") is True or status.get("status_str") == "success":
                history = latest_history
                break
            if status.get("status_str") == "error" or status.get("messages") and any(
                isinstance(m, (list, tuple)) and m and m[0] == "execution_error" for m in status["messages"]
            ):
                history = latest_history
                break
        time.sleep(max(0.1, poll_interval))
    else:
        raise ValidationError(f"Timed out after {timeout}s waiting for {prompt_id}; latest history: {latest_history!r}")

    elapsed = time.monotonic() - start
    history_file = run_dir / "history.json"
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    status = history.get("status", {}) or {}
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    outputs = _download_history_outputs(base, history, outputs_dir) if download_outputs else []
    result = {**metadata, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "wall_seconds": round(elapsed, 3), "status": status,
              "node_execution_times": _node_times(history), "outputs": outputs,
              "history_path": str(history_file)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if status.get("status_str") == "error" or status.get("completed") is False:
        raise ValidationError(f"ComfyUI run failed; inspect {history_file}")
    return result


def run_manifest(manifest_file: Path, *, base_url: str | None, out_root: Path | None,
                 timeout: float | None, poll_interval: float | None) -> list[dict[str, Any]]:
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValidationError("Manifest must be an object with a cases list.")
    root = out_root or _graph_path(data.get("output_dir", str(DEFAULT_OUT)), manifest_file.parent)
    server = (base_url or data.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
    variables = {"RUN_ID": str(data.get("run_id") or f"{_timestamp()}_{uuid.uuid4().hex[:6]}")}
    variables.update({str(k): str(v) for k, v in (data.get("variables") or {}).items()})
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, case in enumerate(data["cases"], 1):
        if not isinstance(case, dict) or not case.get("graph"):
            raise ValidationError(f"Case {index} needs a graph path.")
        graph = _graph_path(str(case["graph"]), manifest_file.parent)
        try:
            print(f"[{index}/{len(data['cases'])}] {case.get('name', graph.stem)} -> {server}", flush=True)
            report = run_graph(
                graph, name=str(case.get("name", graph.stem)), base_url=server, out_root=root,
                overrides=case.get("overrides"), timeout=timeout or float(data.get("timeout_seconds", 1800)),
                poll_interval=poll_interval or float(data.get("poll_interval_seconds", 1)),
                download_outputs=bool(case.get("download_outputs", True)),
                variables=variables,
            )
            reports.append(report)
            print(f"  completed in {report['wall_seconds']:.2f}s; artifacts: {report['history_path']}", flush=True)
        except Exception as exc:
            errors.append({"name": str(case.get("name", graph.stem)), "error": str(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            if not data.get("continue_after_error", False):
                break
    summary = {"manifest": str(manifest_file.resolve()), "base_url": server,
               "completed": reports, "errors": errors,
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest_result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if errors:
        raise ValidationError(f"{len(errors)} manifest case(s) failed; see {root / 'manifest_result.json'}")
    return reports


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ValidationError("ffmpeg is required for video analysis but was not found on PATH.")
    return exe


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValidationError("ffprobe is required for video analysis but was not found on PATH.")
    proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
                           "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if proc.returncode:
        raise ValidationError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _decode_rgb_frame(path: Path, frame: int, width: int, height: int) -> bytes:
    ffmpeg = _require_ffmpeg()
    proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-"],
                          capture_output=True, check=False)
    expected = width * height * 3
    if proc.returncode or len(proc.stdout) != expected:
        raise ValidationError(f"Could not decode frame {frame} from {path}: {proc.stderr.decode('utf-8', 'replace')}")
    return proc.stdout


def _frame_array(raw: bytes, width: int, height: int):
    try:
        import numpy as np
    except ImportError as exc:
        raise ValidationError("numpy is required for numeric video metrics.") from exc
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(np.float32)


def _frame_metrics(a, b) -> dict[str, float]:
    import numpy as np
    delta = np.abs(a - b)
    return {"mae_0_255": float(delta.mean()), "max_abs_0_255": float(delta.max()),
            "mean_rgb_delta": [float(x) for x in (b.mean((0, 1)) - a.mean((0, 1)))],
            "rmse_0_255": float(np.sqrt(np.square(a - b).mean()))}


def analyze_video(path: Path, *, seam_frame: int | None = None, every: int = 8,
                  contact_sheet: bool = False, seam_clip_radius: int = 24,
                  reference: Path | None = None,
                  compare_intervals: list[tuple[int, int]] | None = None,
                  out_dir: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"Video does not exist: {path}")
    probe = _probe_video(path)
    streams = probe.get("streams") or []
    if not streams:
        raise ValidationError(f"No video stream found in {path}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    nb_frames = stream.get("nb_frames")
    if nb_frames in (None, "N/A"):
        # duration * frame rate is a metadata estimate, marked as such below
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
        num, den = (int(x) for x in rate.split("/"))
        nb_frames = round(float(stream.get("duration") or (probe.get("format") or {}).get("duration") or 0) * num / den)
    frames = int(nb_frames)
    if frames <= 0:
        raise ValidationError(f"Could not determine frame count from {path}")
    if seam_frame is not None and not (1 <= seam_frame < frames):
        raise ValidationError(f"Seam frame must be within 1..{frames-1}; got {seam_frame}")
    output_dir = out_dir or path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"video": str(path.resolve()), "probe": probe, "frame_count": frames,
                              "dimensions": [width, height], "frame_count_is_estimate": stream.get("nb_frames") in (None, "N/A")}
    if seam_frame is not None:
        p = _frame_array(_decode_rgb_frame(path, seam_frame - 1, width, height), width, height)
        q = _frame_array(_decode_rgb_frame(path, seam_frame, width, height), width, height)
        report["seam"] = {"frame_index_zero_based": seam_frame,
                          "before_vs_after": _frame_metrics(p, q),
                          "interpretation": "Decoded pixel difference only; motion-aware human review is still required."}
        if reference is not None:
            ref_probe = _probe_video(reference)
            ref_stream = (ref_probe.get("streams") or [{}])[0]
            if int(ref_stream.get("width", -1)) != width or int(ref_stream.get("height", -1)) != height:
                raise ValidationError("Reference and generated videos must have the same dimensions for direct pixel comparison.")
            ref_a = _frame_array(_decode_rgb_frame(reference, seam_frame - 1, width, height), width, height)
            ref_b = _frame_array(_decode_rgb_frame(reference, seam_frame, width, height), width, height)
            report["seam"]["reference_before_vs_after"] = _frame_metrics(ref_a, ref_b)
            report["seam"]["source_to_generated_before"] = _frame_metrics(ref_a, p)
            report["seam"]["source_to_generated_after"] = _frame_metrics(ref_b, q)
        lo, hi = max(0, seam_frame - seam_clip_radius), min(frames - 1, seam_frame + seam_clip_radius)
        seam_path = output_dir / f"{path.stem}_seam_{seam_frame}.mp4"
        vf = f"select='between(n,{lo},{hi})',setpts=N/FRAME_RATE/TB"
        proc = subprocess.run([_require_ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vf", vf,
                               "-an", "-c:v", "libx264", "-crf", "12", "-preset", "fast", str(seam_path)],
                              capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            report["seam_clip"] = str(seam_path)
        else:
            report["seam_clip_error"] = proc.stderr.strip()
    if compare_intervals:
        if reference is None:
            raise ValidationError("--compare-interval requires --reference.")
        ref_probe = _probe_video(reference)
        ref_stream = (ref_probe.get("streams") or [{}])[0]
        if (int(ref_stream.get("width", -1)), int(ref_stream.get("height", -1))) != (width, height):
            raise ValidationError("Reference and generated videos must have matching dimensions for interval comparison.")
        ref_frames = ref_stream.get("nb_frames")
        if ref_frames not in (None, "N/A") and int(ref_frames) != frames:
            raise ValidationError("Reference and generated videos must have matching frame counts for interval comparison.")
        intervals = []
        for start, end in compare_intervals:
            if not (0 <= start < end <= frames):
                raise ValidationError(f"Comparison interval [{start},{end}) is outside 0..{frames}.")
            vals = []
            for i in range(start, end):
                original = _frame_array(_decode_rgb_frame(reference, i, width, height), width, height)
                rendered = _frame_array(_decode_rgb_frame(path, i, width, height), width, height)
                vals.append(_frame_metrics(original, rendered)["mae_0_255"])
            intervals.append({"frames": [start, end], "compared_frames": len(vals),
                              "mean_frame_mae_0_255": sum(vals) / len(vals),
                              "max_frame_mae_0_255": max(vals),
                              "min_frame_mae_0_255": min(vals)})
        report["reference_intervals"] = {"reference": str(reference.resolve()), "intervals": intervals,
                                         "interpretation": "H.264 encode/decode can change copied pixels; use latent comparisons for exact pre-encode preservation."}
    if contact_sheet:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise ValidationError("Pillow is required to create a contact sheet.") from exc
        every = max(1, every)
        selected = list(range(0, frames, every))
        if selected[-1] != frames - 1:
            selected.append(frames - 1)
        thumb_w, thumb_h = min(384, width), round(height * min(384, width) / width)
        cols = min(5, len(selected))
        rows = math.ceil(len(selected) / cols)
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for i, frame_index in enumerate(selected):
            raw = _decode_rgb_frame(path, frame_index, width, height)
            frame = Image.frombytes("RGB", (width, height), raw).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + 24)
            sheet.paste(frame, (x, y))
            draw.text((x + 5, y + thumb_h + 4), f"frame {frame_index} / {frame_index/24:.2f}s", fill="white")
        sheet_path = output_dir / f"{path.stem}_contact.jpg"
        sheet.save(sheet_path, quality=92)
        report["contact_sheet"] = str(sheet_path)
    return report


def parse_interval(text: str) -> tuple[int, int]:
    try:
        a, b = (int(value) for value in text.split(":", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError("Use START:END, pixel-frame indices, end exclusive.") from exc
    if a < 0 or b <= a:
        raise argparse.ArgumentTypeError("Interval requires 0 <= START < END.")
    return a, b


def parse_pair(text: str) -> tuple[int, int, int]:
    try:
        source_start, generated_start, length = (int(value) for value in text.split(":", 2))
    except Exception as exc:
        raise argparse.ArgumentTypeError("Use SOURCE_START:GENERATED_START:LENGTH in pixel frames.") from exc
    if min(source_start, generated_start) < 0 or length < 1:
        raise argparse.ArgumentTypeError("Starts must be non-negative and length positive.")
    return source_start, generated_start, length


def compare_latents(source: Path, generated: Path, *, preserve: list[tuple[int, int]],
                    pairs: list[tuple[int, int, int]] | None = None,
                    fps: int = 24, audio_hz: int = 40) -> dict[str, Any]:
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise ValidationError("Install safetensors and torch in the analysis Python to compare H3 latent files.") from exc
    # MiniMax H3 uses a repeating 1,4,4,4,4 pixel-frame coverage cycle.
    # Keep analysis independent from ComfyUI's runtime import path.
    frame_per_token = (1, 4, 4, 4, 4)
    def load(path: Path) -> dict[str, Any]:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key).float() for key in handle.keys()}
    src, gen = load(source), load(generated)
    regions = [(start, start, end - start) for start, end in preserve] + list(pairs or [])
    report: dict[str, Any] = {"source": str(source.resolve()), "generated": str(generated.resolve()),
                              "preserve_intervals_frames": preserve, "paired_intervals_frames": regions,
                              "streams": {}}
    if not regions:
        raise ValidationError("At least one --preserve START:END or --pair SOURCE_START:GENERATED_START:LENGTH interval is required.")
    for key in ("video", "audio"):
        if key not in src or key not in gen:
            raise ValidationError(f"Both safetensors files must contain a {key!r} stream.")
        a, b = src[key], gen[key]
        errors = []
        if key == "video":
            offsets = []
            cursor = 0
            for k in range(a.shape[2]):
                span = frame_per_token[k % len(frame_per_token)]
                offsets.append((cursor, cursor + span))
                cursor += span
            source_offsets = offsets
            generated_offsets = []
            cursor = 0
            for k in range(b.shape[2]):
                span = frame_per_token[k % len(frame_per_token)]
                generated_offsets.append((cursor, cursor + span))
                cursor += span
            for source_start, generated_start, length in regions:
                source_end, generated_end = source_start + length, generated_start + length
                src_idx = [i for i, (lo, hi) in enumerate(source_offsets) if hi > source_start and lo < source_end]
                gen_idx = [i for i, (lo, hi) in enumerate(generated_offsets) if hi > generated_start and lo < generated_end]
                if not src_idx or len(src_idx) != len(gen_idx):
                    errors.append({"source_frames": [source_start, source_end],
                                   "generated_frames": [generated_start, generated_end],
                                   "error": f"incompatible token coverage ({len(src_idx)} source vs {len(gen_idx)} generated steps)"})
                    continue
                delta = (a[:, :, src_idx] - b[:, :, gen_idx]).abs()
                errors.append({"source_frames": [source_start, source_end],
                               "generated_frames": [generated_start, generated_end],
                               "source_latent_steps": [src_idx[0], src_idx[-1] + 1],
                               "generated_latent_steps": [gen_idx[0], gen_idx[-1] + 1],
                               "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                               "exact": bool(torch.equal(a[:, :, src_idx], b[:, :, gen_idx]))})
        else:
            for source_start, generated_start, length in regions:
                # H3 clip/audio lengths are rounded onto a 40 Hz grid; round
                # both boundaries consistently to match that contract.
                slo = max(0, round(source_start * audio_hz / fps))
                shi = min(a.shape[-1], round((source_start + length) * audio_hz / fps))
                glo = max(0, round(generated_start * audio_hz / fps))
                ghi = min(b.shape[-1], round((generated_start + length) * audio_hz / fps))
                if shi <= slo or ghi <= glo or shi - slo != ghi - glo:
                    errors.append({"source_audio_ticks": [slo, shi], "generated_audio_ticks": [glo, ghi],
                                   "error": "incompatible audio tick coverage"})
                    continue
                delta = (a[..., slo:shi] - b[..., glo:ghi]).abs()
                errors.append({"source_audio_ticks": [slo, shi], "generated_audio_ticks": [glo, ghi],
                               "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                               "exact": bool(torch.equal(a[..., slo:shi], b[..., glo:ghi]))})
        report["streams"][key] = {"source_shape": list(a.shape), "generated_shape": list(b.shape), "regions": errors}
    report["note"] = "Latent tensor comparisons do not prove decoded pixel identity; output codec may add further loss."
    return report


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    value = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(value + "\n", encoding="utf-8")
    print(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Submit one saved ComfyUI API graph and archive its result.")
    run.add_argument("--graph", type=Path, required=True)
    run.add_argument("--name", default=None)
    run.add_argument("--base-url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run.add_argument("--overrides", type=Path, help="JSON mapping node ids to input overrides.")
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--poll", type=float, default=1)
    run.add_argument("--no-download", action="store_true")
    run.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                     help="Expand {{KEY}} tokens in graph strings; default RUN_ID is generated.")
    batch = sub.add_parser("manifest", help="Run saved graphs sequentially from a JSON manifest.")
    batch.add_argument("--file", type=Path, required=True)
    batch.add_argument("--base-url", default=None)
    batch.add_argument("--out-dir", type=Path, default=None)
    batch.add_argument("--timeout", type=float, default=None)
    batch.add_argument("--poll", type=float, default=None)
    video = sub.add_parser("analyze-video", help="Report dimensions, frame count and numeric seam difference; optionally make preview assets.")
    video.add_argument("video", type=Path)
    video.add_argument("--seam-frame", type=int)
    video.add_argument("--reference", type=Path)
    video.add_argument("--compare-interval", type=parse_interval, action="append", default=[],
                       help="Compare this half-open frame interval to --reference; may be repeated.")
    video.add_argument("--every", type=int, default=8)
    video.add_argument("--contact-sheet", action="store_true")
    video.add_argument("--seam-clip-radius", type=int, default=24)
    video.add_argument("--out-dir", type=Path)
    video.add_argument("--report", type=Path)
    latent = sub.add_parser("compare-latents", help="Compare selected protected H3 AV latent intervals.")
    latent.add_argument("source", type=Path)
    latent.add_argument("generated", type=Path)
    latent.add_argument("--preserve", type=parse_interval, action="append", default=[])
    latent.add_argument("--pair", type=parse_pair, action="append", default=[],
                        help="Compare matching intervals at different source/generated frame offsets.")
    latent.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            overrides = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides else None
            variables = {}
            for token in args.var:
                if "=" not in token:
                    raise ValidationError(f"Invalid --var {token!r}; use KEY=VALUE.")
                key, value = token.split("=", 1)
                variables[key] = value
            report = run_graph(args.graph.resolve(), name=args.name or args.graph.stem,
                               base_url=args.base_url, out_root=args.out_dir.resolve(), overrides=overrides,
                               timeout=args.timeout, poll_interval=args.poll, download_outputs=not args.no_download,
                               variables=variables)
            _write_report(report, None)
        elif args.command == "manifest":
            run_manifest(args.file.resolve(), base_url=args.base_url,
                         out_root=args.out_dir.resolve() if args.out_dir else None,
                         timeout=args.timeout, poll_interval=args.poll)
        elif args.command == "analyze-video":
            report = analyze_video(args.video.resolve(), seam_frame=args.seam_frame, every=args.every,
                                   contact_sheet=args.contact_sheet, seam_clip_radius=args.seam_clip_radius,
                                   reference=args.reference.resolve() if args.reference else None,
                                   compare_intervals=args.compare_interval,
                                   out_dir=args.out_dir.resolve() if args.out_dir else None)
            _write_report(report, args.report.resolve() if args.report else None)
        elif args.command == "compare-latents":
            report = compare_latents(args.source.resolve(), args.generated.resolve(),
                                     preserve=args.preserve, pairs=args.pair)
            _write_report(report, args.report.resolve() if args.report else None)
        return 0
    except (ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"validate_gpu: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
