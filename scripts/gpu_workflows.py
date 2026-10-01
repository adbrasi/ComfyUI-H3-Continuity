#!/usr/bin/env python3
"""Build reproducible API workflows for MiniMax H3 continuity GPU studies.

This file only builds API JSON. It does not start ComfyUI or submit work;
submit the generated graphs with validate_gpu.py after reviewing them.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
WORKFLOWS = REPO / "workflows"
RETAKE_API = REPO / "milestones" / "research_retake_01" / "api" / "retake_feather8_91234.api.json"


def read_graph(path: Path) -> dict[str, Any]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not graph:
        raise ValueError(f"Expected non-empty API graph: {path}")
    return graph


def save_graph(graph: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def add_node(graph: dict[str, Any], class_type: str, inputs: dict[str, Any]) -> str:
    node_id = str(max((int(key) for key in graph if str(key).isdigit()), default=0) + 1)
    graph[node_id] = {"class_type": class_type, "inputs": inputs}
    return node_id


def rewire(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list):
        if len(value) == 2 and str(value[0]) in mapping and isinstance(value[1], int):
            return [mapping[str(value[0])], value[1]]
        return [rewire(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: rewire(item, mapping) for key, item in value.items()}
    return value


def add_reference_image(graph: dict[str, Any], image_source: list[Any], *, index: int) -> str:
    return add_node(graph, "ImageFromBatch", {
        "image": image_source, "batch_index": index, "length": 1,
    })


def build_continuation_chain(*, source_latent: str, clips: int = 7, seed: int = 94001,
                             context_frames: int = 39, target_frames: int = 141,
                             mode: str = "hard", reference: bool = True,
                             feather_frames: int = 17, feather_strength: float = 0.5,
                             label: str | None = None,
                             prompt: str | None = None) -> dict[str, Any]:
    """Build one serial graph, saving each untrimmed latent and final timeline.

    The initial source is an existing H3Continuity checkpoint in ComfyUI's
    output directory. Each subsequent join consumes the preceding sampler's
    raw AV latent, uses the original source's first frame as image identity
    reference, and appends only the new non-overlap frames to a cumulative
    preview timeline.
    """
    if clips < 1 or clips > 8:
        raise ValueError("clips must be between 1 and 8 continuation samples")
    if mode not in {"hard", "feather"}:
        raise ValueError("mode must be hard or feather")
    graph = read_graph(WORKFLOWS / "02_continuar_latent.api.json")
    graph["20"]["inputs"]["filename"] = source_latent
    graph["5"]["inputs"]["length"] = target_frames
    graph["21"]["inputs"].update({
        "context_frames": context_frames,
        "method": "pinned_av",
        "feather_frames": 0 if mode == "hard" else feather_frames,
        "feather_strength": 1.0 if mode == "hard" else feather_strength,
    })
    graph["8"]["inputs"]["noise_seed"] = seed
    if prompt is not None:
        graph["5"]["inputs"]["prompt"] = prompt
    # The same original first-frame reference is fed through the native H3
    # image tokenizer on every continuation; a no-reference control is also
    # available to isolate the value of that persistent anchor.
    if reference:
        first_ref = add_reference_image(graph, ["22", 0], index=0)
        graph["5"]["inputs"]["first_frame"] = [first_ref, 0]
    else:
        graph["5"]["inputs"].pop("first_frame", None)

    # Keep only graph nodes required for serial sampling from the base graph.
    # The second through Nth copies share already-loaded model/VAE/CLIP nodes.
    branch_ids = ("7", "8", "9", "10", "11", "12", "13", "16", "21", "22", "23", "24")
    previous_sampler = "11"
    previous_assemble = None
    for clip_index in range(1, clips + 1):
        if clip_index == 1:
            mapping = {node: node for node in branch_ids}
        else:
            mapping = {node: add_node(graph, graph[node]["class_type"],
                                      rewire(copy.deepcopy(graph[node]["inputs"]), {}))
                       for node in branch_ids}
            # Replace the temporary empty-node definitions with a full remapped
            # copy. The ID mapping above is intentionally generated first.
            for node in branch_ids:
                graph[mapping[node]] = {"class_type": graph[node]["class_type"],
                                        "inputs": rewire(copy.deepcopy(graph[node]["inputs"]), mapping)}
        ids = mapping
        # Every segment starts with a fresh target canvas/latent and identical
        # prompt/reference conditioning, then joins the prior sampler latent.
        graph[ids["8"]]["inputs"]["noise_seed"] = seed + clip_index - 1
        graph[ids["21"]]["inputs"]["source_latent"] = ["20", 0] if clip_index == 1 else [previous_sampler, 0]
        graph[ids["21"]]["inputs"]["positive"] = ["5", 0]
        graph[ids["21"]]["inputs"]["latent"] = ["5", 1]
        graph[ids["21"]]["inputs"].update({
            "context_frames": context_frames, "method": "pinned_av",
            "feather_frames": 0 if mode == "hard" else feather_frames,
            "feather_strength": 1.0 if mode == "hard" else feather_strength,
        })
        graph[ids["7"]]["inputs"]["conditioning"] = [ids["21"], 0]
        graph[ids["11"]]["inputs"]["latent_image"] = [ids["21"], 1]
        graph[ids["24"]]["inputs"]["source_images"] = ["22", 0] if clip_index == 1 else [previous_assemble, 0]
        graph[ids["24"]]["inputs"]["source_audio"] = ["23", 0] if clip_index == 1 else [previous_assemble, 1]
        graph[ids["24"]]["inputs"]["images"] = [ids["12"], 0]
        graph[ids["24"]]["inputs"]["audio"] = [ids["13"], 0]
        graph[ids["24"]]["inputs"]["plan"] = [ids["21"], 2]
        graph[ids["16"]]["inputs"]["latent"] = [ids["11"], 0]
        graph[ids["16"]]["inputs"]["plan"] = [ids["21"], 2]
        suffix = label or ("reference" if reference else "noref")
        graph[ids["16"]]["inputs"]["filename"] = f"h3_validation/{{{{RUN_ID}}}}/chain_{suffix}_{clip_index:02d}.safetensors"

        previous_sampler = ids["11"]
        previous_assemble = ids["24"]

    # Remove legacy per-segment encoded-video nodes. The final assembled output
    # already contains the full image/audio timeline; raw AV latents are saved
    # after every generation for review and resume.
    for node_id in ("14", "15", "17", "25", "26", "30"):
        graph.pop(node_id, None)
    final_video = add_node(graph, "CreateVideo", {
        "images": [previous_assemble, 0], "audio": [previous_assemble, 1], "fps": 24,
    })
    suffix = label or ("reference" if reference else "noref")
    add_node(graph, "SaveVideo", {
        "video": [final_video, 0], "filename_prefix": f"h3_validation/{{{{RUN_ID}}}}/chain_{suffix}_{mode}",
        "format": "auto", "codec": "auto",
    })
    return graph


def build_retake(*, source_latent: str, seed: int = 55001, start: int = 34, end: int = 90,
                 feather_frames: int = 0, prompt: str | None = None) -> dict[str, Any]:
    """Build the known 124-frame mid-gesture retake with first/last references."""
    graph = read_graph(RETAKE_API)
    graph["20"]["inputs"]["filename"] = source_latent
    graph["8"]["inputs"]["noise_seed"] = seed
    graph["21"]["inputs"].update({"start_frame": start, "end_frame": end,
                                  "feather_frames": feather_frames, "curve": "smoothstep",
                                  "edit_audio": False})
    if prompt is not None:
        graph["5"]["inputs"]["prompt"] = prompt
    first = add_reference_image(graph, ["22", 0], index=0)
    last = add_reference_image(graph, ["22", 0], index=-1)
    graph["5"]["inputs"]["first_frame"] = [first, 0]
    graph["5"]["inputs"]["last_frame"] = [last, 0]
    graph["15"]["inputs"]["filename_prefix"] = f"h3_validation/{{{{RUN_ID}}}}/retake_{'hard' if feather_frames == 0 else f'feather{feather_frames}'}"
    graph["16"]["inputs"]["filename"] = f"h3_validation/{{{{RUN_ID}}}}/retake_{'hard' if feather_frames == 0 else f'feather{feather_frames}'}.safetensors"
    graph["29"]["inputs"]["filename_prefix"] = f"h3_validation/{{{{RUN_ID}}}}/retake_mask"
    return graph


def build_bridge(*, left_latent: str, right_latent: str, seed: int = 66001,
                 target_frames: int = 192, context_frames: int = 39,
                 feather_frames: int = 17, feather_strength: float = 0.5,
                 audio_policy: str = "preserve", prompt: str | None = None) -> dict[str, Any]:
    """Build a two-sided bridge between saved, compatible H3 latent clips."""
    graph = read_graph(WORKFLOWS / "00_gerar_fonte.api.json")
    graph["5"]["inputs"].update({"length": target_frames})
    graph["8"]["inputs"]["noise_seed"] = seed
    if prompt is not None:
        graph["5"]["inputs"]["prompt"] = prompt
    left_load = add_node(graph, "H3ContinuityLoad", {"filename": left_latent})
    right_load = add_node(graph, "H3ContinuityLoad", {"filename": right_latent})
    left_video = add_node(graph, "VAEDecode", {"samples": [left_load, 0], "vae": ["3", 0]})
    left_audio = add_node(graph, "VAEDecodeAudio", {"samples": [left_load, 0], "vae": ["4", 0]})
    right_video = add_node(graph, "VAEDecode", {"samples": [right_load, 0], "vae": ["3", 0]})
    right_audio = add_node(graph, "VAEDecodeAudio", {"samples": [right_load, 0], "vae": ["4", 0]})
    # The source clips in the current validation suite are 124/141 frames;
    # select the known final frame explicitly because current ImageFromBatch
    # schema constrains batch_index to be non-negative.
    left_end_ref = add_reference_image(graph, [left_video, 0], index=123)
    right_start_ref = add_reference_image(graph, [right_video, 0], index=0)
    graph["5"]["inputs"]["first_frame"] = [left_end_ref, 0]
    graph["5"]["inputs"]["last_frame"] = [right_start_ref, 0]
    bridge = add_node(graph, "H3BridgePrepare", {
        "positive": ["5", 0], "latent": ["5", 1],
        "left_latent": [left_load, 0], "right_latent": [right_load, 0],
        "context_frames": context_frames, "feather_frames": feather_frames,
        "feather_strength": feather_strength, "audio_policy": audio_policy,
    })
    graph["7"]["inputs"]["conditioning"] = [bridge, 0]
    graph["11"]["inputs"]["latent_image"] = [bridge, 1]
    graph["16"]["inputs"] = {"latent": ["11", 0],
                              "filename": "h3_validation/{{RUN_ID}}/bridge_latent.safetensors"}
    bridge_video = add_node(graph, "H3BridgeAssemble", {
        "images": ["12", 0], "left_images": [left_video, 0],
        "right_images": [right_video, 0], "plan": [bridge, 2],
        "audio": ["13", 0], "left_audio": [left_audio, 0], "right_audio": [right_audio, 0],
    })
    graph["14"]["inputs"]["images"] = [bridge_video, 0]
    graph["14"]["inputs"]["audio"] = [bridge_video, 1]
    graph["15"]["inputs"]["filename_prefix"] = "h3_validation/{{RUN_ID}}/bridge"
    graph["17"]["inputs"]["audio"] = [bridge_video, 1]
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    chain = sub.add_parser("chain")
    chain.add_argument("--source-latent", required=True)
    chain.add_argument("--clips", type=int, default=7)
    chain.add_argument("--seed", type=int, default=94001)
    chain.add_argument("--context", type=int, default=39)
    chain.add_argument("--target-frames", type=int, default=141)
    chain.add_argument("--mode", choices=("hard", "feather"), default="hard")
    chain.add_argument("--reference", choices=("yes", "no"), default="yes")
    chain.add_argument("--feather-frames", type=int, default=17)
    chain.add_argument("--feather-strength", type=float, default=0.5)
    chain.add_argument("--label", default=None)
    chain.add_argument("--out", type=Path, required=True)
    retake = sub.add_parser("retake")
    retake.add_argument("--source-latent", required=True)
    retake.add_argument("--seed", type=int, default=55001)
    retake.add_argument("--start", type=int, default=34)
    retake.add_argument("--end", type=int, default=90)
    retake.add_argument("--feather-frames", type=int, default=0)
    retake.add_argument("--out", type=Path, required=True)
    bridge = sub.add_parser("bridge")
    bridge.add_argument("--left-latent", required=True)
    bridge.add_argument("--right-latent", required=True)
    bridge.add_argument("--seed", type=int, default=66001)
    bridge.add_argument("--target-frames", type=int, default=192)
    bridge.add_argument("--context", type=int, default=39)
    bridge.add_argument("--feather-frames", type=int, default=17)
    bridge.add_argument("--feather-strength", type=float, default=0.5)
    bridge.add_argument("--audio-policy", choices=("preserve", "generate"), default="preserve")
    bridge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "chain":
        graph = build_continuation_chain(source_latent=args.source_latent, clips=args.clips,
                                         seed=args.seed, context_frames=args.context,
                                         target_frames=args.target_frames, mode=args.mode,
                                         reference=args.reference == "yes",
                                         feather_frames=args.feather_frames, feather_strength=args.feather_strength,
                                         label=args.label)
    elif args.command == "retake":
        graph = build_retake(source_latent=args.source_latent, seed=args.seed, start=args.start,
                             end=args.end, feather_frames=args.feather_frames)
    else:
        graph = build_bridge(left_latent=args.left_latent, right_latent=args.right_latent,
                             seed=args.seed, target_frames=args.target_frames,
                             context_frames=args.context, feather_frames=args.feather_frames,
                             feather_strength=args.feather_strength, audio_policy=args.audio_policy)
    save_graph(graph, args.out.resolve())
    print(f"Wrote {len(graph)} nodes to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
