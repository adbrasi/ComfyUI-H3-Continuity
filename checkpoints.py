"""Safe, lossless safetensors persistence for MiniMax H3 conditioning."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import folder_paths

from .nodes import CATEGORY, output_path


_FORMAT = "h3_conditioning_checkpoint"
_VERSION = "1"
_TREE_KEY = "conditioning_tree"
_SETTINGS_KEY = "settings_json"
_MAX_DEPTH = 128


def _json_settings(value: str) -> str:
    """Validate a JSON object while retaining the user's exact JSON text."""
    if not isinstance(value, str):
        raise ValueError("Settings must be supplied as a JSON object string.")

    def reject_constant(constant: str):
        raise ValueError(f"Settings JSON cannot contain {constant}.")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Settings must contain valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Settings JSON must have an object at its root.")
    return value


def _encode(value: Any, tensors: dict[str, torch.Tensor], path: str, depth: int = 0):
    if depth > _MAX_DEPTH:
        raise ValueError(f"Conditioning metadata is nested too deeply at {path}.")
    if torch.is_tensor(value):
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(f"Unsupported tensor layout at {path}.")
        key = f"tensor_{len(tensors):06d}"
        # Safetensors stores dense tensors; contiguous() changes layout only and
        # keeps the values and dtype intact. Detaching is appropriate for saved
        # inference conditioning, which does not preserve an autograd graph.
        tensors[key] = value.detach().contiguous().cpu()
        return {"type": "tensor", "key": key}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float in conditioning at {path}.")
        # repr() is Python's shortest round-trippable representation.
        return {"type": "float", "value": repr(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _encode(item, tensors, f"{path}[{index}]", depth + 1)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _encode(item, tensors, f"{path}[{index}]", depth + 1)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, dict):
        pairs = []
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise ValueError(f"Unsupported dictionary key at {path}.")
            pairs.append([
                _encode(key, tensors, f"{path}.key[{index}]", depth + 1),
                _encode(item, tensors, f"{path}[{key!r}]", depth + 1),
            ])
        return {"type": "dict", "items": pairs}
    raise ValueError(
        f"Unsupported conditioning value {type(value).__name__} at {path}. "
        "Only tensors, dictionaries, lists, tuples, and JSON scalar values are supported."
    )


def _decode(tree: Any, tensors: dict[str, torch.Tensor], used: set[str], depth: int = 0):
    if depth > _MAX_DEPTH or not isinstance(tree, dict):
        raise ValueError("Invalid or excessively nested conditioning checkpoint tree.")
    kind = tree.get("type")
    if kind == "tensor":
        key = tree.get("key")
        if not isinstance(key, str) or key not in tensors:
            raise ValueError("Conditioning checkpoint references a missing tensor.")
        used.add(key)
        return tensors[key]
    if kind == "none":
        return None
    if kind == "bool" and isinstance(tree.get("value"), bool):
        return tree["value"]
    if kind == "str" and isinstance(tree.get("value"), str):
        return tree["value"]
    if kind == "int":
        try:
            return int(tree["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid integer in conditioning checkpoint.") from exc
    if kind == "float":
        try:
            value = float(tree["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid float in conditioning checkpoint.") from exc
        if not math.isfinite(value):
            raise ValueError("Non-finite float in conditioning checkpoint.")
        return value
    if kind in ("list", "tuple"):
        items = tree.get("items")
        if not isinstance(items, list):
            raise ValueError("Invalid sequence in conditioning checkpoint.")
        decoded = [_decode(item, tensors, used, depth + 1) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        pairs = tree.get("items")
        if not isinstance(pairs, list):
            raise ValueError("Invalid mapping in conditioning checkpoint.")
        result = {}
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid mapping entry in conditioning checkpoint.")
            key = _decode(pair[0], tensors, used, depth + 1)
            value = _decode(pair[1], tensors, used, depth + 1)
            try:
                result[key] = value
            except TypeError as exc:
                raise ValueError("Unhashable key in conditioning checkpoint.") from exc
        return result
    raise ValueError("Unknown value type in conditioning checkpoint.")


def _reserve_path(requested: Path) -> Path:
    """Claim an unused output filename without overwriting another checkpoint."""
    base = requested
    suffix = 1
    while True:
        candidate = base if suffix == 1 else base.with_stem(f"{base.stem}_{suffix}")
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            suffix += 1
            continue
        os.close(descriptor)
        return candidate


def save_conditioning(positive: Any, negative: Any, filename: str, settings: str) -> Path:
    settings_json = _json_settings(settings)
    tensors: dict[str, torch.Tensor] = {}
    tree = {
        "positive": _encode(positive, tensors, "positive"),
        "negative": _encode(negative, tensors, "negative"),
    }
    metadata = {
        "format": _FORMAT,
        "version": _VERSION,
        _TREE_KEY: json.dumps(tree, ensure_ascii=False, separators=(",", ":")),
        _SETTINGS_KEY: settings_json,
    }
    requested = output_path(filename)
    requested.parent.mkdir(parents=True, exist_ok=True)
    path = _reserve_path(requested)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        safetensors.torch.save_file(tensors, str(temporary), metadata=metadata)
        # The final path was reserved exclusively above, so concurrent saves
        # become separate suffixed files instead of replacing one another.
        os.replace(temporary, path)
        temporary = None
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_conditioning(filename: str):
    path = output_path(filename)
    with safetensors.torch.safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != _FORMAT or metadata.get("version") != _VERSION:
            raise ValueError("This file is not a supported H3 conditioning checkpoint.")
        try:
            tree = json.loads(metadata[_TREE_KEY])
            settings = _json_settings(metadata[_SETTINGS_KEY])
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("H3 conditioning checkpoint metadata is incomplete.") from exc
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    used: set[str] = set()
    if not isinstance(tree, dict) or set(tree) != {"positive", "negative"}:
        raise ValueError("H3 conditioning checkpoint has an invalid root tree.")
    positive = _decode(tree["positive"], tensors, used)
    negative = _decode(tree["negative"], tensors, used)
    if used != set(tensors):
        raise ValueError("H3 conditioning checkpoint contains unreferenced tensors.")
    return positive, negative, settings


class H3ConditioningSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "filename": ("STRING", {"default": "h3_conditioning/take_001.safetensors"}),
                "settings": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {"negative": ("CONDITIONING",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "report")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Save H3 conditioning tensors and metadata in a safe, lossless safetensors checkpoint."

    def save(self, positive, filename, settings="{}", negative=None):
        path = save_conditioning(positive, negative, filename, settings)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        return str(path.relative_to(output_root)), f"Saved conditioning checkpoint ({_VERSION})."


class H3ConditioningLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": "h3_conditioning/take_001.safetensors"})}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "settings", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load saved H3 conditioning without pickle; tensors are loaded on CPU for ComfyUI to place as needed."

    @classmethod
    def IS_CHANGED(cls, filename):
        path = output_path(filename)
        if not path.exists():
            return float("nan")
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, filename):
        positive, negative, settings = load_conditioning(filename)
        return positive, negative if negative is not None else [], settings, "Loaded H3 conditioning checkpoint."


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls for cls in (H3ConditioningSave, H3ConditioningLoad)
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ConditioningSave": "H3 Continuity · Save Conditioning",
    "H3ConditioningLoad": "H3 Continuity · Load Conditioning",
}
