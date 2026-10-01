#!/usr/bin/env python3
"""Convert a ComfyUI API prompt JSON graph to an importable UI workflow."""

import argparse
import json
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

WIDGET_TYPES = {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO"}


def object_info(url):
    with urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=15) as response:
        return json.load(response)


def linked(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


def spec_type(spec):
    return "COMBO" if isinstance(spec, list) and spec and isinstance(spec[0], list) else (
        spec[0] if isinstance(spec, list) else spec)


def convert(graph, schemas):
    ids = list(graph)
    if not all(str(node_id).isdigit() for node_id in ids):
        raise ValueError("ComfyUI API node IDs must be numeric strings.")
    idmap = {str(node_id): int(node_id) for node_id in ids}
    nodes, links, node_by_id = [], [], {}
    depths, pending = {}, set(ids)
    while pending:
        ready = [key for key in pending if all(
            not linked(value) or value[0] in depths
            for value in graph[key]["inputs"].values())]
        if not ready:
            raise ValueError("API graph contains a cycle or a link to a missing node.")
        for key in ready:
            depths[key] = max((depths[src] + 1 for value in graph[key]["inputs"].values()
                               if linked(value) for src in [value[0]]), default=0)
            pending.remove(key)

    by_depth = defaultdict(list)
    for key in ids:
        by_depth[depths[key]].append(key)
    positions = {}
    for depth, keys in by_depth.items():
        for row, key in enumerate(keys):
            positions[key] = [depth * 460 + 40, row * 300 + 40]

    next_link = 1
    for order, key in enumerate(sorted(ids, key=lambda k: (depths[k], int(k)))):
        api_node = graph[key]
        name, values = api_node["class_type"], api_node["inputs"]
        if name not in schemas:
            raise ValueError(f"Node {name!r} is missing from /object_info.")
        schema = schemas[name]
        input_schema = schema.get("input", {})
        ordered_names = [n for group in ("required", "optional")
                         for n in schema.get("input_order", {}).get(group, input_schema.get(group, {}))]
        specs = {**input_schema.get("required", {}), **input_schema.get("optional", {})}
        socket_defs, widget_values, slot_by_name = [], [], {}

        # Expand ComfyUI autogrow groups from the flattened API socket names.
        dynamic = {}
        for group_name, group_spec in specs.items():
            if spec_type(group_spec) == "COMFY_AUTOGROW_V3":
                dynamic[group_name] = group_spec[1]
        for api_name, value in values.items():
            if linked(value) and api_name not in specs:
                parent, _, socket_name = api_name.partition(".")
                dynamic_schema = dynamic.get(parent, {})
                template = dynamic_schema.get("template", {}).get("input", {}).get("required", {})
                prefix = dynamic_schema.get("prefix", "")
                if not socket_name.startswith(prefix) or not template:
                    raise ValueError(f"Cannot resolve dynamic socket {name}.{api_name}.")
                specs[api_name] = next(iter(template.values()))
                ordered_names.append(api_name)

        for input_name in ordered_names:
            if input_name not in specs:
                continue
            spec = specs[input_name]
            kind = spec_type(spec)
            value = values.get(input_name)
            if kind == "COMFY_AUTOGROW_V3":
                continue
            if kind in WIDGET_TYPES:
                if linked(value):
                    raise ValueError(f"Widget input {name}.{input_name} cannot be linked.")
                metadata = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}
                if value is None:
                    options = spec[0] if isinstance(spec, list) else None
                    value = metadata.get("default", options[0] if isinstance(options, list) and options else None)
                widget_values.append(value)
                if metadata.get("control_after_generate"):
                    widget_values.append("fixed")
                continue
            is_optional = input_name not in input_schema.get("required", {})
            socket = {"name": input_name, "type": kind, "link": None}
            if is_optional or "." in input_name:
                socket["shape"] = 7
            if "." in input_name:
                socket["label"] = input_name.rsplit(".", 1)[1]
            slot_by_name[input_name] = len(socket_defs)
            socket_defs.append(socket)

        output_types = schema.get("output", [])
        output_names = schema.get("output_name", output_types)
        outputs = [{"name": output_names[i] if i < len(output_names) else output_types[i],
                    "type": output_types[i], "links": [], "slot_index": i}
                   for i in range(len(output_types))]
        for input_name, value in values.items():
            if not linked(value):
                continue
            source_id, source_slot = value
            if source_id not in graph:
                raise ValueError(f"Link {key}.{input_name} references missing node {source_id}.")
            target_slot = slot_by_name.get(input_name)
            if target_slot is None:
                raise ValueError(f"Linked input {name}.{input_name} has no UI socket.")
            source_schema = schemas[graph[source_id]["class_type"]]
            if source_slot < 0 or source_slot >= len(source_schema.get("output", [])):
                raise ValueError(f"Source {source_id} has no output slot {source_slot}.")
            out_type = source_schema["output"][source_slot]
            link_id = next_link
            next_link += 1
            links.append([link_id, idmap[source_id], source_slot, idmap[key], target_slot, out_type])
            socket_defs[target_slot]["link"] = link_id
            node_by_id[idmap[source_id]]["outputs"][source_slot]["links"].append(link_id)

        ui_node = {
            "id": idmap[key], "type": name, "pos": positions[key],
            "size": [360, max(100, 78 + 26 * max(len(socket_defs), len(widget_values)))],
            "flags": {}, "order": order, "mode": 0,
            "inputs": socket_defs, "outputs": outputs,
            "properties": {"Node name for S&R": name}, "widgets_values": widget_values,
        }
        nodes.append(ui_node)
        node_by_id[idmap[key]] = ui_node

    result = {"id": str(uuid.uuid4()), "revision": 0,
            "last_node_id": max(idmap.values(), default=0), "last_link_id": next_link - 1,
            "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {}, "version": 0.4}
    assert len({node["id"] for node in nodes}) == len(nodes)
    assert {link[0] for link in links} == {link_id for node in nodes
                                           for output in node["outputs"]
                                           for link_id in output["links"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    parser.add_argument("--object-info", type=Path, help="Use a saved /object_info JSON response")
    args = parser.parse_args()
    graph = json.loads(args.api_json.read_text(encoding="utf-8"))
    schemas = json.loads(args.object_info.read_text(encoding="utf-8")) if args.object_info else object_info(args.comfy)
    workflow = convert(graph, schemas)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(workflow['nodes'])} nodes and {len(workflow['links'])} links to {args.output_json}")


if __name__ == "__main__":
    main()
