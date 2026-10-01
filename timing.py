"""H3 token boundaries and delivery handovers, in 24 fps pixel frames."""
from comfy.ldm.minimax.model import FRAME_PER_TOKEN


def frame_edges(steps):
    edges = [0]
    for k in range(steps):
        edges.append(edges[-1] + FRAME_PER_TOKEN[k % 5])
    return edges


def handover(mask, steps, side):
    # Partial spatial protection cannot establish a full-frame handover.
    held = (mask[:, :, :steps] == 0).flatten(3).all(-1).all(0).all(0).tolist()
    edges = frame_edges(steps)
    if side == 'left':
        first = next((i for i, value in enumerate(held) if not value), steps)
        return edges[first]
    last = next((i for i in range(steps - 1, -1, -1) if not held[i]), -1)
    return edges[last + 1]
