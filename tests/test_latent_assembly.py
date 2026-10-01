import pytest
import torch
from itertools import product

from h3_continuity.latent_assembly import H3ContinuityLatentAssemble
from h3_continuity.nodes import NestedTensor, pixel_frames
from h3_continuity.timing import frame_edges


def latent(steps, value):
    frames = pixel_frames(steps)
    return {'samples': NestedTensor([
        torch.full((1, 24, steps, 2, 2), value),
        torch.full((1, 32, 2, round(frames * 5 / 3)), value),
    ])}


def steps_for_frames(frames):
    return next(steps for steps in range(2, 200, 5) if pixel_frames(steps) == frames)


@pytest.mark.parametrize('handover_steps', [0, 7, 12])
def test_latent_assembly_preserves_source_and_splices_on_native_boundaries(handover_steps):
    source, sampled = latent(37, 1), latent(42, 2)
    source['noise_mask'] = object()
    sampled['noise_mask'] = object()
    plan = {'context_frames': 39, 'target_frames': pixel_frames(42),
            'handover_frame': frame_edges(42)[handover_steps], 'end_seconds': 9.5}

    merged, report = H3ContinuityLatentAssemble().assemble(source, sampled, plan)
    video, audio = merged['samples'].unbind()
    cut = 37 - 12 + handover_steps
    assert video.shape[2] == 67
    assert torch.all(video[:, :, :cut] == 1)
    assert torch.all(video[:, :, cut:] == 2)
    assert audio.shape[-1] == 377
    assert torch.all(audio[..., :207] == 1)
    assert torch.all(audio[..., 207:] == 2)
    assert 'noise_mask' not in merged
    assert merged['h3_end_seconds'] == 9.5
    assert sampled['noise_mask'] is not None
    assert str(video.shape[2]) in report


def test_latent_assembly_rejects_canvas_or_plan_timing_mismatch():
    source, sampled = latent(37, 1), latent(42, 2)
    plan = {'context_frames': 39, 'target_frames': pixel_frames(42),
            'handover_frame': 22}
    sampled['samples'].tensors[0] = torch.zeros(1, 24, 42, 4, 2)
    with pytest.raises(ValueError, match='canvases'):
        H3ContinuityLatentAssemble().assemble(source, sampled, plan)
    sampled = latent(42, 2)
    plan['target_frames'] += 1
    with pytest.raises(ValueError, match='timing'):
        H3ContinuityLatentAssemble().assemble(source, sampled, plan)


@pytest.mark.parametrize('source_frames,target_frames,context', [
    (source, target, context)
    for source, target in product((22, 39, 56, 73, 124), repeat=2)
    for context in (5, 22, 39) if context <= min(source, target)
])
def test_phase_grid_timelines_keep_complete_source_audio(source_frames, target_frames, context):
    source = latent(steps_for_frames(source_frames), 1)
    sampled = latent(steps_for_frames(target_frames), 2)
    plan = {'context_frames': context, 'target_frames': target_frames,
            'handover_frame': context}

    merged, _ = H3ContinuityLatentAssemble().assemble(source, sampled, plan)
    video, audio = merged['samples'].unbind()
    assert pixel_frames(video.shape[2]) == source_frames + target_frames - context
    assert audio.shape[-1] == round((source_frames + target_frames - context) * 5 / 3)
    source_audio = source['samples'].tensors[1]
    assert torch.equal(audio[..., :source_audio.shape[-1]], source_audio)
