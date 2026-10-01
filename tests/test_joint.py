from types import SimpleNamespace

import pytest
import torch

import comfy.conds
import comfy.utils
from comfy.model_base import MiniMaxH3
from comfy.nested_tensor import NestedTensor
from h3_continuity.joint import (H3JointRefine, H3JointWindowHandler,
                                 window_grid, window_memory_budget)
from h3_continuity.nodes import pixel_frames
from h3_continuity.timing import frame_edges


def latent(steps=17):
    video = torch.randn(1, 24, steps, 2, 2)
    audio = torch.randn(1, 32, 2, round(pixel_frames(steps) * 5 / 3))
    return {'samples': NestedTensor([video, audio])}


def conditions(shapes, payload=None):
    mc = {'latent_shapes': comfy.conds.CONDConstant(shapes),
          'c_crossattn': comfy.conds.CONDRegular(torch.zeros(1, 2, 8)),
          'minimax_payload': comfy.conds.CONDConstant(payload or {})}
    return [[{'model_conds': mc}], None]


def model():
    base = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(base)
    return base


@pytest.mark.parametrize('total,length,stride', [(17, 7, 5), (52, 17, 10), (107, 37, 25), (7, 37, 25)])
def test_window_grid_covers_video_and_audio_without_cycle_shift(total, length, stride):
    windows = window_grid(total, length, stride)
    coverage = torch.zeros(total)
    edges = frame_edges(total)
    audio = torch.zeros(round(edges[-1] * 5 / 3))
    for start, stop in windows:
        assert start % 5 == 0 and (stop - start) % 5 == 2
        assert stop - start <= length
        coverage[start:stop] += 1
        audio[round(edges[start] * 5 / 3):round(edges[stop] * 5 / 3)] += 1
    assert windows[0][0] == 0 and windows[-1][1] == total
    assert coverage.min() > 0 and audio.min() > 0


def test_identity_model_fuses_every_av_value_without_altering_shared_state():
    source = latent()
    tensors = source['samples'].unbind()
    packed, shapes = comfy.utils.pack_latents(tensors)
    before = packed.clone()
    conds = conditions(shapes)
    handler = H3JointWindowHandler(shapes, 7, 5)
    seen = []
    def identity(_, local, window, sigma, options):
        seen.append(window.clone())
        return [window.clone(), torch.zeros_like(window)]
    output = handler.execute(identity, model(), conds, packed, torch.tensor([.5]), {})
    assert len(seen) == 3
    assert torch.allclose(output[0], before, atol=2e-7)
    assert torch.equal(output[1], torch.zeros_like(packed))
    assert torch.equal(packed, before)


def test_distinct_window_predictions_use_normalized_triangular_overlap_weights():
    packed, shapes = comfy.utils.pack_latents(latent()['samples'].unbind())
    handler = H3JointWindowHandler(shapes, 7, 5)
    calls = []
    def prediction(_, conds, window, sigma, options):
        calls.append(float(sigma[0]))
        return [torch.full_like(window, len(calls)), None]
    outputs = handler.execute(prediction, model(), conditions(shapes), packed, torch.tensor([.4]), {})
    video, audio = comfy.utils.unpack_latents(outputs[0], shapes)
    assert calls == pytest.approx([.4] * 3)
    assert video[0, 0, 0, 0, 0].item() == 1
    assert video[0, 0, 5, 0, 0].item() == pytest.approx(4 / 3)
    assert video[0, 0, -1, 0, 0].item() == 3
    assert audio[0, 0, 0, 28].item() == pytest.approx(1.1)


def test_window_conditioning_slices_protection_and_offsets_keyframes_without_mutating_refs():
    packed, shapes = comfy.utils.pack_latents(latent()['samples'].unbind())
    first = {'resolved_frame_index': 0, 'latent': torch.zeros(1, 24, 1, 2, 2)}
    later = {'resolved_frame_index': 35, 'latent': torch.zeros(1, 24, 1, 2, 2)}
    refs = [{'kind': 'image', 'latent': torch.zeros(1, 24, 1, 2, 2), 'latent_h': 2, 'latent_w': 2}]
    conds = conditions(shapes, {'keyframes': [first, later], 'refs': refs, 'seed': 9, 'audio_scale': 4.})
    mc = conds[0][0]['model_conds']
    mc['denoise_mask'] = comfy.conds.CONDRegular(torch.arange(17).view(1, 1, 17, 1, 1).expand(1, 1, 17, 2, 2))
    mc['audio_denoise_mask'] = comfy.conds.CONDRegular(torch.arange(shapes[1][-1]).view(1, 1, 1, -1).expand(1, 1, 2, -1))
    _, local_shapes = comfy.utils.pack_latents([torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 37)])
    local = H3JointWindowHandler(shapes, 7, 5).window_conds(conds[0], 5, 12, 28, 65, local_shapes)
    result = local[0]['model_conds']
    payload = result['minimax_payload'].cond
    assert len(payload['keyframes']) == 1 and payload['keyframes'][0]['resolved_frame_index'] == 18
    assert payload['refs'] is refs and payload['seed'] == 9 and payload['audio_scale'] == 4.
    assert payload['cond_video_latents'][0] is later['latent']
    assert result['denoise_mask'].cond.shape[2] == 7
    assert result['denoise_mask'].cond[0, 0, 0, 0, 0] == 5
    assert result['audio_denoise_mask'].cond.shape[-1] == 37
    assert result['audio_denoise_mask'].cond[0, 0, 0, 0] == 28
    assert later['resolved_frame_index'] == 35 and mc['latent_shapes'].cond == shapes
    assert payload['layout'] is not mc['minimax_payload'].cond.get('layout')


class Patcher:
    def __init__(self, options=None):
        self.model, self.model_options = model(), options or {}
        self.wrappers = []
    def clone(self):
        result = Patcher(dict(self.model_options))
        result.wrappers = list(self.wrappers)
        return result
    def add_wrapper_with_key(self, *args):
        self.wrappers.append(args)


def test_node_preserves_model_and_budgets_only_one_real_video_window():
    original = Patcher()
    refined, report = H3JointRefine().apply(original, latent(), 22, 5)
    assert original.model_options == {} and original.wrappers == []
    assert refined.model_options['context_handler'].length == 7
    assert '3 shared-state windows' in report
    seen = []
    window_memory_budget(lambda _, shape, conds, **kw: seen.append(shape), refined, (1, 1, 10000), {}, model_options=refined.model_options)
    assert seen == [[1, 24, 7, 2, 2]]


@pytest.mark.parametrize('options', [{'context_handler': object()}, {'denoise_mask_function': object()}])
def test_node_rejects_incompatible_sampling_handlers(options):
    with pytest.raises(ValueError):
        H3JointRefine().apply(Patcher(options), latent(), 22, 5)


def test_handler_rejects_different_latent_before_sampling():
    _, shapes = comfy.utils.pack_latents(latent()['samples'].unbind())
    handler = H3JointWindowHandler(shapes, 7, 5)
    _, other = comfy.utils.pack_latents(latent(22)['samples'].unbind())
    with pytest.raises(ValueError, match='same full AV latent'):
        handler.should_use_context(model(), conditions(other), torch.zeros(1), torch.zeros(1), {})


def test_handler_rejects_area_conditioning_instead_of_misaligning_masks():
    _, shapes = comfy.utils.pack_latents(latent()['samples'].unbind())
    conds = conditions(shapes)[0]
    conds[0]['area'] = (1, 1, 0, 0)
    with pytest.raises(ValueError, match='area masks'):
        H3JointWindowHandler(shapes, 7, 5).window_conds(conds, 0, 7, 0, 37, shapes)
