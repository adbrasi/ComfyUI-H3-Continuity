import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'custom_nodes'))
import comfy.cli_args
comfy.cli_args.args.cpu = True
from h3_continuity.nodes import (H3ContinuityPrepare, H3ContinuityAssemble, H3ContinuitySave,
                                 H3ContinuityLoad, NestedTensor, pixel_frames, check_layout)


def latent(t=37):
    frames = pixel_frames(t)
    return {'samples': NestedTensor([torch.randn(1, 24, t, 2, 2), torch.randn(1, 32, 2, round(frames * 5/3))])}


def cond():
    return [[torch.zeros(1, 2, 3), {'minimax_refs': [{'kind': 'audio', 'ref_audio_t': 4}],
                                 'minimax_keyframes': [{'resolved_frame_index': 123, 'latent': torch.zeros(1, 24, 1, 2, 2)}]}]]


@pytest.mark.parametrize('t', [17, 22, 27, 32, 37, 42, 72])
@pytest.mark.parametrize('n', [5, 22, 39, 56])
def test_latent_tail_and_audio_grid(t, n):
    source, target, positive = latent(t), latent(), cond()
    result, prepared, plan, report = H3ContinuityPrepare().prepare(positive, target, n, source_latent=source)
    kfs = result[0][1]['minimax_keyframes']
    assert kfs[0]['resolved_frame_index'] == 123
    assert result[0][1]['minimax_refs'] == positive[0][1]['minimax_refs']
    tail = kfs[1]['latent']
    assert torch.equal(tail, source['samples'].tensors[0][:, :, -tail.shape[2]:])
    assert tail.untyped_storage().nbytes() == tail.numel() * tail.element_size()
    audio = kfs[2]
    assert abs(audio['resolved_frame_index']*5/3 - round(audio['resolved_frame_index']*5/3)) < 1e-6
    assert abs(audio['resolved_frame_index']*5/3 + audio['audio_latent'].shape[-1] - n*5/3) <= 1
    assert len(positive[0][1]['minimax_keyframes']) == 1
    assert 'h3_end_seconds' not in target


def test_native_layout():
    check_layout()


class FakeVAE:
    audio_sample_rate = 32000
    def encode(self, x):
        self.seen = x.clone()
        if x.ndim == 4:
            return torch.zeros(1, 24, 2 + 5*((len(x)-5)//17), 2, 2)
        return torch.zeros(1, 32, 2, round(x.shape[1]/32000*40))


def test_external_72_frames_keeps_the_end():
    images = (torch.arange(72.)/100)[:, None, None, None].expand(-1, 32, 32, 3)
    vae = FakeVAE()
    _, _, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), 56, source_images=images, vae=vae)
    assert vae.seen[0, 0, 0, 0].item() == pytest.approx(.16)
    assert vae.seen[-1, 0, 0, 0].item() == pytest.approx(.71)
    assert plan['source_end_seconds'] == 3


def test_binary_pin_is_exact_and_inputs_unmodified():
    source, target = latent(), latent()
    original = target['samples'].tensors[0].clone()
    _, out, _, _ = H3ContinuityPrepare().prepare(cond(), target, method='pinned_prefix', source_latent=source)
    v, a = out['samples'].tensors
    mv, ma = out['noise_mask'].tensors
    assert torch.equal(v[:, :, :7], source['samples'].tensors[0][:, :, -7:])
    assert mv[:, :, :7].count_nonzero() == 0
    assert torch.all(mv[:, :, 7:] == 1) and torch.all(ma == 1)
    assert torch.equal(target['samples'].tensors[0], original)


@pytest.mark.parametrize('frames', [124, 141, 158])
def test_assembly_no_duplicate_frames_or_audio_drift(frames):
    source = torch.arange(72.)[:, None, None, None]
    generated = torch.arange(frames)[:, None, None, None].float() + 1000
    audio = {'waveform': torch.ones(1, 2, round(frames*5/3)*800), 'sample_rate': 32000}
    source_audio = {'waveform': torch.full((1, 2, 96000), 0.5), 'sample_rate': 32000}
    plan = {'context_frames': 22, 'target_frames': frames}
    im, au, report = H3ContinuityAssemble().assemble(generated, plan, audio, source, source_audio)
    assert torch.equal(im[:72], source)
    assert im[72].item() == 1022
    assert len(im) == 72+frames-22
    assert au['waveform'].shape[-1] == round(len(im)/24*32000)
    assert torch.equal(au['waveform'][..., :96000], source_audio['waveform'])


def test_original_soundtrack_samples_preserved():
    track = {'waveform': torch.rand(1, 1, 48000*12), 'sample_rate': 48000}
    _, _, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(),
                                                soundtrack=track, audio_vae=FakeVAE())
    im, au, _ = H3ContinuityAssemble().assemble(torch.zeros(124, 2, 2, 3), plan)
    assert torch.equal(au['waveform'], track['waveform'][..., 248000:452000])


def test_save_load_exact_and_safe_paths(tmp_path):
    source = latent()
    source['samples'] = source['samples'].to(dtype=torch.bfloat16)
    with patch('folder_paths.get_output_directory', return_value=str(tmp_path)):
        name, = H3ContinuitySave().save(source, 'chain/a.safetensors', {'end_seconds': 50})
        name2, = H3ContinuitySave().save(source, 'chain/a.safetensors')
        assert name != name2
        restored, = H3ContinuityLoad().load(name)
        assert restored['h3_end_seconds'] == 50
        for a, b in zip(source['samples'].tensors, restored['samples'].tensors):
            assert a.dtype == b.dtype and torch.equal(a, b)
        with pytest.raises(ValueError):
            H3ContinuitySave().save(source, '../escape.safetensors')


def test_short_soundtrack_pads_missing_samples_with_silence():
    _, _, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(), audio_vae=FakeVAE(),
        soundtrack={'waveform': torch.ones(1, 2, 32000), 'sample_rate': 32000})
    assert plan['soundtrack_segment']['waveform'].shape[-1] == 136000
    assert plan['soundtrack_segment']['waveform'].count_nonzero() == 0


def test_original_track_locks_audio_and_preserves_prior_mask():
    target = latent()
    target['noise_mask'] = NestedTensor([torch.full((1,1,37,2,2),.5), torch.ones(1,1,2,207)])
    before = target['samples'].tensors[1].clone()
    _, out, _, _ = H3ContinuityPrepare().prepare(cond(), target, source_latent=latent(),
        soundtrack={'waveform': torch.ones(1,2,48000*12), 'sample_rate':48000}, audio_vae=FakeVAE())
    assert out['noise_mask'].tensors[1].count_nonzero() == 0
    assert torch.all(out['noise_mask'].tensors[0] == .5)
    assert torch.all(target['noise_mask'].tensors[1] == 1)
    assert torch.equal(target['samples'].tensors[1], before)


def test_pinned_av_uses_only_aligned_complete_audio_rows():
    source = latent()
    _, out, _, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=source, method='pinned_av')
    assert torch.equal(out['samples'].tensors[1][..., :36], source['samples'].tensors[1][...,170:206])
    assert torch.all(out['noise_mask'].tensors[1][...,:36] == 0)
    assert torch.all(out['noise_mask'].tensors[1][...,36:] == 1)


def test_import_fps_keeps_duration_and_audio():
    from types import SimpleNamespace
    from h3_continuity.nodes import H3ContinuityImport
    image = torch.arange(90.)[:,None,None,None].expand(-1,32,32,3)
    wave = torch.rand(1,2,144000)
    class Video:
        def get_components(self):
            return SimpleNamespace(images=image, frame_rate=30, audio={'waveform':wave,'sample_rate':48000})
    images, audio, _ = H3ContinuityImport().convert(Video(),32,32)
    assert len(images)==72 and images[-1,0,0,0]==88
    assert torch.equal(audio['waveform'],wave)


@pytest.mark.parametrize('requested,used', [(90,90),(107,107),(112,107),(124,124),(243,243),(362,362)])
def test_extended_context_preserves_tail_and_reports_delivered_frames(requested,used):
    source,target=latent(152),latent(177)
    positive,out,plan,report=H3ContinuityPrepare().prepare(cond(),target,context_frames=requested,
        source_latent=source,method='pinned_prefix',audio_context_seconds=0)
    steps=2+5*((used-5)//17)
    assert plan['context_frames']==used
    assert plan['target_frames']==600
    assert torch.equal(out['samples'].tensors[0][:,:,:steps],source['samples'].tensors[0][:,:,-steps:])
    assert torch.all(out['noise_mask'].tensors[0][:,:,:steps]==0)
    assert torch.all(out['noise_mask'].tensors[0][:,:,steps:]==1)
    if requested!=used:assert f'Requested {requested}' in report
