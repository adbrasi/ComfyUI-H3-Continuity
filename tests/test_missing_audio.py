from types import SimpleNamespace

import pytest
import torch

from h3_continuity.nodes import H3ContinuityPrepare, H3ContinuityAssemble, H3ContinuityImport, audio_samples
from test_nodes import cond, latent, FakeVAE


def external_prepare(track, source_audio=None, mode='continuation_only'):
    return H3ContinuityPrepare().prepare(cond(), latent(), source_images=torch.zeros(72, 32, 32, 3),
        vae=FakeVAE(), audio_vae=FakeVAE(), source_audio=source_audio, soundtrack=track, soundtrack_mode=mode)


def test_partial_recording_releases_both_uncovered_sides_and_preserves_silent_recording():
    # Silence inside a supplied file is still real audio, not a missing interval.
    _, out, plan, _ = external_prepare({'waveform': torch.zeros(1, 2, 16000), 'sample_rate': 32000})
    mask = out['noise_mask'].tensors[1]
    assert torch.all(mask[..., :36] == 1)
    assert torch.all(mask[..., 37:56] == 0)
    assert torch.all(mask[..., 57:] == 1)
    assert float(mask[0, 0, 0, 36]) == pytest.approx(2/3, abs=1e-5)
    assert float(mask[0, 0, 0, 56]) == pytest.approx(1/3, abs=1e-5)
    assert plan['soundtrack_segment']['waveform'].shape[-1] == 16000
    assert plan['missing_audio'] == 'generate'


@pytest.mark.parametrize('seconds', [0, 1, 6, 20])
def test_full_track_locks_only_available_rows(seconds):
    _, out, _, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(),
        soundtrack={'waveform': torch.zeros(1, 2, seconds*32000), 'sample_rate': 32000}, audio_vae=FakeVAE())
    mask = out['noise_mask'].tensors[1]
    # Source ends at 124/24; sampling starts 22/24 earlier, at 4.25 seconds.
    fixed = max(0, min(207, round((seconds - 4.25)*40)))
    assert torch.all(mask[..., :fixed] == 0)
    assert torch.all(mask[..., fixed:] == 1)


def test_assembly_preserves_recording_and_uses_generated_context_and_tail():
    track = {'waveform': torch.full((1, 1, 16000), -.25), 'sample_rate': 32000}
    _, _, plan, _ = external_prepare(track)
    generated_audio = {'waveform': torch.full((1, 2, 166400), .5), 'sample_rate': 32000}
    frames = torch.zeros(124, 32, 32, 3)
    out, sound, report = H3ContinuityAssemble().assemble(frames, plan, audio=generated_audio, source_images=frames[:72])
    wave = sound['waveform']
    assert len(out) == 174 and wave.shape[-1] == 232000
    window_start = round(50/24*32000)
    assert torch.all(wave[..., :window_start] == 0)  # Source outside this model call.
    assert torch.all(wave[..., window_start:96000] == .5)
    assert torch.all(wave[..., 96000:112000] == -.25)
    assert torch.all(wave[..., 112000:] == .5)
    assert '2.083s' in report and 'before the sampling window' in report
    _, suffix, _ = H3ContinuityAssemble().assemble(frames, plan, audio=generated_audio)
    assert torch.all(suffix['waveform'][..., :16000] == -.25)
    assert torch.all(suffix['waveform'][..., 16000:] == .5)


def test_full_track_ends_then_generated_audio_fills_remainder_without_old_sound():
    track = {'waveform': torch.full((1, 1, 4*32000), -.25), 'sample_rate': 32000}
    _, _, plan, _ = external_prepare(track, mode='full_video')
    frames = torch.zeros(124, 32, 32, 3)
    old = {'waveform': torch.full((1, 2, 96000), -.75), 'sample_rate': 32000}
    generated = {'waveform': torch.full((1, 2, 166400), .5), 'sample_rate': 32000}
    _, sound, _ = H3ContinuityAssemble().assemble(frames, plan, audio=generated, source_images=frames[:72], source_audio=old)
    assert torch.all(sound['waveform'][..., :128000] == -.25)
    assert torch.all(sound['waveform'][..., 128000:] == .5)


def test_import_marks_missing_audio_without_treating_recorded_silence_as_absent():
    class Video:
        def __init__(self, audio): self.audio = audio
        def get_components(self): return SimpleNamespace(images=torch.zeros(72, 32, 32, 3), frame_rate=24, audio=self.audio)
    _, absent, _ = H3ContinuityImport().convert(Video(None), 32, 32)
    assert audio_samples(absent) == 0 and absent['waveform'].shape[-1] == 96000
    _, short, _ = H3ContinuityImport().convert(Video({'waveform': torch.zeros(1, 2, 16000), 'sample_rate': 32000}), 32, 32)
    assert audio_samples(short) == 16000 and short['waveform'].shape[-1] == 96000
    track = {'waveform': torch.ones(1, 2, 16000), 'sample_rate': 32000}
    _, out, _, _ = external_prepare(track, absent)
    assert torch.all(out['noise_mask'].tensors[1][..., :36] == 1)


def test_short_source_sound_does_not_overwrite_generated_audio_with_padding():
    source = {'waveform': torch.full((1, 2, 96000), -.75), 'sample_rate': 32000, 'h3_valid_samples': 80000}
    track = {'waveform': torch.full((1, 1, 16000), -.25), 'sample_rate': 32000}
    _, _, plan, _ = external_prepare(track, source)
    frames = torch.zeros(124, 32, 32, 3)
    generated = {'waveform': torch.full((1, 2, 166400), .5), 'sample_rate': 32000}
    _, audio, _ = H3ContinuityAssemble().assemble(frames, plan, audio=generated, source_images=frames[:72])
    assert torch.all(audio['waveform'][..., :80000] == -.75)
    assert torch.all(audio['waveform'][..., 80000:96000] == .5)
    assert torch.all(audio['waveform'][..., 96000:112000] == -.25)
    assert torch.all(audio['waveform'][..., 112000:] == .5)
