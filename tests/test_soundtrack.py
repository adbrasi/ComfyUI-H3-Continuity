import pytest
import torch

from h3_continuity.nodes import H3ContinuityPrepare, H3ContinuityAssemble
from test_nodes import latent, cond, FakeVAE


class RecordingVAE(FakeVAE):
    def encode(self, wave):
        self.received = wave.clone()
        return super().encode(wave)


@pytest.mark.parametrize('mode', ['full_video', 'continuation_only'])
@pytest.mark.parametrize('seconds', [0, 1, 20])
def test_soundtrack_modes_preserve_recording_trim_excess_and_pad_short_input(mode, seconds):
    sr = 32000
    track = {'waveform': torch.rand(1, 2, sr * seconds), 'sample_rate': sr}
    original = {'waveform': torch.full((1, 2, round(124 / 24 * sr)), -.75), 'sample_rate': sr}
    _, prepared, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(),
        soundtrack=track, source_audio=original, soundtrack_mode=mode, audio_vae=FakeVAE())
    assert prepared['noise_mask'].tensors[1].count_nonzero() == 0
    generated = torch.zeros(124, 2, 2, 3)
    images, audio, _ = H3ContinuityAssemble().assemble(generated, plan, source_images=generated)
    count = round(len(images) / 24 * sr)
    expected = torch.zeros(1, 2, count)
    offset = 0
    if mode == 'continuation_only':
        offset = original['waveform'].shape[-1]
        expected[..., :offset] = original['waveform']
    copy = min(track['waveform'].shape[-1], count - offset)
    expected[..., offset:offset + copy] = track['waveform'][..., :copy]
    assert torch.equal(audio['waveform'], expected)


def test_full_track_overrides_explicit_source_audio_and_uses_chain_time():
    sr = 32000
    track = {'waveform': torch.arange(sr * 30).reshape(1, 1, -1).float() / (sr * 30), 'sample_rate': sr}
    original = {'waveform': torch.ones(1, 2, sr * 6), 'sample_rate': sr}
    _, _, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(),
        source_end_seconds=20, soundtrack=track, audio_vae=FakeVAE())
    frames = torch.zeros(124, 2, 2, 3)
    _, sound, _ = H3ContinuityAssemble().assemble(frames, plan, source_images=frames, source_audio=original)
    start = round((20 - 124 / 24) * sr)
    assert sound['waveform'].shape[1] == 1
    assert torch.equal(sound['waveform'], track['waveform'][..., start:start + sound['waveform'].shape[-1]])


def test_new_recording_starts_at_join_in_vae_input_and_output():
    sr = 32000
    track = {'waveform': torch.full((1, 1, sr * 20), .25), 'sample_rate': sr}
    original = {'waveform': torch.full((1, 2, sr * 6), -.5), 'sample_rate': sr}
    vae = RecordingVAE()
    _, _, plan, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=latent(),
        source_end_seconds=20, source_audio=original, soundtrack=track,
        soundtrack_mode='continuation_only', audio_vae=vae)
    cut = round(22 / 24 * sr)
    assert torch.all(vae.received[:, :cut] == -.5)
    assert torch.all(vae.received[:, cut:] == .25)
    _, audio, _ = H3ContinuityAssemble().assemble(torch.zeros(124, 2, 2, 3), plan)
    assert torch.all(audio['waveform'] == .25)


def test_new_recording_uses_latent_context_if_source_waveform_is_absent():
    src = latent()
    _, out, _, _ = H3ContinuityPrepare().prepare(cond(), latent(), source_latent=src,
        soundtrack={'waveform': torch.ones(1, 2, 320000), 'sample_rate': 32000},
        soundtrack_mode='continuation_only', audio_vae=FakeVAE())
    assert torch.equal(out['samples'].tensors[1][..., :36], src['samples'].tensors[1][..., 170:206])
