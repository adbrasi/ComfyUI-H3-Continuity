import pytest
import torch

from h3_continuity.nodes import H3ContinuityAssemble
from h3_continuity.retake import H3RetakeAssemble


@pytest.mark.parametrize('retake', [False, True])
def test_assembly_resizes_and_center_crops_without_changing_timing_or_audio(retake):
    source = torch.zeros(10, 4, 12, 3)
    source[:, :, :4, 0] = 1
    source[:, :, 8:, 0] = 1
    source[:, :, 4:8, 1] = 1
    source[:, :, 4:8, 2] = torch.arange(10).view(10, 1, 1) / 10
    before = source.clone()
    generated = torch.zeros(14, 8, 8, 3)
    generated[..., 0] = 1
    audio = {'waveform': torch.rand(1, 2, 20000), 'sample_rate': 48000}
    if retake:
        plan = {'frames': 10, 'start': 3, 'end': 7, 'edit_audio': False}
        result, sound, report = H3RetakeAssemble().assemble(generated, source, plan, source_audio=audio)
        assert result.shape == (10, 8, 8, 3)
        assert torch.equal(result[3:7], generated[3:7])
        protected = [0, 1, 2, 7, 8, 9]
    else:
        plan = {'context_frames': 5, 'target_frames': 14}
        result, sound, report = H3ContinuityAssemble().assemble(generated, plan, source_images=source, source_audio=audio)
        assert result.shape == (19, 8, 8, 3)
        assert torch.equal(result[10:], generated[5:])
        protected = list(range(10))
    assert result[protected, :, :, 0].count_nonzero() == 0
    assert torch.all(result[protected, :, :, 1] == 1)
    assert torch.allclose(result[protected, :, :, 2].mean((1, 2)), torch.tensor(protected) / 10, atol=1/255)
    assert torch.equal(sound['waveform'][..., :20000], audio['waveform'])
    assert sound['waveform'].shape[-1] == len(result) * 2000
    assert torch.equal(source, before)
    assert 'resized to 8×8' in report
