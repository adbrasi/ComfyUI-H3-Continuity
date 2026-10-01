from types import SimpleNamespace

import pytest
import torch
from h3_continuity.retake import H3RetakePrepare, H3RetakeAssemble, temporal_mask, differential_mask
from h3_continuity.nodes import H3ContinuityPrepare, NestedTensor, pixel_frames


def source(t=37):
    n = pixel_frames(t)
    return {'samples': NestedTensor([torch.randn(1,24,t,2,2), torch.randn(1,32,2,round(n*5/3))])}


def test_hard_mask_edits_only_middle_and_preserves_input():
    src = source(); v,a = src['samples'].unbind(); before=v.clone()
    out,plan,_,_=H3RetakePrepare().prepare(34,90,feather_frames=0,source_latent=src)
    vm,am=out['noise_mask'].unbind()
    assert torch.equal(v,before)
    assert torch.all(vm[:,:,:10]==0) and torch.all(vm[:,:,10:27]==1) and torch.all(vm[:,:,27:]==0)
    assert am.count_nonzero()==0
    assert plan['start']==34 and plan['end']==90


def test_feather_offset_and_grow():
    weights,lo,hi=temporal_mask(124,34,90,3,2,8,'smoothstep')
    assert (lo,hi)==(35,95)
    assert weights[:35].count_nonzero()==0 and weights[95:].count_nonzero()==0
    assert 0<weights[35]<weights[39]<1 and weights[50]==1
    assert weights[35]==weights[94]


def test_assemble_restores_original_pixels_and_audio():
    src=source();_,plan,_,_=H3RetakePrepare().prepare(34,90,source_latent=src)
    original=torch.rand(124,2,2,3);generated=torch.zeros_like(original)
    audio={'waveform':torch.rand(1,1,248000),'sample_rate':48000}
    out,sound,_=H3RetakeAssemble().assemble(generated,original,plan,source_audio=audio)
    assert torch.equal(out[:34],original[:34]) and torch.equal(out[90:],original[90:])
    assert out[34:90].count_nonzero()==0
    assert torch.equal(sound['waveform'],audio['waveform'])


@pytest.mark.parametrize('strength',[0,.5,1])
def test_differential_keeps_hard_regions_at_all_sigmas(strength):
    sampling=SimpleNamespace(sigma_min=0.,timestep=lambda x:x)
    mask=torch.tensor([0.,.25,.75,1.]);sigmas=torch.tensor([1.,.5,0.])
    for sigma in [1.,.5,0.]:
        result=differential_mask(mask,torch.tensor([sigma]),sampling,sigmas,strength)
        assert result[0]==0 and result[-1]==1
        if strength==0:assert torch.equal(result,mask)
    early=differential_mask(mask,torch.tensor([.9]),sampling,sigmas,strength)
    late=differential_mask(mask,torch.tensor([.1]),sampling,sigmas,strength)
    assert torch.all(late>=early)


def test_default_continuation_stays_identical_and_feather_is_opt_in():
    src=source();target=source();positive=[[torch.zeros(1,2,3),{}]]
    node=H3ContinuityPrepare()
    _,base,_,_=node.prepare(positive,target,source_latent=src,method='pinned_prefix',audio_context_seconds=0)
    _,disabled,_,_=node.prepare(positive,target,source_latent=src,method='pinned_prefix',audio_context_seconds=0,feather_frames=0)
    assert torch.equal(base['noise_mask'].tensors[0],disabled['noise_mask'].tensors[0])
    _,soft,_,_=node.prepare(positive,target,source_latent=src,method='pinned_prefix',audio_context_seconds=0,feather_frames=8)
    mask=soft['noise_mask'].tensors[0][0,0,:,0,0]
    assert mask[:4].count_nonzero()==0 and torch.any((mask[:7]>0)&(mask[:7]<1))
    assert torch.all(mask[7:]==1)


def test_external_frames_are_padded_without_shortening_delivery():
    class VAE:
        def encode(self, frames):
            assert len(frames)==73
            assert torch.equal(frames[-1],frames[-2])
            return torch.zeros(1,24,22,2,2)
    images=torch.rand(72,32,32,3)
    out,plan,_,_=H3RetakePrepare().prepare(22,56,source_images=images,vae=VAE())
    assert plan['frames']==72 and plan['target_frames']==73
    assert out['noise_mask'].tensors[0][:,:,-1].count_nonzero()==0
    edited,sound,_=H3RetakeAssemble().assemble(torch.zeros(73,32,32,3),images,plan)
    assert len(edited)==72 and sound['waveform'].shape[-1]==96000


def test_audio_retake_keeps_original_outside_window():
    src=source();_,plan,_,_=H3RetakePrepare().prepare(34,90,feather_frames=8,source_latent=src,edit_audio=True)
    frames=torch.zeros(124,2,2,3)
    audio={'waveform':torch.ones(1,1,248000),'sample_rate':48000}
    generated={'waveform':torch.zeros(1,2,165333),'sample_rate':32000}
    _,sound,_=H3RetakeAssemble().assemble(frames,frames,plan,audio=generated,source_audio=audio)
    assert torch.all(sound['waveform'][...,:68000]==1) and torch.all(sound['waveform'][...,180000:]==1)
    assert sound['waveform'][...,120000].item()==0


def test_differential_adapter_sends_same_mask_to_sampler_and_h3():
    from h3_continuity.retake import H3RetakeDifferentialDiffusion
    from comfy.model_base import MiniMaxH3
    import comfy.utils
    base=MiniMaxH3.__new__(MiniMaxH3);torch.nn.Module.__init__(base)
    base.diffusion_model=SimpleNamespace(patch_size=(1,2,2))
    base.model_sampling=SimpleNamespace(sigma_min=0.,timestep=lambda x:x)
    class Patcher:
        def __init__(self,options=None):self.model=base;self.model_options=options or {}
        def clone(self):return Patcher(dict(self.model_options))
        def set_model_denoise_mask_function(self,fn):self.model_options['denoise_mask_function']=fn
        def set_model_unet_function_wrapper(self,fn):self.model_options['model_function_wrapper']=fn
    latent,_,_,_=H3RetakePrepare().prepare(34,90,feather_frames=8,source_latent=source())
    original=Patcher();patched,=H3RetakeDifferentialDiffusion().apply(original,latent,torch.tensor([1.,.5,0.]))
    assert original.model_options=={}
    v,a=latent['samples'].unbind();vm,am=latent['noise_mask'].unbind()
    packed=comfy.utils.pack_latents([vm.expand_as(v),am.expand_as(a)])[0]
    for sigma in [torch.tensor([1.]),torch.tensor([.5]),torch.tensor([0.])]:
        expected=patched.model_options['denoise_mask_function'](sigma,packed,{})
        c=patched.model_options['model_function_wrapper'](lambda x,t,**kwargs:kwargs,{'input':packed,'timestep':sigma,'c':{},'cond_or_uncond':[0]})
        values=base._denoise_mask_values(expected,[v.shape,a.shape])
        for k in values:assert torch.equal(c[k],values[k])


def spatial_source(height=4, width=4):
    src = source()
    video, audio = src['samples'].unbind()
    src['samples'] = NestedTensor([torch.randn(1, 24, video.shape[2], height, width), audio])
    return src


def test_static_spatial_mask_snaps_to_native_patches_and_restores_source_pixels():
    from comfy.ldm.minimax.model import mask_row_values
    mask = torch.zeros(8, 8)
    mask[1, 1] = 1
    src = spatial_source()
    prepared, plan, _, _ = H3RetakePrepare().prepare(35, 39, feather_frames=0, source_latent=src, mask=mask)
    vm, am = prepared['noise_mask'].unbind()
    assert vm.shape == (1, 1, 37, 4, 4)
    assert (plan['start'], plan['end']) == (35, 39)
    assert torch.all(vm[0, 0, 11, :2, :2] == 1)
    assert vm[0, 0, 11].sum() == 4
    # Sampler voxel masks and H3 row masks agree: no mixed strengths in a patch.
    expected = vm[0, 0, :, ::2, ::2].flatten()
    assert torch.equal(mask_row_values(vm[0, 0], 37, 4, 4), expected)
    assert am.count_nonzero() == 0
    original = torch.rand(124, 8, 8, 3)
    generated = torch.full_like(original, -1)
    edited, _, _ = H3RetakeAssemble().assemble(generated, original, plan)
    assert torch.all(edited[35:39, :4, :4] == -1)
    protected = torch.ones(124, 8, 8, dtype=torch.bool)
    protected[35:39, :4, :4] = False
    assert torch.equal(edited[protected], original[protected])


def test_moving_mask_pools_only_frames_inside_requested_interval():
    mask = torch.zeros(124, 8, 8)
    mask[34, 7, 7] = 1  # This frame is outside [35, 40).
    mask[36, 0, 0] = .5
    mask[39, 7, 7] = 1
    prepared, plan, _, _ = H3RetakePrepare().prepare(35, 40, feather_frames=0, source_latent=spatial_source(), mask=mask)
    vm = prepared['noise_mask'].tensors[0][0, 0]
    assert (plan['start'], plan['end']) == (35, 43)
    assert torch.all(vm[11, :2, :2] == .5) and vm[11, 2:, 2:].count_nonzero() == 0
    assert torch.all(vm[12, 2:, 2:] == 1) and vm[12, :2, :2].count_nonzero() == 0
    original = torch.rand(124, 8, 8, 3)
    generated = torch.zeros_like(original)
    edited, _, _ = H3RetakeAssemble().assemble(generated, original, plan)
    assert torch.equal(edited[34], original[34])
    assert torch.equal(edited[35:39, 4:, 4:], original[35:39, 4:, 4:])
    assert edited[39:43, 4:, 4:].count_nonzero() == 0
    assert torch.equal(edited[39:43, :4, :4], original[39:43, :4, :4])


@pytest.mark.parametrize('mask,strength', [(torch.zeros(1, 8, 8), 1.), (torch.ones(8, 8), 0.), (None, 0.)])
def test_zero_visual_support_is_safe_and_preserves_all_source_pixels(mask, strength):
    prepared, plan, preview, _ = H3RetakePrepare().prepare(34, 90, source_latent=spatial_source(), mask=mask, video_strength=strength)
    assert prepared['noise_mask'].tensors[0].count_nonzero() == 0
    assert (plan['start'], plan['end']) == (0, 0)
    assert preview.count_nonzero() == 0
    original = torch.rand(124, 8, 8, 3)
    edited, _, _ = H3RetakeAssemble().assemble(torch.zeros_like(original), original, plan)
    assert torch.equal(edited, original)


def test_video_strength_keeps_audio_edit_independent():
    prepared, plan, _, _ = H3RetakePrepare().prepare(34, 90, feather_frames=0, source_latent=source(), video_strength=0., edit_audio=True)
    vm, am = prepared['noise_mask'].unbind()
    assert vm.count_nonzero() == 0 and am.count_nonzero() > 0
    assert plan['edit_audio'] and plan['audio_start'] == 34


@pytest.mark.parametrize('mask', [torch.zeros(2, 8, 8), torch.zeros(1, 1, 8, 8), torch.full((8, 8), float('nan'))])
def test_invalid_spatial_masks_raise_clear_error(mask):
    with pytest.raises(ValueError, match='MASK'):
        H3RetakePrepare().prepare(34, 90, source_latent=source(), mask=mask)


def test_reversed_interval_cannot_be_hidden_by_grow():
    with pytest.raises(ValueError, match='end_frame'):
        H3RetakePrepare().prepare(50, 40, grow_frames=20, source_latent=source())


def test_retake_halo_unlocks_only_expanded_temporal_support():
    prepared, plan, _, _ = H3RetakePrepare().prepare(39, 56, grow_frames=4, feather_frames=0, source_latent=source())
    assert (plan['requested_start'], plan['requested_end']) == (35, 60)
    assert (plan['start'], plan['end']) == (35, 60)
    vm = prepared['noise_mask'].tensors[0]
    assert vm[:, :, :11].count_nonzero() == 0 and vm[:, :, 18:].count_nonzero() == 0


def test_audio_retake_uses_selected_linear_feather():
    _, plan, _, _ = H3RetakePrepare().prepare(34, 90, feather_frames=8, curve='linear', source_latent=source(), edit_audio=True)
    images = torch.zeros(124, 2, 2, 3)
    original = {'waveform': torch.ones(1, 1, 248000), 'sample_rate': 48000}
    generated = {'waveform': torch.zeros(1, 1, 248000), 'sample_rate': 48000}
    _, sound, _ = H3RetakeAssemble().assemble(images, images, plan, audio=generated, source_audio=original)
    # At 36 frames the linear ramp is 2/8; smoothstep would be 0.15625.
    assert sound['waveform'][0, 0, 72000].item() == pytest.approx(.75)
