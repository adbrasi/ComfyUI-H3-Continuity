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
