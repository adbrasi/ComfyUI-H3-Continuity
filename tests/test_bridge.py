import pytest
import torch
from h3_continuity.nodes import H3ContinuityPrepare, H3ContinuityAssemble, NestedTensor, pixel_frames
from h3_continuity.bridge import H3BridgePrepare, H3BridgeAssemble


def latent(steps=57):
    return {'samples': NestedTensor([torch.randn(1,24,steps,2,2),
                                    torch.randn(1,32,2,round(pixel_frames(steps)*5/3))])}


def cond():
    return [[torch.zeros(1,2,3), {'minimax_refs': [{'kind':'image','latent':torch.ones(1,24,1,2,2)}]}]]


def test_bridge_hard_pins_are_exact_and_preserve_inputs():
    left,right,target=latent(),latent(),latent()
    before=target['samples'].tensors[0].clone()
    positive,out,plan,_=H3BridgePrepare().prepare(cond(),target,left,right)
    v,a=out['samples'].unbind();vm,am=out['noise_mask'].unbind()
    assert torch.equal(v[:,:,:12],left['samples'].tensors[0][:,:,-12:])
    assert torch.equal(v[:,:,-12:],right['samples'].tensors[0][:,:,:12])
    assert vm[:,:,:12].count_nonzero()==0 and vm[:,:,-12:].count_nonzero()==0
    assert torch.all(vm[:,:,12:-12]==1)
    assert torch.equal(a[...,:65],left['samples'].tensors[1][...,-65:])
    assert torch.equal(a[...,-65:],right['samples'].tensors[1][...,:65])
    assert am[...,:65].count_nonzero()==0 and am[...,-65:].count_nonzero()==0
    assert plan['middle_frames']==114
    assert torch.equal(before,target['samples'].tensors[0])
    assert 'minimax_refs' in positive[0][1]


@pytest.mark.parametrize('feather',[0,8,17,39])
def test_bridge_assembly_delivers_released_rows_and_keeps_duration(feather):
    left,right,target=latent(),latent(),latent()
    _,_,plan,_=H3BridgePrepare().prepare(cond(),target,left,right,feather_frames=feather)
    l=torch.ones(192,2,2,3);r=torch.full_like(l,3);g=torch.full_like(l,2)
    la={'waveform':torch.ones(1,1,192*2000),'sample_rate':48000}
    ra={'waveform':torch.full((1,1,192*2000),3.),'sample_rate':48000}
    ga={'waveform':torch.full((1,2,192*2000),2.),'sample_rate':48000}
    images,audio,_=H3BridgeAssemble().assemble(g,l,r,plan,ga,la,ra)
    assert len(images)==498 and audio['waveform'].shape[-1]==498*2000
    cut=192-39+plan['left_handover']
    assert torch.all(images[:cut]==1) and torch.all(images[cut:306+plan['right_handover']]==2)
    assert torch.all(images[306+plan['right_handover']:] ==3)
    assert torch.all(audio['waveform'][...,:192*2000]==1)
    assert torch.all(audio['waveform'][...,306*2000:]==3)


def test_continuation_feather_is_delivered_instead_of_discarded():
    _,_,plan,_=H3ContinuityPrepare().prepare(cond(),latent(),context_frames=39,method='pinned_prefix',
        source_latent=latent(),feather_frames=17,audio_context_seconds=0)
    assert plan['handover_frame']==22
    source=torch.ones(192,2,2,3);generated=torch.full_like(source,2)
    images,_,_=H3ContinuityAssemble().assemble(generated,plan,source_images=source)
    assert len(images)==345
    assert torch.all(images[:175]==1) and torch.all(images[175:]==2)
    with pytest.raises(ValueError,match='source_images'):
        H3ContinuityAssemble().assemble(generated,plan)


def test_feather_requires_full_frame_protection_for_handover():
    target=latent();vm=torch.ones(1,1,57,2,2);vm[:,:,0,0,0]=0
    target['noise_mask']=NestedTensor([vm,torch.ones(1,1,2,320)])
    _,_,plan,_=H3BridgePrepare().prepare(cond(),target,latent(),latent(),feather_frames=39)
    assert plan['left_handover']==0


def test_bridge_removes_only_overlapping_video_guides():
    c=cond();c[0][1]['minimax_keyframes']=[
        {'resolved_frame_index':0,'latent':torch.ones(1,24,1,2,2)},
        {'resolved_frame_index':80,'latent':torch.ones(1,24,1,2,2)},
        {'resolved_frame_index':170,'audio_latent':torch.ones(1,32,2,8)}]
    out,_,_,_=H3BridgePrepare().prepare(c,latent(),latent(),latent())
    assert [kf['resolved_frame_index'] for kf in out[0][1]['minimax_keyframes']]==[80,170]
    assert len(c[0][1]['minimax_keyframes'])==3


def test_bridge_cannot_pin_entire_target_or_change_resolution():
    with pytest.raises(ValueError,match='middle'):
        H3BridgePrepare().prepare(cond(),latent(22),latent(),latent())
    bad=latent();bad['samples'].tensors[0]=torch.ones(1,24,57,4,4)
    with pytest.raises(ValueError,match='resolutions'):
        H3BridgePrepare().prepare(cond(),latent(),bad,latent())
