"""Two-sided H3 latent inpainting, with a delivery plan for both seams."""
import torch

from comfy.nested_tensor import NestedTensor
from .nodes import streams, pixel_frames, tail_length, FPS, fit_audio, place_audio
from .timing import frame_edges, handover


def release_mask(steps, frames, feather, strength, side, tensor):
    edges = frame_edges(steps)
    span = min(float(feather), frames)
    values = []
    for a, b in zip(edges, edges[1:]):
        distance = frames - (a+b)/2 if side == 'left' else (a+b)/2
        x = max(0., min(1., 1-distance/span)) if span > 0 else 0.
        values.append(x*x*(3-2*x)*strength)
    return tensor.new_tensor(values).view(1, 1, steps, 1, 1)


class H3BridgePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'positive': ('CONDITIONING',), 'latent': ('LATENT',),
            'left_latent': ('LATENT',), 'right_latent': ('LATENT',),
            'context_frames': ('INT', {'default': 39, 'min': 5, 'max': 3600, 'step': 17}),
            'feather_frames': ('INT', {'default': 0, 'min': 0, 'max': 3600}),
            'feather_strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': .05}),
            'audio_policy': (['preserve', 'generate'],),
        }}
    RETURN_TYPES = ('CONDITIONING', 'LATENT', 'H3_BRIDGE_PLAN', 'STRING')
    RETURN_NAMES = ('positive', 'latent', 'plan', 'report')
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3/Continuity/Bridge'
    DESCRIPTION = 'Freeze the tail of A and the head of B in a target latent. Generate the middle with the normal sampler. Assemble delivers released edge frames too.'

    def prepare(self, positive, latent, left_latent, right_latent, context_frames=39,
                feather_frames=0, feather_strength=1., audio_policy='preserve'):
        v, a = streams(latent)
        lv, la = streams(left_latent)
        rv, ra = streams(right_latent)
        if lv.shape[3:] != v.shape[3:] or rv.shape[3:] != v.shape[3:]:
            raise ValueError('Bridge source and target latent resolutions must match.')
        total = pixel_frames(v.shape[2])
        left_frames, right_frames = pixel_frames(lv.shape[2]), pixel_frames(rv.shape[2])
        n = tail_length(context_frames, min(left_frames, right_frames))
        steps = 2 + 5*((n-5)//17)
        if n < 5 or 2*steps >= v.shape[2]:
            raise ValueError('The bridge target must leave a generated middle after both context windows.')
        pv, pa = v.clone(), a.clone()
        pv[:, :, :steps] = lv[:, :, -steps:].to(pv)
        pv[:, :, -steps:] = rv[:, :, :steps].to(pv)
        vm, am = torch.ones_like(pv[:, :1]), torch.ones_like(pa[:, :1])
        vm[:, :, :steps] = release_mask(steps, n, feather_frames, feather_strength, 'left', vm)
        vm[:, :, -steps:] = release_mask(steps, n, feather_frames, feather_strength, 'right', vm)
        audio_rounding = 0.
        if audio_policy == 'preserve':
            # Preserve complete rows, leaving each boundary's fractional row
            # generated. Source waveforms are restored by the assembler.
            left_end = round(n*40/FPS + la.shape[-1]-left_frames*40/FPS)
            left_stop = min(int(n*40/FPS), left_end)
            left_start = max(0, left_end-la.shape[-1])
            pa[..., left_start:left_stop] = la[..., la.shape[-1]-left_end+left_start:la.shape[-1]-left_end+left_stop].to(pa)
            am[..., left_start:left_stop] = 0
            right_start = round((total-n)*40/FPS)
            count = min(ra.shape[-1], pa.shape[-1]-right_start, int(n*40/FPS))
            pa[..., right_start:right_start+count] = ra[..., :count].to(pa)
            am[..., right_start:right_start+count] = 0
            audio_rounding = max(abs(left_end-(n*40/FPS+la.shape[-1]-left_frames*40/FPS)),
                                 abs(right_start-(total-n)*40/FPS))/40
        elif audio_policy != 'generate':
            raise ValueError('Unknown bridge audio policy.')
        if 'noise_mask' in latent:
            old_v, old_a = latent['noise_mask'].unbind()
            vm *= old_v.to(vm)
            am *= old_a.to(am)
        hleft = handover(vm, steps, 'left')
        hright = handover(vm[:, :, -steps:], steps, 'right')
        plan = {'version': 1, 'target_frames': total, 'context_frames': n,
                'middle_frames': total-2*n, 'left_handover': hleft,
                'right_handover': hright, 'right_frames': right_frames,
                'audio_policy': audio_policy}
        result = []
        for embedding, metadata in positive:
            info = dict(metadata)
            kept = []
            for keyframe in info.get('minimax_keyframes', []):
                kf = dict(keyframe)
                if 'latent' in kf:
                    start = kf['resolved_frame_index']
                    end = start + pixel_frames(kf['latent'].shape[2])
                    if start < n or end > total-n:
                        kf.pop('latent')
                if 'latent' in kf or 'audio_latent' in kf:
                    kept.append(kf)
            info['minimax_keyframes'] = kept
            result.append([embedding, info])
        prepared = dict(latent, samples=NestedTensor([pv, pa]), noise_mask=NestedTensor([vm, am]))
        report = f'{n} context frames per side; {total-2*n} new middle frames; deliver {n-hleft} left and {hright} right repair frames. Audio={audio_policy}.'
        if audio_rounding:
            report += f' Context audio placement rounded by up to {audio_rounding*1000:.3f} ms; use aligned AV lengths for exact row placement.'
        return result, prepared, plan, report


class H3BridgeAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'left_images': ('IMAGE',),
                             'right_images': ('IMAGE',), 'plan': ('H3_BRIDGE_PLAN',)},
                'optional': {'audio': ('AUDIO',), 'left_audio': ('AUDIO',), 'right_audio': ('AUDIO',)}}
    RETURN_TYPES = ('IMAGE', 'AUDIO', 'STRING')
    FUNCTION = 'assemble'
    CATEGORY = 'MiniMax H3/Continuity/Bridge'
    DESCRIPTION = 'Join A, the generated bridge and B without duplicate context. Original waveform samples are restored outside the middle. Both source canvases must match the bridge.'

    def assemble(self, images, left_images, right_images, plan, audio=None, left_audio=None, right_audio=None):
        n, total = plan['context_frames'], plan['target_frames']
        if len(images) != total or len(left_images) < n or len(right_images) != plan['right_frames']:
            raise ValueError('Use the full bridge decode, the complete left context and the exact right clip decode.')
        if left_images.shape[1:] != images.shape[1:] or right_images.shape[1:] != images.shape[1:]:
            raise ValueError('Bridge and source image canvases must match.')
        left_cut = len(left_images)-n+plan['left_handover']
        right_cut = plan['right_handover']
        out = torch.cat([left_images[:left_cut], images[plan['left_handover']:total-n+right_cut].to(left_images),
                         right_images[right_cut:].to(left_images)], 0)
        parts = [p for p in (left_audio, audio, right_audio) if p is not None]
        sr = int(parts[0]['sample_rate']) if parts else 32000
        channels = max(p['waveform'].shape[1] for p in parts) if parts else 2
        wave = parts[0]['waveform'] if parts else out
        sound = {'waveform': wave.new_zeros(1, channels, round(len(out)/FPS*sr)), 'sample_rate': sr}
        if audio is not None:
            place_audio(sound, audio, (len(left_images)-n)/FPS)
        if left_audio is not None:
            place_audio(sound, fit_audio(left_audio, round(len(left_images)/FPS*left_audio['sample_rate'])), 0)
        right_at = (len(left_images)+plan['middle_frames'])/FPS
        if right_audio is not None:
            place_audio(sound, fit_audio(right_audio, round(len(right_images)/FPS*right_audio['sample_rate'])), right_at)
        return out, sound, f'{len(out)} frames at 24 fps; {plan["middle_frames"]} inserted frames; replaced {n-plan["left_handover"]}/{right_cut} edge frames. Evaluate both seams in motion.'


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3BridgePrepare, H3BridgeAssemble)}
NODE_DISPLAY_NAME_MAPPINGS = {'H3BridgePrepare': 'H3 Bridge · Prepare', 'H3BridgeAssemble': 'H3 Bridge · Assemble'}
