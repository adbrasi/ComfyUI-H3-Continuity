"""Temporal retakes using native H3 masks; isolated from continuation nodes."""
import math

import torch

import comfy.utils
from comfy.model_base import MiniMaxH3
from comfy.nested_tensor import NestedTensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from .nodes import streams, pixel_frames, fit_audio, encode_audio, FPS


def temporal_mask(count, start, end, offset, grow, feather, curve):
    lo = max(0, min(count, start + offset - grow))
    hi = max(0, min(count, end + offset + grow))
    if hi <= lo:
        raise ValueError('The retake interval is empty after offset/grow. End is exclusive.')
    t = torch.arange(count, dtype=torch.float32) + .5
    weights = ((t >= lo) & (t < hi)).float()
    if feather > 0:
        ramp = torch.minimum((t-lo)/feather, (hi-t)/feather).clamp(0, 1)
        if curve == 'smoothstep':
            ramp = ramp.square() * (3-2*ramp)
        weights *= ramp
    return weights, lo, hi


class H3RetakePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'start_frame': ('INT', {'default': 34, 'min': 0, 'max': 86400}),
            'end_frame': ('INT', {'default': 90, 'min': 1, 'max': 86400, 'tooltip': 'Exclusive end, at 24 fps. Start/end refer to the supplied clip, not the entire movie.'}),
            'offset_frames': ('INT', {'default': 0, 'min': -86400, 'max': 86400}),
            'grow_frames': ('INT', {'default': 0, 'min': -3600, 'max': 3600, 'tooltip': 'Expand both ends; negative values contract the interval.'}),
            'feather_frames': ('INT', {'default': 8, 'min': 0, 'max': 3600, 'tooltip': 'Fade inside both boundaries. 0 is a hard mask. Long feathers reduce the fully editable center.'}),
            'curve': (['smoothstep', 'linear'],),
            'edit_audio': ('BOOLEAN', {'default': False}),
        }, 'optional': {
            'source_latent': ('LATENT',), 'source_images': ('IMAGE',),
            'vae': ('VAE',), 'source_audio': ('AUDIO',), 'audio_vae': ('VAE',),
        }}
    RETURN_TYPES = ('LATENT', 'H3_RETAKE_PLAN', 'IMAGE', 'STRING')
    RETURN_NAMES = ('latent', 'plan', 'mask_timeline', 'report')
    FUNCTION = 'prepare'
    CATEGORY = 'MiniMax H3/Continuity/Retake'
    DESCRIPTION = 'Preserve both ends and regenerate a temporal interval. Connect this latent to the sampler, then use Retake Assemble. External images must be at 24 fps; use Import Video.'

    def prepare(self, start_frame, end_frame, offset_frames=0, grow_frames=0, feather_frames=8,
                curve='smoothstep', edit_audio=False, source_latent=None, source_images=None,
                vae=None, source_audio=None, audio_vae=None):
        if source_latent is not None:
            v, a = streams(source_latent)
            count = pixel_frames(v.shape[2])
            result = dict(source_latent)
            if source_images is not None and len(source_images) != count:
                raise ValueError('For raw-latent retakes, source_images must decode that exact untrimmed latent, with the same frame count.')
            origin = 'Raw latent; no VAE re-encode.'
        else:
            if source_images is None or len(source_images) == 0 or vae is None:
                raise ValueError('Connect source_latent, or source_images plus the H3 video VAE.')
            count = len(source_images)
            total = max(5, 5 + 17*math.ceil((count-5)/17))
            images = source_images[..., :3]
            if total > count:
                images = torch.cat([images, images[-1:].expand(total-count, -1, -1, -1)], 0)
            v = vae.encode(images)
            audio_steps = round(total*5/3)
            a = v.new_zeros((1, 32, 2, audio_steps))
            if source_audio is not None and audio_vae is not None:
                sr = int(source_audio['sample_rate'])
                audio = fit_audio(source_audio, math.ceil((audio_steps+1)/40*sr))
                encoded = encode_audio(audio_vae, audio)
                if encoded.shape[-1] < audio_steps:
                    raise ValueError('Audio VAE returned too few rows for the retake.')
                a = encoded[..., :audio_steps]
            result = {'samples': NestedTensor([v, a])}
            streams(result)
            origin = f'External video encoded once; padded internally to {total} frames, delivery stays {count}.'
        total = pixel_frames(v.shape[2])
        profile, lo, hi = temporal_mask(count, start_frame, end_frame, offset_frames, grow_frames, feather_frames, curve)
        padded = torch.nn.functional.pad(profile, (0, total-count))
        weights, intervals, pos = [], [], 0
        for k in range(v.shape[2]):
            stop = pos + FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)]
            weights.append(padded[pos:stop].max())
            intervals.append((pos, stop))
            pos = stop
        w = torch.stack(weights)
        vm = w.to(v).view(1, 1, -1, 1, 1).expand(1, 1, v.shape[2], v.shape[3], v.shape[4]).clone()
        # Audio rows are 40 Hz. Map their support to the requested pixel-time profile.
        am = torch.zeros_like(a[:, :1])
        if edit_audio:
            indices = (torch.arange(a.shape[-1])*FPS/40).long().clamp(max=count-1)
            aw = profile[indices]
            aw[torch.arange(a.shape[-1])*FPS/40 >= count] = 0
            am = aw.to(a).view(1,1,1,-1).expand_as(am).clone()
        # A retake defines a new edit region; any prior generation mask is replaced.
        result['noise_mask'] = NestedTensor([vm, am])
        active = (w > 0).nonzero().flatten().tolist()
        actual_lo = intervals[active[0]][0]
        actual_hi = min(count, intervals[active[-1]][1])
        plan = {'version': 1, 'frames': count, 'target_frames': total, 'start': actual_lo, 'end': actual_hi,
                'requested_start': lo, 'requested_end': hi, 'edit_audio': bool(edit_audio),
                'audio_start': lo, 'audio_end': hi, 'feather_frames': feather_frames}
        timeline = torch.repeat_interleave(w, torch.tensor([b-a for a,b in intervals]))[:count]
        preview = timeline.view(1,1,count,1).expand(1,64,count,3).clone()
        report = f'{origin} Edit [{lo}, {hi}); latent support [{actual_lo}, {actual_hi}) at 24 fps. Feather {feather_frames} frames inside each edge. Zero-mask latents stay fixed; VAE pixels can have wider temporal influence. Assemble restores original frames outside the reported support.'
        return result, plan, preview, report


class H3RetakeAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'source_images': ('IMAGE',), 'plan': ('H3_RETAKE_PLAN',)},
                'optional': {'audio': ('AUDIO',), 'source_audio': ('AUDIO',)}}
    RETURN_TYPES = ('IMAGE', 'AUDIO', 'STRING')
    FUNCTION = 'assemble'
    CATEGORY = 'MiniMax H3/Continuity/Retake'
    DESCRIPTION = 'Keep original pixels outside the effective retake interval. No video crossfade. Original audio is preserved unless edit_audio was selected.'

    def assemble(self, images, source_images, plan, audio=None, source_audio=None):
        count, lo, hi = plan['frames'], plan['start'], plan['end']
        if len(source_images) != count or len(images) < count or images.shape[1:] != source_images.shape[1:]:
            raise ValueError('Use the exact source frames and the full sampler decode at matching resolution.')
        result = source_images.clone()
        result[lo:hi] = images[lo:hi].to(result)
        sr = int(source_audio['sample_rate'] if source_audio is not None else audio['sample_rate'] if audio is not None else 32000)
        base = source_audio if source_audio is not None else {'waveform': torch.zeros(1,2,round(count/FPS*sr)), 'sample_rate':sr}
        out_audio = fit_audio(base, round(count/FPS*sr))
        if plan['edit_audio']:
            if audio is None:
                raise ValueError('Connect the generated audio to assemble an audio retake.')
            generated = fit_audio(audio, round(count/FPS*sr), sr)
            if generated['waveform'].shape[1] != out_audio['waveform'].shape[1]:
                if out_audio['waveform'].shape[1] == 1:
                    generated['waveform'] = generated['waveform'].mean(1, keepdim=True)
                else:
                    generated['waveform'] = generated['waveform'].expand(-1,out_audio['waveform'].shape[1],-1)
            x = torch.arange(round(count/FPS*sr), device=out_audio['waveform'].device)*FPS/sr
            gain = ((x >= plan['audio_start']) & (x < plan['audio_end'])).float()
            f = plan['feather_frames']
            if f:
                ramp = torch.minimum((x-plan['audio_start'])/f, (plan['audio_end']-x)/f).clamp(0,1)
                gain *= ramp.square()*(3-2*ramp)
            out_audio = {'waveform': out_audio['waveform']*(1-gain) + generated['waveform'].to(out_audio['waveform'])*gain, 'sample_rate':sr}
        return result, out_audio, f'{count} frames; original pixels outside [{lo}, {hi}) retained. Audio '+('edited with edge fades.' if plan['edit_audio'] else 'preserved from source_audio; silence if absent.')


def differential_mask(mask, sigma, sampling, sigmas, strength):
    if strength == 0:
        return mask
    to_sigma = max(float(sampling.sigma_min), float(sigmas[-1]))
    start = sampling.timestep(sigmas[0].to(sigma))
    end = sampling.timestep(sigma.new_tensor(to_sigma))
    threshold = ((sampling.timestep(sigma.flatten()[0])-end)/(start-end).clamp(min=1e-8)).clamp(0,1)
    binary = (mask >= threshold).to(mask)
    mixed = mask.lerp(binary, strength)
    # Never unlock protected regions, including at threshold zero.
    return torch.where(mask <= 0, torch.zeros_like(mixed), torch.where(mask >= 1, torch.ones_like(mixed), mixed))


class H3RetakeDifferentialDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'latent': ('LATENT',), 'sigmas': ('SIGMAS',),
                             'strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': .05})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'MiniMax H3/Continuity/Retake'
    DESCRIPTION = 'Experimental Differential Diffusion synchronized with H3 row masks. Connect the exact retake latent and the same SIGMAS used by the sampler. Does not alter global model code.'

    def apply(self, model, latent, sigmas, strength=1.):
        if not isinstance(model.model, MiniMaxH3):
            raise ValueError('This Differential Diffusion adapter is only for MiniMax H3.')
        if 'denoise_mask_function' in model.model_options:
            raise ValueError('Disconnect the other dynamic mask node before using H3 Retake Differential Diffusion.')
        v, a = streams(latent)
        vm, am = latent['noise_mask'].unbind()
        packed = comfy.utils.pack_latents([vm.expand_as(v), am.expand_as(a)])[0]
        shapes = [v.shape, a.shape]
        sampling = model.model.model_sampling
        result = model.clone()
        previous = result.model_options.get('model_function_wrapper')
        def mask_fn(sigma, mask, extra_options):
            return differential_mask(mask, sigma, sampling, sigmas, strength)
        def wrapper(apply_model, args):
            current = differential_mask(packed.to(args['input']), args['timestep'], sampling, sigmas, strength)
            c = dict(args['c'])
            c.pop('denoise_mask', None)
            c.pop('audio_denoise_mask', None)
            c.update(model.model._denoise_mask_values(current, shapes))
            updated = dict(args, c=c)
            if previous is not None:
                return previous(apply_model, updated)
            return apply_model(updated['input'], updated['timestep'], **updated['c'])
        result.set_model_denoise_mask_function(mask_fn)
        result.set_model_unet_function_wrapper(wrapper)
        return (result,)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3RetakePrepare, H3RetakeAssemble, H3RetakeDifferentialDiffusion)}
NODE_DISPLAY_NAME_MAPPINGS = {'H3RetakePrepare': 'H3 Retake · Prepare Mask', 'H3RetakeAssemble': 'H3 Retake · Assemble', 'H3RetakeDifferentialDiffusion': 'H3 Retake · Differential Diffusion (experimental)'}
