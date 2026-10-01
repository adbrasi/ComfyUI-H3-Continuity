"""Optional, local context-window sampling of one shared H3 AV canvas."""
import math

import torch

import comfy.conds
import comfy.patcher_extension
import comfy.utils
from comfy.model_base import MiniMaxH3
from comfy.ldm.minimax.model import PackedLayout
from .nodes import streams, pixel_frames
from .timing import frame_edges


def window_grid(total, length, stride):
    if total % 5 != 2 or length % 5 != 2 or stride < 5 or stride % 5:
        raise ValueError('Joint windows need clip-shaped lengths and cycle-aligned starts.')
    if total <= length:
        return [(0, total)]
    last = total - length
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + length) for start in starts]


def triangular_weights(length):
    positions = torch.arange(length, dtype=torch.float32)
    return torch.minimum(positions + 1, length - positions)


class H3JointWindowHandler:
    def __init__(self, shapes, length, stride):
        self.shapes = [tuple(shape) for shape in shapes]
        self.length, self.stride = length, stride

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        if not isinstance(model, MiniMaxH3):
            raise ValueError('H3 Joint Refine requires the native MiniMax H3 model.')
        for items in conds:
            for cond in items or []:
                value = cond.get('model_conds', {}).get('latent_shapes')
                if value is None or [tuple(s) for s in value.cond] != self.shapes:
                    raise ValueError('Connect the same full AV latent to Joint Refine and the sampler.')
        return self.shapes[0][2] > self.length

    def window_conds(self, cond_list, start, stop, audio_start, audio_stop, shapes):
        if cond_list is None:
            return None
        result = []
        edges = frame_edges(self.shapes[0][2])
        first, last = edges[start], edges[stop]
        for cond in cond_list:
            if any(cond.get(key) is not None for key in ('area', 'mask', 'control')):
                raise ValueError('Joint Refine supports one global prompt/reference conditioning; area masks and controls are unsupported.')
            mc = dict(cond['model_conds'])
            mc['latent_shapes'] = comfy.conds.CONDConstant(shapes)
            for key, axis, lo, hi in (('denoise_mask', 2, start, stop), ('audio_denoise_mask', -1, audio_start, audio_stop)):
                if key in mc:
                    value = mc[key]
                    mc[key] = value._copy_with(value.cond.narrow(axis, lo, hi - lo))
            original = mc['minimax_payload'].cond
            payload = dict(original)
            keyframes = []
            for keyframe in original.get('keyframes', []):
                index = keyframe['resolved_frame_index']
                video, audio = keyframe.get('latent'), keyframe.get('audio_latent')
                duration = max(pixel_frames(video.shape[2]) if video is not None else 1,
                               audio.shape[-1] * 24 / 40 if audio is not None else 1)
                if index < last and index + duration > first:
                    keyframes.append(dict(keyframe, resolved_frame_index=index - first))
            payload['keyframes'] = keyframes
            blocks = keyframes + list(payload.get('refs', []))
            payload['cond_video_latents'] = [block['latent'] for block in blocks if block.get('latent') is not None]
            payload['cond_audio_latents'] = [block['audio_latent'] for block in blocks if block.get('audio_latent') is not None]
            video_shape, audio_shape = shapes
            payload['layout'] = PackedLayout(mc['c_crossattn'].cond.shape[1], video_shape[2],
                                            math.ceil(video_shape[3] / 2) * 2, math.ceil(video_shape[4] / 2) * 2,
                                            audio_shape[-1], keyframes=keyframes, refs=payload.get('refs'))
            mc['minimax_payload'] = comfy.conds.CONDConstant(payload)
            result.append(dict(cond, model_conds=mc))
        return result

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        video, audio = comfy.utils.unpack_latents(x_in, self.shapes)
        total = video.shape[2]
        edges = frame_edges(total)
        counts_v = torch.zeros(total)
        counts_a = torch.zeros(audio.shape[-1])
        # Only one window's activations and prediction live on the card.
        sums = [(torch.zeros(video.shape, dtype=torch.float32), torch.zeros(audio.shape, dtype=torch.float32))
                for _ in conds]
        for start, stop in window_grid(total, self.length, self.stride):
            audio_start = round(edges[start] * 5 / 3)
            audio_stop = audio.shape[-1] if stop == total else min(audio.shape[-1], round(edges[stop] * 5 / 3))
            if audio_stop <= audio_start:
                raise ValueError('The AV latent has too few audio rows for its video timeline.')
            window, shapes = comfy.utils.pack_latents([video[:, :, start:stop], audio[..., audio_start:audio_stop]])
            local = [self.window_conds(items, start, stop, audio_start, audio_stop, shapes) for items in conds]
            outputs = calc_cond_batch(model, local, window, timestep, model_options)
            if len(outputs) != len(conds):
                raise ValueError('The sampler returned an unexpected number of conditioning predictions.')
            wv, wa = triangular_weights(stop - start), triangular_weights(audio_stop - audio_start)
            for (sum_v, sum_a), output in zip(sums, outputs):
                if output is None:
                    continue
                ov, oa = comfy.utils.unpack_latents(output, shapes)
                sum_v[:, :, start:stop] += ov.float().cpu() * wv[None, None, :, None, None]
                sum_a[..., audio_start:audio_stop] += oa.float().cpu() * wa[None, None, None, :]
            counts_v[start:stop] += wv
            counts_a[audio_start:audio_stop] += wa
        if not bool((counts_v > 0).all() and (counts_a > 0).all()):
            raise ValueError('Joint windows left an uncovered region of the AV canvas.')
        outputs = []
        for items, (sum_v, sum_a) in zip(conds, sums):
            if items is None:
                outputs.append(torch.zeros_like(x_in))
            else:
                v = sum_v / counts_v[None, None, :, None, None]
                a = sum_a / counts_a[None, None, None, :]
                outputs.append(comfy.utils.pack_latents([v.to(video), a.to(audio)])[0])
        return outputs


def window_memory_budget(executor, model, noise_shape, conds, *args, **kwargs):
    handler = (kwargs.get('model_options') or {}).get('context_handler')
    if isinstance(handler, H3JointWindowHandler):
        shape = list(handler.shapes[0])
        shape[2] = min(shape[2], handler.length)
        noise_shape = shape
    return executor(model, noise_shape, conds, *args, **kwargs)


class H3JointRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'latent': ('LATENT',),
                             'window_frames': ('INT', {'default': 124, 'min': 22, 'max': 3600}),
                             'overlap_frames': ('INT', {'default': 39, 'min': 0, 'max': 3600})}}
    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'report')
    FUNCTION = 'apply'
    CATEGORY = 'MiniMax H3/Continuity/Experimental'
    DESCRIPTION = 'Refine one full AV latent with overlapping context windows at every denoising step. Reuse one global prompt and canonical references. Set low denoise in your sampler. This does not assemble a timeline or guarantee drift elimination. Dynamic Differential Diffusion and conditioning by area are unsupported.'

    def apply(self, model, latent, window_frames=124, overlap_frames=39):
        if not isinstance(model.model, MiniMaxH3):
            raise ValueError('H3 Joint Refine requires the native MiniMax H3 model.')
        if model.model_options.get('context_handler') is not None:
            raise ValueError('Disconnect the other context-window node before using H3 Joint Refine.')
        if model.model_options.get('denoise_mask_function') is not None:
            raise ValueError('Joint Refine cannot use dynamic Differential Diffusion masks.')
        if window_frames < 22 or overlap_frames < 0 or overlap_frames >= window_frames:
            raise ValueError('Use a window of at least 22 frames and a smaller nonnegative overlap.')
        video, audio = streams(latent)
        length = 7 + 5 * ((int(window_frames) - 22) // 17)
        # Native windows always end at cycle phase 2. The smallest aligned
        # overlap is therefore two latent steps (five pixel frames).
        overlap = 2 + 5 * max(0, math.ceil((int(overlap_frames) - 5) / 17))
        stride = length - overlap
        if stride < 5:
            raise ValueError('The rounded overlap leaves no complete H3 cycle between windows.')
        result = model.clone()
        handler = H3JointWindowHandler([video.shape, audio.shape], length, stride)
        result.model_options['context_handler'] = handler
        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                    'h3_continuity_joint_memory', window_memory_budget)
        windows = window_grid(video.shape[2], length, stride)
        report = (f'{len(windows)} shared-state windows; {pixel_frames(length)} frames/window, '
                  f'at least {pixel_frames(overlap)} overlapping frames; starts aligned to the 17-frame H3 cycle. '
                  'Predictions blend before each sampler update. Use the same full latent and one prompt/reference set.')
        return result, report


NODE_CLASS_MAPPINGS = {'H3JointRefine': H3JointRefine}
NODE_DISPLAY_NAME_MAPPINGS = {'H3JointRefine': 'H3 Joint Refine · Context Windows (experimental)'}
