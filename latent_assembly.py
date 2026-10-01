"""Join saved H3 continuation latents without decoding and re-encoding."""

import torch
from comfy.nested_tensor import NestedTensor

from .nodes import CATEGORY, FPS, streams
from .timing import frame_edges


def _step_at(edges, frame, label):
    try:
        return edges.index(frame)
    except ValueError as exc:
        raise ValueError(f'{label} must fall on an H3 token boundary.') from exc


class H3ContinuityLatentAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source_latent': ('LATENT',), 'sampled_latent': ('LATENT',),
                             'plan': ('H3_CONTINUITY_PLAN',)}}

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('latent', 'report')
    FUNCTION = 'assemble'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Join an approved source and continuation in latent space. Preserves the source except for any regenerated handover frames, then appends the sampled continuation.'

    def assemble(self, source_latent, sampled_latent, plan):
        sv, sa = streams(source_latent)
        tv, ta = streams(sampled_latent)
        if sv.shape[3:] != tv.shape[3:]:
            raise ValueError('Source and sampled latent canvases must match.')
        se, te = frame_edges(sv.shape[2]), frame_edges(tv.shape[2])
        context, total = int(plan['context_frames']), int(plan['target_frames'])
        handover = int(plan.get('handover_frame', context))
        if te[-1] != total or len(se) != sv.shape[2] + 1:
            raise ValueError('Plan timing must match the full sampled H3 latent.')
        if context not in se or context not in te or not 0 <= handover <= context:
            raise ValueError('Context and handover must align with H3 token boundaries.')
        if handover not in te:
            raise ValueError('Handover must align with an H3 token boundary.')
        source_cut_frame = se[-1] - context + handover
        source_cut = _step_at(se, source_cut_frame, 'Source splice')
        target_cut = _step_at(te, handover, 'Handover')
        expected = round((se[-1] + total - context) * 40 / FPS)
        audio_append = expected - sa.shape[-1]
        audio_start = ta.shape[-1] - audio_append
        nominal_start = round(context * 40 / FPS)
        if sa.shape[-1] != round(se[-1] * 40 / FPS) or ta.shape[-1] != round(te[-1] * 40 / FPS):
            raise ValueError('Audio latent row counts must align with their video timelines.')
        if not 0 <= audio_append <= ta.shape[-1] or abs(audio_start - nominal_start) > 1:
            raise ValueError('Audio latent rows cannot be aligned to the merged video duration.')
        video = torch.cat((sv[:, :, :source_cut], tv[:, :, target_cut:].to(sv)), dim=2)
        audio = torch.cat((sa, ta[..., audio_start:].to(sa)), dim=-1)
        result = dict(sampled_latent, samples=NestedTensor([video, audio]))
        result.pop('noise_mask', None)
        if 'end_seconds' in plan:
            result['h3_end_seconds'] = float(plan['end_seconds'])
        if audio.shape[-1] != expected:
            raise ValueError('Merged audio length does not match the merged video timeline.')
        report = (f'{video.shape[2]} latent steps ({se[-1] + total - context} frames); '
                  f'kept {source_cut} source steps and appended {tv.shape[2] - target_cut} sampled steps. '
                  f'Preserved the complete source audio and skipped {audio_start} overlap audio rows.')
        if audio_start != nominal_start:
            report += f' Adjusted the overlap by {audio_start - nominal_start:+d} audio row to match the exact merged duration.'
        return result, report


NODE_CLASS_MAPPINGS = {'H3ContinuityLatentAssemble': H3ContinuityLatentAssemble}
NODE_DISPLAY_NAME_MAPPINGS = {'H3ContinuityLatentAssemble': 'H3 Continuity · Assemble Latents'}
