"""Split H3 AV tensors without VAE conversion or precision changes."""

from bisect import bisect_right

from comfy.nested_tensor import NestedTensor

from .timing import frame_edges


class H3ContinuityLatentSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'latent': ('LATENT',),
            'context_frames': ('INT', {'default': 22, 'min': 1, 'max': 8640000,
                                       'tooltip': 'Initial context in frames at 24 fps. Cuts at the nearest H3 token boundary at or before this frame. No overlap or padding is added.'}),
        }}

    RETURN_TYPES = ('LATENT', 'LATENT', 'INT', 'INT', 'STRING')
    RETURN_NAMES = ('context_latent', 'generated_latent', 'context_frames',
                    'generated_frames', 'report')
    FUNCTION = 'split'
    CATEGORY = 'MiniMax H3/Continuity'
    DESCRIPTION = ('Split the raw H3 audio/video latent into two disjoint tensor slices. '
                   'Values, dtype and device are preserved. Frame counts describe the original timeline; '
                   'standalone VAE decoding can change boundary pixels and duration. '
                   'Slices may be outside the full-window grid required by Continuity Prepare/Save.')

    def split(self, latent, context_frames):
        samples = latent['samples']
        if not isinstance(samples, NestedTensor) or len(samples.tensors) != 2:
            raise ValueError('Connect a MiniMax H3 audio/video latent.')
        video, audio = samples.unbind()
        if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
            raise ValueError('Expected H3 video [1,24,T,H,W] and audio [1,32,2,T].')
        edges = frame_edges(video.shape[2])
        total = edges[-1]
        if not 0 < context_frames < total:
            raise ValueError(f'context_frames must be between 1 and {total - 1}; both outputs need frames.')
        cut = bisect_right(edges, context_frames) - 1
        actual = edges[cut]
        audio_cut = round(actual * 40 / 24)
        if not 0 < audio_cut < audio.shape[-1]:
            raise ValueError('The audio latent is too short for two nonempty parts at this cut.')

        context = dict(latent, samples=NestedTensor([
            video[:, :, :cut].clone(), audio[..., :audio_cut].clone()]))
        generated = dict(latent, samples=NestedTensor([
            video[:, :, cut:].clone(), audio[..., audio_cut:].clone()]))
        # These are sampled segments; a new generation must define its own mask.
        context.pop('noise_mask', None)
        generated.pop('noise_mask', None)
        if 'h3_end_seconds' in latent:
            context['h3_end_seconds'] = float(latent['h3_end_seconds']) - (total - actual) / 24
        report = (f'Requested {context_frames}; cut at {actual} frames on the original timeline. '
                  f'Context: {actual}; generated: {total - actual}. '
                  f'Audio cut at row {audio_cut} ({audio_cut / 40:.3f}s); '
                  f'rounding offset {(audio_cut / 40 - actual / 24) * 1000:.3f} ms. '
                  'Disjoint copies: no lost or duplicated tokens, no dtype conversion. '
                  'Standalone decoding may change boundary pixels and duration; these slices '
                  'are not necessarily valid full H3 windows for Continuity Prepare/Save. '
                  'Keep the original full latent for subsequent continuation.')
        return context, generated, actual, total - actual, report


NODE_CLASS_MAPPINGS = {'H3ContinuityLatentSplit': H3ContinuityLatentSplit}
NODE_DISPLAY_NAME_MAPPINGS = {'H3ContinuityLatentSplit': 'H3 Continuity · Split Context / Generated Latent'}
