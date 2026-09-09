"""Small, local MiniMax H3 continuation nodes. No model or sampler patches."""
from pathlib import Path
import tempfile
import math

import torch
import torch.nn.functional as F
import torchaudio
import safetensors.torch

import folder_paths
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, PackedLayout

FPS = 24
CATEGORY = 'MiniMax H3/Continuity'


def streams(latent):
    samples = latent['samples']
    if not isinstance(samples, NestedTensor) or len(samples.tensors) != 2:
        raise ValueError('Connect a MiniMax H3 audio/video latent from the sampler.')
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
        raise ValueError('Expected H3 video [1,24,T,H,W] and audio [1,32,2,T].')
    if video.shape[2] % 5 != 2:
        raise ValueError('H3 continuation needs an untrimmed latent on the 17k+5 frame grid.')
    return video, audio


def pixel_frames(steps):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(steps))


def tail_length(requested, available):
    n = min(requested, available)
    return 1 if n < 5 else 5 + 17 * ((n - 5) // 17)


def fit_audio(audio, count, sample_rate=None):
    sr = int(sample_rate or audio['sample_rate'])
    wave = audio['waveform']
    if int(audio['sample_rate']) != sr:
        wave = torchaudio.functional.resample(wave, int(audio['sample_rate']), sr)
    return {'waveform': F.pad(wave[..., :count], (0, max(0, count - wave.shape[-1]))), 'sample_rate': sr}


def encode_audio(vae, audio):
    sr = int(vae.audio_sample_rate)
    wave = audio['waveform'][:1]
    if audio['sample_rate'] != sr:
        wave = torchaudio.functional.resample(wave, int(audio['sample_rate']), sr)
    if wave.shape[1] == 1:
        wave = wave.repeat(1, 2, 1)
    return vae.encode(wave.movedim(1, -1))


def check_layout():
    # Fractional audio anchors must stay on the target timeline, including with references.
    dummy = torch.empty(1, 32, 2, 40)
    for refs in (None, [{'kind': 'audio', 'ref_audio_t': 9}]):
        layout = PackedLayout(7, 7, 2, 2, 37,
                              keyframes=[{'resolved_frame_index': -1.8, 'audio_latent': dummy}], refs=refs)
        target = next(a for a, b, kind in layout.segments if kind == 'audio')
        anchor = next(a for a, b, kind in layout.segments if kind == 'cond_audio')
        delta = float(layout.position_ids[anchor, 0] - layout.position_ids[target, 0])
        if abs(delta + 3) > 1e-6:
            raise RuntimeError('This ComfyUI H3 layout changed audio placement. Update H3 Continuity before rendering.')



class H3ContinuityImport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video': ('VIDEO',),
                             'width': ('INT', {'default': 736, 'min': 32, 'max': 8192, 'step': 32}),
                             'height': ('INT', {'default': 416, 'min': 32, 'max': 8192, 'step': 32})}}
    RETURN_TYPES = ('IMAGE', 'AUDIO', 'STRING')
    RETURN_NAMES = ('images', 'audio', 'report')
    FUNCTION = 'convert'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Connect Load Video. Converts the full source to 24 fps and the target canvas before Prepare and Assemble, keeping its duration and synchronized audio.'

    def convert(self, video, width, height):
        components = video.get_components()
        images = components.images
        fps = float(components.frame_rate)
        if len(images) == 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('The source video has no frames or no valid frame rate.')
        count = max(1, round(len(images) * FPS / fps))
        if fps != FPS:
            indices = [min(len(images)-1, int(i*fps/FPS)) for i in range(count)]
            images = images[indices]
        if images.shape[1:3] != (height, width):
            images = comfy.utils.common_upscale(images[..., :3].movedim(-1, 1), width, height, 'lanczos', 'center').movedim(1, -1)
        audio = components.audio
        if audio is None:
            audio = {'waveform': torch.zeros(1, 2, round(count/FPS*32000)), 'sample_rate': 32000}
        else:
            audio = fit_audio(audio, round(count/FPS*audio['sample_rate']))
        return images, audio, f'{fps:g} → 24 fps; {count} frames; {width}×{height}; {count/FPS:.3f}s. Rate conversion uses frame selection, not synthesized motion.'


class H3ContinuityPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'positive': ('CONDITIONING',), 'latent': ('LATENT',),
            'context_frames': ([22, 39, 56, 5],),
            'method': (['pinned_av', 'pinned_prefix', 'anchors'], {'tooltip': 'pinned_prefix locks video and guides audio; pinned_av locks aligned audio rows too; anchors regenerates the overlap.'}),
            'audio_context_seconds': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 10, 'step': 0.1}),
        }, 'optional': {
            'source_latent': ('LATENT', {'tooltip': 'Preferred: untrimmed sampler output. Avoids VAE re-encoding. Takes priority over source_images.'}),
            'source_images': ('IMAGE', {'tooltip': 'External clip, decoded at 24 fps. The last frames are used, never the first.'}),
            'vae': ('VAE',), 'source_audio': ('AUDIO',), 'audio_vae': ('VAE',),
            'soundtrack': ('AUDIO', {'tooltip': 'Original full timeline recording, starting at time zero. Conditions the new video; Assemble preserves these samples.'}),
            'source_end_seconds': ('FLOAT', {'default': 0, 'min': 0, 'max': 86400, 'step': 0.01, 'tooltip': '0 = infer from source or saved chain. Override for a tail taken from a longer movie.'}),
        }}

    RETURN_TYPES = ('CONDITIONING', 'LATENT', 'H3_CONTINUITY_PLAN', 'STRING')
    RETURN_NAMES = ('positive', 'latent', 'plan', 'report')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Continue the actual tail, preserve references and end anchors, and align the audio window with the join. Connect plan to Assemble.'

    def prepare(self, positive, latent, context_frames=22, method='anchors', audio_context_seconds=1,
                source_latent=None, source_images=None, vae=None, source_audio=None,
                audio_vae=None, soundtrack=None, source_end_seconds=0):
        check_layout()
        video, audio = streams(latent)
        total = pixel_frames(video.shape[2])
        if source_latent is None and source_images is None:
            raise ValueError('Connect source_latent or source_images from the video to continue.')
        warnings = []
        if source_latent is not None:
            sv, sa = streams(source_latent)
            available = pixel_frames(sv.shape[2])
            if sv.shape[3:] != video.shape[3:]:
                raise ValueError('Source and target latent resolutions differ. Use source_images + VAE to resize an external clip.')
            n = tail_length(int(context_frames), available)
            steps = 2 + 5 * ((n - 5) // 17)
            tail = sv[:, :, -steps:].clone()
            origin = 'latent'
            inferred_end = source_latent.get('h3_end_seconds', available / FPS)
        else:
            available = len(source_images)
            if available == 0 or vae is None:
                raise ValueError('source_images needs nonempty frames and the H3 video VAE.')
            n = tail_length(int(context_frames), available)
            frames = source_images[-n:, ..., :3]
            h, w = video.shape[3] * 16, video.shape[4] * 16
            if frames.shape[1:3] != (h, w):
                frames = comfy.utils.common_upscale(frames.movedim(-1, 1), w, h, 'lanczos', 'center').movedim(1, -1)
            tail = vae.encode(frames)
            origin = 'pixels'
            inferred_end = available / FPS
            warnings.append('External frames require one lossy VAE encode; use saved latents for subsequent links.')
        if n >= total:
            raise ValueError(f'The target has {total} frames but context uses {n}. Increase target length.')
        end = float(source_end_seconds or inferred_end)
        if end < n / FPS:
            raise ValueError('source_end_seconds is earlier than the context window.')
        new_frames = total - n
        plan = {'version': 1, 'context_frames': n, 'target_frames': total,
                'source_end_seconds': end, 'end_seconds': end + new_frames / FPS}
        kfs = []
        prepared = dict(latent)
        if method in ('pinned_prefix', 'pinned_av'):
            pv, pa = video.clone(), audio.clone()
            pv[:, :, :tail.shape[2]] = tail.to(pv)
            vm = torch.ones_like(pv[:, :1])
            vm[:, :, :tail.shape[2]] = 0
            am = torch.ones_like(pa[:, :1])
            if 'noise_mask' in latent:
                old_vm, old_am = latent['noise_mask'].unbind()
                vm = vm * old_vm.to(vm)
                am = am * old_am.to(am)
            prepared['samples'] = NestedTensor([pv, pa])
            prepared['noise_mask'] = NestedTensor([vm, am])
        elif method == 'anchors':
            kfs.append({'resolved_frame_index': 0, 'latent': tail})
        else:
            raise ValueError('Unknown continuation method.')

        if soundtrack is not None:
            if audio_vae is None:
                raise ValueError('soundtrack needs the H3 audio VAE.')
            sr = int(soundtrack['sample_rate'])
            start = round((end - n / FPS) * sr)
            # VAE encoding floors its 40 Hz grid. One lookahead step covers the
            # rounded target and encoder boundary; only target rows are retained.
            stop = start + math.ceil((audio.shape[-1] + 1) / 40 * sr)
            if soundtrack['waveform'].shape[-1] < round(plan['end_seconds'] * sr):
                raise ValueError('The original soundtrack ends before the new clip. Supply a longer recording or disconnect soundtrack.')
            wave = soundtrack['waveform'][..., start:stop].clone()
            if wave.shape[-1] < stop - start:
                wave = F.pad(wave, (0, stop - start - wave.shape[-1]), mode='replicate')
            segment = {'waveform': wave, 'sample_rate': sr}
            z = encode_audio(audio_vae, segment)[..., :audio.shape[-1]].clone()
            if z.shape[-1] < audio.shape[-1]:
                raise ValueError('Audio VAE returned too few steps despite encoder lookahead. Check that the connected VAE is the H3 audio VAE.')
            current_video, _ = streams(prepared)
            video_mask = prepared['noise_mask'].tensors[0].clone() if 'noise_mask' in prepared else torch.ones_like(video[:, :1])
            prepared['samples'] = NestedTensor([current_video, z.to(audio)])
            prepared['noise_mask'] = NestedTensor([video_mask, torch.zeros_like(audio[:, :1])])
            plan['soundtrack_segment'] = {'waveform': soundtrack['waveform'][..., round(end * sr):round(plan['end_seconds'] * sr)].clone(), 'sample_rate': sr}
            warnings.append('Original soundtrack is preserved by Assemble. Conditioning cannot guarantee perfect lip sync.')
        elif audio_context_seconds > 0 and (source_latent is not None or source_audio is not None):
            requested = max(1, round(audio_context_seconds * 40))
            if source_latent is not None:
                rt = min(requested, sa.shape[-1])
                z = sa[..., -rt:].clone()
                overhang = sa.shape[-1] - available * FRAME_RESCALE
                if abs(overhang) >= 0.5:
                    raise ValueError('Source audio/video latent lengths do not share the H3 time grid.')
                end_coord = round(n * FRAME_RESCALE + overhang)
            else:
                if audio_vae is None:
                    raise ValueError('source_audio needs the H3 audio VAE.')
                sr = int(source_audio['sample_rate'])
                # The audio file can overrun the picture. End the window at the actual source frame boundary.
                audio_end = min(round(available / FPS * sr), source_audio['waveform'].shape[-1])
                audio_start = max(0, audio_end - round(audio_context_seconds * sr))
                clip = {'waveform': source_audio['waveform'][..., audio_start:audio_end].clone(), 'sample_rate': sr}
                if audio_end <= audio_start:
                    raise ValueError('Source audio is empty.')
                z = encode_audio(audio_vae, clip)
                rt = z.shape[-1]
                end_coord = round(n * FRAME_RESCALE)
            audio_start_coord = end_coord - rt
            if method == 'pinned_av':
                # Only complete target audio rows before the visual join are fixed.
                start_row = max(0, audio_start_coord)
                stop_row = min(int(n * FRAME_RESCALE), end_coord)
                if stop_row > start_row:
                    pa = prepared['samples'].tensors[1]
                    pa[..., start_row:stop_row] = z[..., start_row-audio_start_coord:stop_row-audio_start_coord].to(pa)
                    prepared['noise_mask'].tensors[1][..., start_row:stop_row] = 0
            kfs.append({'resolved_frame_index': audio_start_coord / FRAME_RESCALE, 'audio_latent': z})

        result = []
        dropped = 0
        for emb, metadata in positive:
            info = metadata.copy()
            kept = []
            for old in info.get('minimax_keyframes', []):
                old = dict(old)
                index = old['resolved_frame_index']
                if 'latent' in old:
                    span = pixel_frames(old['latent'].shape[2])
                    if index < n and index + span > 0:
                        del old['latent']
                        dropped += 1
                # Preserve audio-only guides unless a full original soundtrack replaces them.
                if soundtrack is not None:
                    old.pop('audio_latent', None)
                if 'latent' in old or 'audio_latent' in old:
                    kept.append(old)
            info['minimax_keyframes'] = kept + kfs
            result.append([emb, info])
        if dropped:
            warnings.append(f'Replaced {dropped} overlapping video anchors with the continuation tail; other guides are kept.')
        prepared['h3_end_seconds'] = plan['end_seconds']
        report = f'{origin}; {method}; last {n} frames ({n/FPS:.3f}s); generate {total}, deliver {new_frames} new frames ({new_frames/FPS:.3f}s).'
        if warnings:
            report += '\n' + '\n'.join(warnings)
        return result, prepared, plan, report


class H3ContinuityAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'plan': ('H3_CONTINUITY_PLAN',)},
                'optional': {'audio': ('AUDIO',), 'source_images': ('IMAGE',), 'source_audio': ('AUDIO',)}}

    RETURN_TYPES = ('IMAGE', 'AUDIO', 'STRING')
    RETURN_NAMES = ('images', 'audio', 'report')
    FUNCTION = 'assemble'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Remove the regenerated overlap and align audio to exactly the delivered frames. Optional source inputs prepend the untouched original. No video crossfade or interpolation.'

    def assemble(self, images, plan, audio=None, source_images=None, source_audio=None):
        n, total = plan['context_frames'], plan['target_frames']
        if len(images) != total:
            raise ValueError(f'Decode the full sampler output: expected {total} frames, received {len(images)}.')
        new = images[n:].clone()
        report = f'Removed {n} overlap frames; {len(new)} new frames at 24 fps.'
        if source_images is not None:
            if source_images.shape[1:] != new.shape[1:]:
                raise ValueError('Resize source_images to the output canvas before assembly.')
            # Lightweight seam evidence, not a perceptual quality score.
            seam = float((new[0] - source_images[-1]).abs().mean())
            within = float((source_images[-1] - source_images[-2]).abs().mean()) if len(source_images) > 1 else 0
            report += f' Pixel MAE at join={seam:.5f}; last source transition={within:.5f}. This is not a motion-quality verdict.'
        if 'soundtrack_segment' in plan:
            out_audio = plan['soundtrack_segment']
        elif audio is not None:
            sr = int(audio['sample_rate'])
            a = round(n / FPS * sr)
            b = round(total / FPS * sr)
            out_audio = fit_audio({'waveform': audio['waveform'][..., a:b], 'sample_rate': sr}, round(len(new) / FPS * sr))
        else:
            out_audio = {'waveform': torch.zeros(1, 2, round(len(new) / FPS * 32000)), 'sample_rate': 32000}
            report += ' No generated audio connected; silence used.'
        if source_images is not None:
            sr = int(source_audio['sample_rate'] if source_audio is not None else out_audio['sample_rate'])
            out_audio = fit_audio(out_audio, round(len(new) / FPS * sr), sr)
            count = round(len(source_images) / FPS * sr)
            if source_audio is not None:
                prefix = fit_audio(source_audio, count, sr)['waveform']
                suffix = out_audio['waveform']
                if prefix.shape[1] != suffix.shape[1]:
                    if prefix.shape[1] == 1:
                        prefix = prefix.repeat(1, suffix.shape[1], 1)
                    elif suffix.shape[1] == 1:
                        suffix = suffix.repeat(1, prefix.shape[1], 1)
                    else:
                        raise ValueError('Source and continuation audio channel counts differ.')
            else:
                suffix = out_audio['waveform']
                prefix = suffix.new_zeros(suffix.shape[0], suffix.shape[1], count)
                report += ' Connect source_audio to retain the original sound before the join.'
            out_audio = {'waveform': torch.cat([prefix.to(suffix), suffix], -1), 'sample_rate': sr}
            new = torch.cat([source_images, new.to(source_images)], 0)
        out_audio = fit_audio(out_audio, round(len(new) / FPS * out_audio['sample_rate']))
        return new, out_audio, report


def output_path(name):
    root = Path(folder_paths.get_output_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix != '.safetensors':
        raise ValueError('Use a .safetensors filename inside the ComfyUI output directory.')
    return path


class H3ContinuitySave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': ('LATENT',), 'filename': ('STRING', {'default': 'h3_continuity/take_001.safetensors'})},
                'optional': {'plan': ('H3_CONTINUITY_PLAN',)}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('saved_path',)
    FUNCTION = 'save'
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = 'Save both untrimmed H3 streams without dtype conversion. Existing files get a numbered suffix.'

    def save(self, latent, filename, plan=None):
        v, a = streams(latent)
        path = output_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        base, i = path, 2
        while path.exists():
            path = base.with_stem(f'{base.stem}_{i}')
            i += 1
        meta = {'h3_continuity_version': '1', 'fps': '24'}
        end = plan['end_seconds'] if plan is not None else latent.get('h3_end_seconds', pixel_frames(v.shape[2]) / FPS)
        meta['end_seconds'] = str(end)
        # Write a complete temporary artifact before exposing the final filename.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
        try:
            safetensors.torch.save_file({'video': v.contiguous().cpu(), 'audio': a.contiguous().cpu()}, str(temporary), metadata=meta)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return (str(path.relative_to(Path(folder_paths.get_output_directory()).resolve())),)


class H3ContinuityLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'filename': ('STRING', {'default': 'h3_continuity/take_001.safetensors'})}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Load an untrimmed H3 Continuity checkpoint from output/. No pickle or model downloads.'

    @classmethod
    def IS_CHANGED(cls, filename):
        path = output_path(filename)
        if not path.exists():
            return float('nan')
        stat = path.stat()
        return f'{stat.st_mtime_ns}:{stat.st_size}'

    def load(self, filename):
        path = output_path(filename)
        with safetensors.torch.safe_open(str(path), framework='pt', device='cpu') as f:
            metadata = f.metadata() or {}
            if metadata.get('h3_continuity_version') != '1':
                raise ValueError('This is not an H3 Continuity checkpoint. Connect another pack’s loaded LATENT to Prepare directly.')
            result = {'samples': NestedTensor([f.get_tensor('video'), f.get_tensor('audio')]),
                      'h3_end_seconds': float(metadata['end_seconds'])}
        streams(result)
        return (result,)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3ContinuityImport, H3ContinuityPrepare, H3ContinuityAssemble, H3ContinuitySave, H3ContinuityLoad)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3ContinuityImport': 'H3 Continuity · Import Video',
    'H3ContinuityPrepare': 'H3 Continuity · Prepare',
    'H3ContinuityAssemble': 'H3 Continuity · Assemble',
    'H3ContinuitySave': 'H3 Continuity · Save Latent',
    'H3ContinuityLoad': 'H3 Continuity · Load Latent',
}
