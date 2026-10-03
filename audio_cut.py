"""Sample-preserving audio cuts using a video frame boundary."""


class H3ContinuityAudioCut:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio': ('AUDIO',),
            'context_frames': ('INT', {'default': 22, 'min': 0, 'max': 8640000}),
            'framerate': ('FLOAT', {'default': 24.0, 'min': 0.001, 'max': 1000.0,
                                   'step': 0.001,
                                   'tooltip': 'Video frames per second. Audio cut time = context_frames / framerate.'}),
        }}

    RETURN_TYPES = ('AUDIO', 'AUDIO')
    RETURN_NAMES = ('generated_audio', 'context_audio')
    FUNCTION = 'cut'
    CATEGORY = 'MiniMax H3/Continuity'
    DESCRIPTION = ('Remove the initial context from decoded audio using a video frame count. '
                   'Returns the remaining audio first and the initial context second. '
                   'Cuts at the nearest sample, preserving channels, sample rate and precision; '
                   'no resampling, fades or encoding.')

    def cut(self, audio, context_frames, framerate):
        if framerate <= 0 or context_frames < 0:
            raise ValueError('framerate must be positive and context_frames nonnegative.')
        wave = audio['waveform']
        cut = min(wave.shape[-1], round(context_frames * int(audio['sample_rate']) / framerate))
        context = dict(audio, waveform=wave[..., :cut].clone())
        generated = dict(audio, waveform=wave[..., cut:].clone())
        if 'h3_valid_samples' in audio:
            valid = max(0, min(wave.shape[-1], int(audio['h3_valid_samples'])))
            context['h3_valid_samples'] = min(valid, cut)
            generated['h3_valid_samples'] = max(0, valid - cut)
        return generated, context


NODE_CLASS_MAPPINGS = {'H3ContinuityAudioCut': H3ContinuityAudioCut}
NODE_DISPLAY_NAME_MAPPINGS = {'H3ContinuityAudioCut': 'H3 Continuity · Cut Audio Context'}
