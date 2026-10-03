"""Frame counts for image batches and ComfyUI image lists."""


from comfy.nested_tensor import NestedTensor

from .timing import frame_edges


def count_images(images):
    return sum(1 if len(image.shape) == 3 else len(image) for image in images)


class H3ImageCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ('INT',)
    RETURN_NAMES = ('frame_count',)
    FUNCTION = 'count'
    CATEGORY = 'MiniMax H3/Continuity'
    DESCRIPTION = 'Count all frames in an image batch or a list of image batches.'

    def count(self, images):
        return (count_images(images),)


class H3ContinuationFrameCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'seconds': ('FLOAT', {'default': 5.0, 'min': 0.0, 'max': 86400.0,
                                  'step': 0.01,
                                  'tooltip': 'New duration to add, at 24 fps. Only up to context_frames from the batch are included before snapping the total to 5 + 17k.'}),
        }, 'optional': {
            'images': ('IMAGE',),
            'context_frames': ('INT', {'default': 22, 'min': 0, 'max': 3600,
                                       'step': 1,
                                       'tooltip': 'Maximum source frames included in the calculation. Match the context_frames used in Continuity Prepare.'}),
            'latent': ('LATENT', {'tooltip': 'H3 audio/video latent. Takes priority over images when both are connected; counts frames without VAE decoding.'}),
        }}

    INPUT_IS_LIST = True
    RETURN_TYPES = ('INT', 'INT')
    RETURN_NAMES = ('total_frames', 'source_frames')
    FUNCTION = 'calculate'
    CATEGORY = 'MiniMax H3/Continuity'
    DESCRIPTION = 'Add min(source frame count from H3 latent or images, context_frames) to max(5, round(seconds * 24)), then round the total up to the H3 grid (5 + 17k).'

    def calculate(self, images=None, seconds=None, context_frames=None, latent=None):
        if latent:
            source_frames = 0
            for item in latent:
                samples = item['samples']
                if not isinstance(samples, NestedTensor) or len(samples.tensors) != 2:
                    raise ValueError('Connect a MiniMax H3 audio/video latent.')
                video = samples.tensors[0]
                if video.ndim != 5 or video.shape[:2] != (1, 24):
                    raise ValueError('Expected H3 video latent [1,24,T,H,W].')
                source_frames += frame_edges(video.shape[2])[-1]
        elif images is not None:
            source_frames = count_images(images)
        else:
            raise ValueError('Connect images or an H3 latent to count the source frames.')
        context = min(source_frames, context_frames[0]) if context_frames is not None else source_frames
        total = context + max(5, round(seconds[0] * 24))
        return (total + (5 - total % 17) % 17, source_frames)


NODE_CLASS_MAPPINGS = {
    'H3ImageCount': H3ImageCount,
    'H3ContinuationFrameCount': H3ContinuationFrameCount,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3ImageCount': 'H3 Continuity · Count Images',
    'H3ContinuationFrameCount': 'H3 Continuity · Continuation Frames',
}
