"""Explicit reference/query roles using standard Qwen image boundary tokens."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from robometer.data.datasets.helpers import resolve_processed_path


def build_conversation(task, reference_images, query_image, max_pixels=None):
    content = [{'type': 'text', 'text': (
        f'Task: {task}\nThe following ordered reference keyframes define relative progress for this query. '
        'The first reference means 0% and the last means 100% of this reference segment, '
        'not necessarily completion of the entire task.\nBEGIN REFERENCE KEYFRAMES\n')}]
    extras = {} if max_pixels is None else {'max_pixels': int(max_pixels)}
    k = len(reference_images)
    for i, img in enumerate(reference_images):
        content.extend([{'type': 'text', 'text': f'Reference {i + 1} of {k}, progress {100 * i / (k - 1):.4f}%:\n'},
                        {'type': 'image', 'image': img, **extras}])
    content.extend([
        {'type': 'text', 'text': '\nEND REFERENCE KEYFRAMES\nCURRENT QUERY IMAGE (not a reference):\n'},
        {'type': 'image', 'image': query_image, **extras},
        {'type': 'text', 'text': '\nLocate the current state relative to the reference keyframes. Relative progress:'},
    ])
    return [{'role': 'user', 'content': content}]


class ICLCollator:
    def __init__(self, processor, root, max_keyframes=11, max_pixels=None):
        self.processor, self.root = processor, Path(root)
        self.max_keyframes, self.max_pixels = max_keyframes, max_pixels
        self.end_token = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.processor.tokenizer.padding_side = 'right'

    @staticmethod
    def _pil(frame):
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            raise ValueError(f'Expected uint8 cached frames, got {frame.dtype}')
        return Image.fromarray(frame).convert('RGB')

    def __call__(self, items):
        if len(items) != 1:
            raise ValueError('Outer DataLoader batch_size must be 1 (each item is a full GPU batch)')
        packed = items[0]
        conversations, selections = [], []
        for sample in packed['samples']:
            row, selected = sample['trajectory'], sample['selection']
            path = Path(resolve_processed_path(row['frames'], str(self.root)))
            with np.load(path) as cached:
                frames = cached['frames']
                if max(*selected.keyframes, selected.query) >= len(frames):
                    raise ValueError(f'Frame metadata does not match cache: {path}')
                refs = [self._pil(frames[i]) for i in selected.keyframes]
                query = self._pil(frames[selected.query])
            conversations.append(build_conversation(row['task'], refs, query, self.max_pixels))
            selections.append(selected)
        texts = [self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in conversations]
        patch_size = self.processor.image_processor.patch_size
        images, _ = process_vision_info(conversations, image_patch_size=patch_size)
        inputs = self.processor(text=texts, images=images, padding=True, truncation=False,
                                do_resize=False, return_tensors='pt')
        batch = len(selections)
        positions = torch.zeros(batch, self.max_keyframes, dtype=torch.long)
        key_mask = torch.zeros(batch, self.max_keyframes, dtype=torch.bool)
        labels = torch.zeros(batch, self.max_keyframes, dtype=torch.float32)
        for i, selected in enumerate(selections):
            k = len(selected.keyframes)
            ends = torch.nonzero(inputs['input_ids'][i] == self.end_token, as_tuple=False).flatten()
            if len(ends) != k + 1:
                raise ValueError(f'Expected {k} references plus one query; found {len(ends)} images')
            positions[i, :k] = ends[:k]
            key_mask[i, :k] = True
            labels[i, :k] = torch.tensor(selected.target)
        # Last non-padding token follows the query and can attend to all images.
        query_positions = inputs['attention_mask'].sum(-1) - 1
        inputs.update(anchor_positions=positions, query_positions=query_positions,
                      anchor_mask=key_mask, labels=labels)
        return dict(inputs)
