"""Temporal interpolation onto the reference images, never fixed global bins."""
import bisect
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameSelection:
    keyframes: tuple[int, ...]
    query: int
    target: tuple[float, ...]


def soft_target(keyframes, query):
    if len(keyframes) < 2 or any(a >= b for a, b in zip(keyframes, keyframes[1:])):
        raise ValueError('Keyframes must be distinct and chronologically ordered')
    if not keyframes[0] < query < keyframes[-1] or query in keyframes:
        raise ValueError('Query must be strictly internal and not a keyframe')
    right = bisect.bisect_right(keyframes, query)
    left = right - 1
    alpha = (query - keyframes[left]) / (keyframes[right] - keyframes[left])
    target = [0.0] * len(keyframes)
    target[left], target[right] = 1.0 - alpha, alpha
    return tuple(target)


def select_frames(num_frames, min_keyframes, max_keyframes, rng):
    # A complete candidate needs at least K keyframes + one distinct query.
    upper = min(max_keyframes, num_frames - 1)
    if upper < min_keyframes:
        return None
    count = int(rng.integers(min_keyframes, upper + 1))
    keys = sorted(int(x) for x in rng.choice(num_frames, count, replace=False))
    # Sample uniformly from available interior frames without allocating an
    # array the length of the video. Empty individual intervals are valid.
    gaps = np.array([b - a - 1 for a, b in zip(keys, keys[1:])], dtype=np.int64)
    available = int(gaps.sum())
    if not available:
        return None
    offset = int(rng.integers(available))
    for left, gap in zip(keys, gaps):
        if offset < gap:
            query = left + 1 + offset
            return FrameSelection(tuple(keys), query, soft_target(keys, query))
        offset -= int(gap)
    raise AssertionError('Unreachable query selection')


def collect_batch(draw, batch_size, max_rounds=100):
    """B+1 candidates first; retain first B valid samples, refill if needed."""
    valid = []
    attempts = 0
    for round_index in range(max_rounds):
        count = batch_size + 1 if round_index == 0 else batch_size - len(valid) + 1
        candidates = [draw() for _ in range(count)]
        attempts += count
        valid.extend(sample for sample in candidates if sample is not None)
        if len(valid) >= batch_size:
            return valid[:batch_size], attempts
    raise RuntimeError(f'Cannot fill a batch of {batch_size} after {attempts} candidates')
