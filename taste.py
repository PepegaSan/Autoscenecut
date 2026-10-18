"""Taste marks (test version): which of your own corrections you made because of taste – how
something is shown, not whether it is action.

A mark changes nothing in the cut, the correction under it does that. It only says why: the
general decision is measured without the marked parts (evaluate.py), and the marks are the raw
material for a personal taste model later, which every user trains on their own marks.

A mark only stands where a correction of the same kind lies (a taste 'drop' on an 'always cut'
correction): when the correction goes or turns the other way, the mark goes with it.

Per video and mode (fs.MODES) in the cache: <name>.taste.<mode>.json = [[start, end, 'keep' | 'drop'], ...]
"""
import os

import numpy as np

import fast_scan as fs


def _path(video, cache_dir=None, decision='extend'):
    return fs._side_path(video, f'taste.{fs.mode_of(decision)}.json', cache_dir)


def load_marks(video, cache_dir=None, decision='extend'):
    try:
        return [(float(a), float(b), str(m)) for a, b, m in fs._read_json(_path(video, cache_dir, decision)) or []]
    except (ValueError, TypeError):
        return []


def save_marks(video, marks, cache_dir=None, decision='extend'):
    path = _path(video, cache_dir, decision)
    if not marks:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    fs._write_json(path, [[round(a, 2), round(b, 2), m] for a, b, m in marks])


def mark_corrections(marks, overrides, a, b):
    """marks plus the parts of the corrections inside [a, b): they were made because of taste."""
    out = list(marks)
    for o_a, o_b, mode in overrides:
        lo, hi = max(a, o_a), min(b, o_b)
        if hi > lo:
            out = fs.set_override(out, lo, hi, mode)
    return out


def unmark(marks, a, b):
    return fs.set_override(marks, a, b, None)


def prune(marks, overrides):
    """Only the parts of marks that still lie on a correction of the same kind."""
    out = []
    for m_a, m_b, mode in marks:
        for o_a, o_b, o_mode in overrides:
            lo, hi = max(m_a, o_a), min(m_b, o_b)
            if o_mode == mode and hi - lo >= 0.1:
                out.append((round(lo, 2), round(hi, 2), mode))
    return sorted(out)


def mask(marks, n):
    """Frames on the SAMPLE_FPS grid covered by a mark."""
    m = np.zeros(n, bool)
    for a, b, _mode in marks:
        m[int(round(a * fs.SAMPLE_FPS)):int(round(b * fs.SAMPLE_FPS))] = True
    return m
