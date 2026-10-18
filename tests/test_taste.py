#!/usr/bin/env python3
"""Tests for taste marks (taste.py) and that evaluate.py leaves marked parts out."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import evaluate as ev  # noqa: E402
import fast_scan as fs  # noqa: E402
import taste  # noqa: E402


def test_marks_roundtrip(tmp_path: Path):
    vid, c = str(Path(tmp_path) / 'v.mp4'), str(tmp_path)
    Path(vid).write_bytes(b'x')
    taste.save_marks(vid, [(10.0, 20.0, 'drop')], cache_dir=c, decision='classic')
    assert taste.load_marks(vid, c, 'classic') == [(10.0, 20.0, 'drop')]
    assert taste.load_marks(vid, c, 'extend') == [], 'per mode'
    taste.save_marks(vid, [], cache_dir=c, decision='classic')
    assert taste.load_marks(vid, c, 'classic') == []
    assert not Path(taste._path(vid, c, 'classic')).exists(), 'no marks, no file'


def test_mark_and_prune():
    overrides = [(10.0, 20.0, 'drop'), (30.0, 40.0, 'keep')]
    # marking a range takes the corrections inside it, clipped, with their kind
    marks = taste.mark_corrections([], overrides, 15.0, 35.0)
    assert marks == [(15.0, 20.0, 'drop'), (30.0, 35.0, 'keep')], marks
    assert taste.mark_corrections([], overrides, 21.0, 29.0) == [], 'nothing to mark between corrections'
    assert taste.unmark(marks, 0.0, 25.0) == [(30.0, 35.0, 'keep')]
    # a mark goes with its correction: removed or turned the other way
    assert taste.prune(marks, [(30.0, 40.0, 'keep')]) == [(30.0, 35.0, 'keep')]
    assert taste.prune(marks, [(10.0, 20.0, 'keep'), (30.0, 40.0, 'keep')]) == [(30.0, 35.0, 'keep')]
    # a new edit that decides the same way keeps the mark (also over a re-split correction)
    assert taste.prune(marks, [(12.0, 18.0, 'drop'), (18.0, 22.0, 'drop'), (30.0, 40.0, 'keep')]) == \
        [(15.0, 18.0, 'drop'), (18.0, 20.0, 'drop'), (30.0, 35.0, 'keep')]
    m = taste.mask([(1.0, 2.0, 'drop')], 50)
    assert m.sum() == fs.SAMPLE_FPS and m[int(fs.SAMPLE_FPS)], m


def test_evaluate_leaves_taste_out(tmp_path: Path):
    vid, c = str(Path(tmp_path) / 'v.mp4'), str(tmp_path)
    Path(vid).write_bytes(b'x')
    n = int(100 * fs.SAMPLE_FPS)
    an = {k: np.zeros(n, np.float32) for k in fs.VIDEO_KEYS + fs.AUDIO_KEYS}
    an['meta'] = {'duration': 100.0, 'has_audio': False}
    fs.save_labels(vid, [(20.0, 60.0)], reviewed=True, source='test', decision='extend', cache_dir=c)
    # 40–50 s cut by taste inside the scene (outside the final cut), 20–25 s kept by taste
    taste.save_marks(vid, [(20.0, 25.0, 'keep'), (40.0, 50.0, 'drop')], cache_dir=c, decision='extend')
    fs.save_labels(vid, [(20.0, 40.0), (50.0, 60.0)], reviewed=True, source='test', decision='extend', cache_dir=c)
    wanted, accepted, unwanted, reviewed, marked = ev.ground_truth(an, vid, fs.Settings(), c, 'extend')
    sec = lambda m: m.sum() / fs.SAMPLE_FPS
    assert reviewed and sec(marked) == 15.0
    assert sec(wanted) == 25.0 and sec(unwanted) == 60.0 and not accepted.any(), (sec(wanted), sec(unwanted))
    assert not (marked & (wanted | unwanted)).any(), 'taste counts neither way'


def main():
    tests = [test_marks_roundtrip, test_mark_and_prune, test_evaluate_leaves_taste_out]
    work_root = _ROOT / 'output' / '_taste_test'
    work_root.mkdir(parents=True, exist_ok=True)
    failed = 0
    for fn in tests:
        name = fn.__name__
        try:
            if fn.__code__.co_argcount:
                case_dir = work_root / name
                shutil.rmtree(case_dir, ignore_errors=True)
                case_dir.mkdir(parents=True)
                fn(case_dir)
            else:
                fn()
            print(f'PASS  {name}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
