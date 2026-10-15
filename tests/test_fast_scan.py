#!/usr/bin/env python3
"""Tests for fast_scan: cut detection, decision logic, and an end-to-end run on a synthetic
video (still / busy / still picture, a tone burst late in the sound to catch time drift)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fast_scan as fs  # noqa: E402

FPS = fs.SAMPLE_FPS


def _overlap(segs, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)


def _fake_analysis(seconds, motion_fn, vocal_fn=None, speech_fn=None, cuts_at=(), loud_fn=None, jitter=0.1):
    """Analysis dict like fs.analyze() returns, from per-second functions."""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    rng = np.random.default_rng(1)
    motion = np.array([motion_fn(x) for x in t]) * rng.uniform(1 - jitter, 1 + jitter, n)
    diff = motion.copy()
    for c in cuts_at:
        diff[int(c * FPS)] = 0.3
    an = {
        'diff': diff.astype(np.float32), 'motion': motion.astype(np.float32),
        'luma': np.full(n, 0.5, np.float32),
        'vocal': np.array([vocal_fn(x) if vocal_fn else 0.0 for x in t], np.float32),
        'speech': np.array([speech_fn(x) if speech_fn else 0.0 for x in t], np.float32),
        'music': np.zeros(n, np.float32), 'silence': np.zeros(n, np.float32),
        'loud': np.array([loud_fn(x) if loud_fn else -40.0 for x in t], np.float32),
        'meta': {'duration': seconds, 'fps': 25.0, 'has_audio': True},
    }
    return an


# ---------- Units ----------

def test_detect_cuts_ignores_fast_motion():
    rng = np.random.default_rng(0)
    diff = rng.uniform(0.01, 0.03, 3000)
    # Busy stretch: high but not isolated; starts and stops abruptly (must not look like cuts).
    # ±20 % jitter – real motion is continuous from frame to frame, unlike white noise
    diff[1000:1300] = rng.uniform(0.12, 0.18, 300)
    for c in (500, 2000, 2500):
        diff[c] = 0.25
    cuts = fs.detect_cuts(diff)
    assert cuts.tolist() == [500, 2000, 2500], cuts.tolist()


def test_mask_helpers():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], bool)
    assert fs.mask_to_runs(m) == [(1, 3), (5, 6), (9, 10)]
    assert fs.mask_to_runs(fs.fill_gaps(m, 2)) == [(1, 6), (9, 10)]
    h = fs.hysteresis(np.array([10, 55, 51, 49, 47, 60]), 50, band=4)
    assert h.tolist() == [False, True, True, True, False, True]


def test_full_length_parts():
    parts = fs.full_length_parts([(0.0, 5.0), (10.0, 20.0)], 30.0)
    assert parts == [(0.0, 5.0, True), (5.0, 10.0, False), (10.0, 20.0, True), (20.0, 30.0, False)], parts
    assert fs.full_length_parts([], 12.0) == [(0.0, 12.0, False)]


def test_overrides(tmp_path: Path):
    segs = [(40.0, 80.0)]
    # keep a gap, cut a hole, later corrections win over earlier ones
    out = fs.apply_overrides(segs, [(5.0, 35.0, 'keep'), (50.0, 60.0, 'drop')], 100.0)
    assert out == [(5.0, 35.0), (40.0, 50.0), (60.0, 80.0)], out
    out = fs.apply_overrides(segs, [(50.0, 60.0, 'drop'), (55.0, 58.0, 'keep')], 100.0)
    assert out == [(40.0, 50.0), (55.0, 58.0), (60.0, 80.0)], out
    assert fs.apply_overrides(segs, [(70.0, 200.0, 'keep')], 100.0) == [(40.0, 100.0)]
    # through decide(): a 2 s manual keep survives min_len, persisted per video
    an = _fake_analysis(120, motion_fn=lambda x: 0.05 if 40 <= x < 80 else 0.01)
    st = fs.Settings(motion_level=70, vocal_keep=False, speech_mode='ignore', min_len_s=4)
    res = fs.decide(an, st, [(10.0, 12.0, 'keep')])
    assert _overlap(res['segments'], 10, 12) == 2.0 and res['overrides'] == [(10.0, 12.0, 'keep')], res['segments']
    vid = Path(tmp_path) / 'v.mp4'
    vid.write_bytes(b'x')
    fs.save_overrides(str(vid), [(10.0, 12.0, 'keep')], cache_dir=str(tmp_path))
    assert fs.load_overrides(str(vid), cache_dir=str(tmp_path)) == [(10.0, 12.0, 'keep')]
    fs.save_overrides(str(vid), [], cache_dir=str(tmp_path))
    assert fs.load_overrides(str(vid), cache_dir=str(tmp_path)) == []


def test_settings_roundtrip():
    s = fs.Settings(motion_level=70, speech_mode='keep')
    d = s.to_dict()
    d['unknown_key'] = 1  # old settings files must not break loading
    assert fs.Settings.from_dict(d) == s


def test_decide_motion_vocal_speech():
    # busy 60–120 s, vocal 200–230 s (quiet picture), speech inside the busy part 90–100 s
    an = _fake_analysis(
        300,
        motion_fn=lambda x: 0.05 if 60 <= x < 120 else 0.01,
        vocal_fn=lambda x: 0.8 if 200 <= x < 230 else 0.0,
        speech_fn=lambda x: 0.9 if 90 <= x < 100 else 0.0,
        cuts_at=(60.1,),
    )
    s = fs.Settings(motion_level=70, speech_mode='ignore', vocal_keep=False)
    segs = fs.decide(an, s)['segments']
    assert _overlap(segs, 60, 120) > 55 and _overlap(segs, 0, 55) == 0, segs
    assert any(abs(a - 60.1) < 0.01 for a, _ in segs), f'start not snapped to the cut: {segs}'

    segs = fs.decide(an, fs.Settings(motion_level=70, speech_mode='remove', vocal_keep=False))['segments']
    assert _overlap(segs, 91, 99) == 0 and _overlap(segs, 60, 88) > 25, segs

    segs = fs.decide(an, fs.Settings(motion_level=70, speech_mode='ignore', vocal_keep=True))['segments']
    assert _overlap(segs, 200, 230) > 25, segs

    segs = fs.decide(an, fs.Settings(motion_keep=False, vocal_keep=False, speech_mode='ignore'))['segments']
    assert segs == [], segs


def _beat(x, base):
    """Rhythmic motion as in action scenes: 1.5 repetitions per second."""
    return base * (1 + 0.6 * np.sin(2 * np.pi * 1.5 * x))


def test_motion_rhythm():
    t = np.arange(1200) / FPS
    rng = np.random.default_rng(3)
    beat = _beat(t, 0.05) * rng.uniform(0.9, 1.1, len(t))
    noise = 0.05 * rng.uniform(0.4, 1.6, 6000)
    r_beat, r_noise = fs.motion_rhythm(beat), fs.motion_rhythm(noise)
    # irregular motion (here: noise) sits around 0.23 like the unwanted parts of real films, just
    # below the default threshold – over a few minutes it can wander above it
    assert r_beat[300:900].min() > 0.5, r_beat[300:900].min()
    assert np.median(r_noise) < fs.extend_level(50) and r_noise.max() < 0.3, (np.median(r_noise), r_noise.max())
    # a busier production (all motion 1.6 times higher) has exactly the same rhythm
    assert np.allclose(fs.motion_rhythm(beat * 1.6), r_beat)
    assert fs.motion_rhythm(np.full(50, 0.01)).tolist() == [0.0] * 50  # shorter than one window


def test_extend_mode():
    # Rhythmic action scenes, each with a short burst of strong motion (the part a strict
    # classic setting finds): 60–300 s with a burst 120–130 and 80 s of calm talk 160–240,
    # 450–550 s with a burst 500–510. 700–760 s: the same rhythmic motion but no burst – nothing
    # picked it, so it must not grow into a scene. 950–960 s: very strong but irregular motion –
    # classic picks it, the combined mode drops it. Everything else: an almost still picture.
    def motion(x):
        if 120 <= x < 130 or 500 <= x < 510:
            return _beat(x, 0.2)
        if 950 <= x < 960:
            return 0.3 * np.random.default_rng(int(x * FPS)).uniform(0.4, 1.6)
        if 160 <= x < 240:
            return 0.012
        return _beat(x, 0.05) if (60 <= x < 300 or 450 <= x < 550 or 700 <= x < 760) else 0.01
    an = _fake_analysis(1200, motion_fn=motion, speech_fn=lambda x: 0.8 if 160 <= x < 240 else 0.0, jitter=0.0)
    base = dict(motion_level=95, vocal_keep=False, speech_level=40, min_len_s=4, merge_gap_s=2)
    classic = fs.decide(an, fs.Settings(**base))
    assert _overlap(classic['segments'], 115, 135) > 8 and _overlap(classic['segments'], 950, 960) > 8, classic['segments']
    assert _overlap(classic['segments'], 60, 110) == 0, classic['segments']
    res = fs.decide(an, fs.Settings(**base, decision='extend', strictness=50, scene_gap_s=60))
    segs = res['segments']
    assert _overlap(segs, 65, 295) > 220, f'first scene grows to its full length, talk included: {segs}'
    assert _overlap(segs, 455, 545) > 85, segs
    assert _overlap(segs, 700, 760) == 0, f'rhythm without a pick stays out: {segs}'
    assert _overlap(segs, 940, 970) == 0, f'a pick without rhythm is dropped: {segs}'
    assert _overlap(segs, 0, 30) == 0 and _overlap(segs, 350, 400) == 0 and _overlap(segs, 1000, 1200) == 0, segs
    assert 'region' in res['curves'] and 0 <= res['region_line'] <= 100
    # short allowed pauses: the long calm talk ends the first scene
    short = fs.decide(an, fs.Settings(**base, decision='extend', strictness=50, scene_gap_s=5))['segments']
    assert _overlap(short, 185, 215) == 0 and _overlap(short, 65, 150) > 80, short
    assert _overlap(short, 250, 300) == 0, f'the rest after the pause has no pick of its own: {short}'
    # classic stays exactly as before, and old saved modes load as the combined one
    assert fs.decide(an, fs.Settings(**base, decision='classic'))['segments'] == classic['segments']
    assert fs.Settings.from_dict({'decision': 'score', 'block_s': 20}).decision == 'extend'


def test_override_helpers():
    ov = fs.set_override([(10.0, 20.0, 'drop')], 15.0, 30.0, 'keep')
    assert ov == [(10.0, 15.0, 'drop'), (15.0, 30.0, 'keep')], ov
    assert fs.set_override(ov, 0.0, 100.0, None) == []
    assert fs.spans_minus([(0.0, 10.0), (20.0, 30.0)], [(5.0, 25.0)]) == [(0.0, 5.0), (25.0, 30.0)]
    # exported 10–20 and 40–50; in the edit 10–20 got trimmed to 12–20 (+ 1 frame rounding at
    # the end), 40–50 deleted, 60–65 added
    out = fs.edit_to_overrides([(10.0, 20.0), (40.0, 50.0)], [(12.0, 20.04), (60.0, 65.0)])
    assert out == [(10.0, 12.0, 'drop'), (40.0, 50.0, 'drop'), (60.0, 65.0, 'keep')], out


def test_labels_and_export_record(tmp_path: Path):
    vid = Path(tmp_path) / 'v.mp4'
    vid.write_bytes(b'x')
    c = str(tmp_path)
    assert fs.load_labels(str(vid), c) is None and fs.load_export_record(str(vid), c) is None
    fs.save_export_record(str(vid), [(1.0, 2.5)], True, 'DaVinci: tl', cache_dir=c)
    rec = fs.load_export_record(str(vid), c)
    assert rec['segments'] == [(1.0, 2.5)] and rec['full_length'] is True, rec
    fs.save_labels(str(vid), [(0.0, 3.0)], reviewed=True, source='davinci', cache_dir=c)
    lab = fs.load_labels(str(vid), c)
    assert lab['final'] == [(0.0, 3.0)] and lab['reviewed'] is True, lab
    assert fs.load_video_settings(str(vid), c) is None
    fs.save_video_settings(str(vid), fs.Settings(decision='extend', strictness=70), cache_dir=c)
    assert fs.load_video_settings(str(vid), c) == fs.Settings(decision='extend', strictness=70)


def test_pool_embeddings():
    emb = np.zeros((5, 1024), np.float32)
    emb[0, 0], emb[1, 0], emb[4, 1] = 1.0, 3.0, 9.0
    p = fs._pool_embeddings(emb)
    assert p.shape == (3, 1024) and p.dtype == np.uint8, p.shape
    assert p[0, 0] == round(2.0 * fs.EMBED_SCALE) and p[2, 1] == 255, (p[0, 0], p[2, 1])  # mean, clipped


def test_decide_min_len_and_still_frames():
    # 2 s burst is shorter than min_len; the rest is a still picture (title card)
    an = _fake_analysis(120, motion_fn=lambda x: 0.05 if 50 <= x < 52 else 0.0005)
    res = fs.decide(an, fs.Settings(min_len_s=4.0, vocal_keep=False, speech_mode='ignore'))
    assert res['segments'] == [], res['segments']
    res = fs.decide(an, fs.Settings(min_len_s=1.0, vocal_keep=False, speech_mode='ignore'))
    assert len(res['segments']) == 1 and _overlap(res['segments'], 50, 52) > 1.5, res['segments']


# ---------- End to end ----------

def _make_video(path: Path) -> Path:
    """150 s: still grey 0–40, game of life 40–80 (motion everywhere), lighter still 80–150.
    Sound: silence with a loud 440 Hz tone 130–131 s. The first seconds of life (random start,
    nearly every pixel changes) are skipped – no real shot looks like that."""
    vf = (
        'color=c=0x404040:s=320x180:r=25:d=40[a];'
        'life=s=320x180:r=25:mold=10:ratio=0.3:death_color=0x202020:life_color=0xe0e0e0,'
        'trim=start=5:duration=40,setpts=PTS-STARTPTS[b];'
        'color=c=0x909090:s=320x180:r=25:d=70[c];'
        '[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]'
    )
    cmd = ['ffmpeg', '-y', '-v', 'error', '-filter_complex', vf,
           '-f', 'lavfi', '-i', "aevalsrc='if(between(t,130,131),0.8*sin(2*PI*440*t),0)':s=16000:d=150",
           '-map', '[v]', '-map', '0:a', '-c:v', 'libx264', '-preset', 'ultrafast', '-g', '50',
           '-c:a', 'aac', '-shortest', str(path)]
    subprocess.run(cmd, check=True, creationflags=fs.CREATE_NO_WINDOW)
    return path


def _video(tmp_path: Path) -> Path:
    cached = _ROOT / 'output' / '_fast_scan_test' / 'synthetic.mp4'
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        _make_video(cached)
    dst = Path(tmp_path) / 'synthetic.mp4'
    shutil.copy(cached, dst)
    return dst


def test_end_to_end(tmp_path: Path):
    vid = _video(tmp_path)
    cache_dir = Path(tmp_path) / 'cache'
    an = fs.analyze(str(vid), cache_dir=str(cache_dir))
    n = len(an['diff'])
    assert abs(n - 150 * FPS) <= 2, n
    assert all(len(an[k]) == n for k in fs.VIDEO_KEYS + fs.AUDIO_KEYS)
    assert an['meta']['has_audio']

    # Sound on the right time even after two 60 s blocks (no drift)
    loud_t = np.arange(n)[an['loud'] > -20] / FPS
    assert len(loud_t) and 129.3 <= loud_t.min() and loud_t.max() <= 131.7, (loud_t.min(), loud_t.max())

    # Cuts at the picture changes
    cuts = fs.detect_cuts(an['diff']) / FPS
    for t in (40, 80):
        assert np.min(np.abs(cuts - t)) <= 0.2, cuts

    res = fs.decide(an, fs.Settings(vocal_keep=False, speech_mode='ignore'))
    segs = res['segments']
    assert _overlap(segs, 40, 80) > 36, segs
    assert _overlap(segs, 0, 39) == 0 and _overlap(segs, 81, 150) == 0, segs

    # Second call comes from the cache
    an2 = fs.analyze(str(vid), cache_dir=str(cache_dir))
    assert np.array_equal(an2['motion'], an['motion'])
    assert len(list(cache_dir.glob('*.features.npz'))) == 1

    # YAMNet embeddings stored next to it, one per 0.96 s
    emb = fs.load_embeddings(str(vid), cache_dir=str(cache_dir))
    assert emb is not None and emb[0].shape[1] == 1024 and abs(len(emb[0]) - 150 / 0.96) <= 2, emb[0].shape
    assert abs(emb[1][-1] - 150) < 2, emb[1][-1]

    # Exports
    xml = fs.export_davinci(str(vid), segs, an['meta']['fps'], str(tmp_path), 'xml')
    edl = fs.export_davinci(str(vid), segs, an['meta']['fps'], str(tmp_path), 'edl')
    assert Path(xml).is_file() and Path(edl).is_file()
    full = fs.export_davinci(str(vid), segs, an['meta']['fps'], str(tmp_path), 'edl', full_length=True,
                             duration=an['meta']['duration'])
    events = [ln for ln in Path(full).read_text(encoding='utf-8').splitlines() if ln[:3].isdigit()]
    assert full.endswith('_scenecut_full.edl') and len(events) == 3, events  # before / kept / after
    mp4 = fs.export_mp4(str(vid), [(40.0, 50.0), (60.0, 65.0)], str(tmp_path))
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', mp4],
                         capture_output=True, text=True)
    dur = float(json.loads(out.stdout)['format']['duration'])
    assert abs(dur - 15.0) < 0.5, dur


def test_cancel(tmp_path: Path):
    import threading
    vid = _video(tmp_path)
    ev = threading.Event()
    ev.set()
    try:
        fs.analyze(str(vid), cache_dir=str(Path(tmp_path) / 'c'), use_cache=False, cancel=ev)
    except fs.Cancelled:
        return
    raise AssertionError('analysis was not cancelled')


def main() -> int:
    tests = [
        test_detect_cuts_ignores_fast_motion,
        test_mask_helpers,
        test_full_length_parts,
        test_overrides,
        test_settings_roundtrip,
        test_decide_motion_vocal_speech,
        test_motion_rhythm,
        test_extend_mode,
        test_override_helpers,
        test_labels_and_export_record,
        test_pool_embeddings,
        test_decide_min_len_and_still_frames,
        test_end_to_end,
        test_cancel,
    ]
    work_root = _ROOT / 'output' / '_fast_scan_test'
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
