"""Fast single-pass scene scan: motion + sound.

Picture: one sequential ffmpeg decode (NVDEC if available) into tiny grayscale frames, no
seeking. Per frame: difference to the previous frame and a camera-motion-compensated difference
(phase correlation), so pans count less than motion in the picture. Cuts are isolated
difference spikes (see detect_cuts).

Sound: the whole track decoded once (16 kHz mono) and run through YAMNet in 60 s blocks, at
the same time as the picture. Per 0.48 s: speech, vocal (moan, breath, scream …), music,
silence and loudness.

All curves end up on one 10 fps grid. decide() turns them into keep-segments from a Settings
object; that part takes milliseconds, so the GUI can redo it on every slider move. The
analysis is cached next to the output (<video>.features.npz).

Usage:
    python fast_scan.py VIDEO [--out DIR] [--no-cache] [--motion-level 60] [--export xml,edl,mp4]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
W, H = 160, 90
SAMPLE_FPS = 10.0
FEATURES_VERSION = 3

AUDIO_SR = 16000
YAMNET_HOP = 7680  # samples (0.48 s)
YAMNET_WINDOW = 15600  # samples (0.975 s)
AUDIO_BLOCK = YAMNET_HOP * 125  # 60 s per model call

# YAMNet display names (lower case) per group; group score = highest class probability
AUDIO_GROUPS = {
    'speech': ['speech', 'male speech, man speaking', 'female speech, woman speaking',
               'child speech, kid speaking', 'conversation', 'narration, monologue'],
    'vocal': ['whimper', 'wail, moan', 'sigh', 'groan', 'grunt', 'breathing', 'gasp', 'pant',
              'screaming', 'shout', 'yell', 'crying, sobbing'],
    'music': ['music', 'musical instrument', 'vocal music', 'background music', 'soundtrack music',
              'pop music', 'electronic music', 'hip hop music', 'rock music', 'electronic dance music'],
    'silence': ['silence'],
}
AUDIO_KEYS = ('speech', 'vocal', 'music', 'silence', 'loud')
VIDEO_KEYS = ('diff', 'motion', 'luma')


class Cancelled(Exception):
    pass


def _module_dir():
    return getattr(sys, '_MEIPASS', None) or os.path.dirname(os.path.abspath(__file__))


def probe(video):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,r_frame_rate:format=duration',
         '-of', 'json', video],
        capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
    )
    info = json.loads(out.stdout or '{}')
    streams = info.get('streams', [])
    v = next((s for s in streams if s.get('codec_type') == 'video'), {})
    num, _, den = (v.get('r_frame_rate') or '30/1').partition('/')
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    has_audio = any(s.get('codec_type') == 'audio' for s in streams)
    return float(info.get('format', {}).get('duration') or 0.0), fps, has_audio


# ---------- Picture ----------

def iter_frames(video, hwaccel=True, cancel=None):
    """Yield (H, W) uint8 frames at SAMPLE_FPS from one sequential decode."""
    cmd = ['ffmpeg', '-v', 'error', '-nostdin']
    if hwaccel:
        cmd += ['-hwaccel', 'cuda']
    cmd += ['-i', video, '-an', '-sn', '-dn',
            '-vf', f'fps={SAMPLE_FPS},scale={W}:{H}:flags=area,format=gray',
            '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=CREATE_NO_WINDOW)
    # Drain stderr so a chatty ffmpeg can't block on a full pipe
    err_lines = []
    t_err = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    t_err.start()
    size = W * H
    n = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise Cancelled()
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            n += 1
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
    finally:
        proc.stdout.close()
        proc.wait()
        t_err.join(timeout=2)
    if n == 0 and hwaccel:
        yield from iter_frames(video, hwaccel=False, cancel=cancel)
    elif n == 0:
        err = b''.join(err_lines).decode(errors='replace').strip()
        raise RuntimeError('ffmpeg: ' + (err.splitlines()[-1] if err else f'exit {proc.returncode}'))


def extract_video_features(video, duration, progress=None, cancel=None):
    """Per-frame arrays (index i compares frame i with frame i-1; index 0 is zero)."""
    window = cv2.createHanningWindow((W, H), cv2.CV_32F)
    diff, motion, luma = [], [], []
    prev = None
    for i, f in enumerate(iter_frames(video, cancel=cancel)):
        g = f.astype(np.float32) * (1.0 / 255.0)
        luma.append(float(g.mean()))
        if prev is None:
            diff.append(0.0)
            motion.append(0.0)
        else:
            d_all = float(np.abs(g - prev).mean())
            # Window applied to copies: cv2.phaseCorrelate multiplies its inputs in place (OpenCV 4.13)
            (dx, dy), _ = cv2.phaseCorrelate(prev * window, g * window)
            if abs(dx) < W / 4 and abs(dy) < H / 4:
                warped = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                                        borderMode=cv2.BORDER_REPLICATE)
                mx, my = int(abs(dx)) + 2, int(abs(dy)) + 2
                resid = float(np.abs(g - warped)[my:H - my, mx:W - mx].mean())
                m = min(d_all, resid)
            else:
                m = d_all
            diff.append(d_all)
            motion.append(m)
        prev = g
        if progress and i % 300 == 0 and duration > 0:
            progress(min(1.0, i / SAMPLE_FPS / duration))
    return {'diff': np.asarray(diff, np.float32), 'motion': np.asarray(motion, np.float32),
            'luma': np.asarray(luma, np.float32)}


# ---------- Sound ----------

_YAMNET = None


def load_yamnet():
    """(onnx session, {group: [class indices]}) or None if model / onnxruntime are missing."""
    global _YAMNET
    if _YAMNET is not None:
        return _YAMNET or None
    _YAMNET = False
    model = os.path.join(_module_dir(), 'yamnet.onnx')
    classmap = os.path.join(_module_dir(), 'yamnet_class_map.csv')
    if not (os.path.isfile(model) and os.path.isfile(classmap)):
        return None
    try:
        import onnxruntime as ort
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                     if p in ort.get_available_providers()]
        sess = ort.InferenceSession(model, providers=providers)
    except Exception as e:
        print(f'YAMNet not available: {e}', flush=True)
        return None
    with open(classmap, encoding='utf-8') as f:
        names = [(row.get('display_name') or '').strip().lower() for row in csv.DictReader(f)]
    groups = {g: [i for i, n in enumerate(names) if n in terms] for g, terms in AUDIO_GROUPS.items()}
    _YAMNET = (sess, groups)
    return _YAMNET


EMBED_POOL = 2  # YAMNet frames per stored embedding (0.96 s)
EMBED_SCALE = 32.0  # embedding values (ReLU, 0..~8) stored as uint8 * 1/32


def _pool_embeddings(emb):
    """(n, 1024) float → (ceil(n / EMBED_POOL), 1024) uint8, mean over EMBED_POOL frames."""
    n = len(emb)
    pad = -n % EMBED_POOL
    if pad:
        emb = np.concatenate([emb, emb[-1:].repeat(pad, axis=0)])
    pooled = emb.reshape(-1, EMBED_POOL, emb.shape[1]).mean(axis=1)
    return np.clip(np.round(pooled * EMBED_SCALE), 0, 255).astype(np.uint8)


def extract_audio_features(video, cancel=None, embeddings=False):
    """Per YAMNet frame (0.48 s): group scores 0..1 and loudness (dBFS). None without sound/model.

    embeddings=True adds 'emb' (YAMNet's 1024-value sound fingerprint per 0.96 s, uint8, see
    EMBED_SCALE) and 'emb_t' (centre times) – raw material for a classifier trained on own labels.
    """
    ym = load_yamnet()
    if ym is None:
        return None
    sess, groups = ym
    inp = sess.get_inputs()[0].name
    emb_parts = []
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', video, '-vn', '-sn', '-dn',
           '-ac', '1', '-ar', str(AUDIO_SR), '-f', 'f32le', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=CREATE_NO_WINDOW)
    out = {g: [] for g in groups}
    loud = []
    carry = np.zeros(0, np.float32)
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise Cancelled()
            buf = proc.stdout.read(AUDIO_BLOCK * 4)
            new = np.frombuffer(buf[:len(buf) // 4 * 4], np.float32)
            w = np.concatenate([carry, new])
            eof = len(buf) < AUDIO_BLOCK * 4
            # Frame k covers w[k*HOP : k*HOP+WINDOW]; only frames that fit completely, the rest
            # carries over – the model's own end padding would shift the grid by a frame now and then
            if eof:
                if len(w) == 0:
                    break
                n = max(1, int(np.ceil(len(w) / YAMNET_HOP)))
            else:
                n = (len(w) - YAMNET_WINDOW) // YAMNET_HOP + 1
                if n < 1:
                    carry = w
                    continue
            x = w[:(n - 1) * YAMNET_HOP + YAMNET_WINDOW]
            x = np.pad(x, (0, (n - 1) * YAMNET_HOP + YAMNET_WINDOW - len(x)))
            carry = w[n * YAMNET_HOP:]
            outs = sess.run(None, {inp: x})
            scores = np.asarray(outs[0])[:n]
            if embeddings:
                emb_parts.append(np.asarray(outs[1], np.float32)[:n])
            for g, idx in groups.items():
                out[g].append(scores[:, idx].max(axis=1) if idx else np.zeros(len(scores)))
            frames = np.lib.stride_tricks.sliding_window_view(x, YAMNET_WINDOW)[::YAMNET_HOP][:n]
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            loud.extend((20 * np.log10(np.maximum(rms, 1e-5))).tolist())
            if eof:
                break
    finally:
        proc.stdout.close()
        proc.wait()
    if not loud:
        return None
    res = {g: np.concatenate(v).astype(np.float32) for g, v in out.items()}
    res['loud'] = np.asarray(loud, np.float32)
    res['t'] = (np.arange(len(loud)) * YAMNET_HOP + YAMNET_WINDOW / 2) / AUDIO_SR
    if embeddings:
        res['emb'] = _pool_embeddings(np.concatenate(emb_parts))
        res['emb_t'] = ((np.arange(len(res['emb'])) * EMBED_POOL + (EMBED_POOL - 1) / 2) * YAMNET_HOP
                        + YAMNET_WINDOW / 2).astype(np.float32) / AUDIO_SR
    return res


# ---------- Analysis (cached) ----------

def default_cache_dir():
    root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    return os.path.join(root, 'Scenecut', 'cache')


def _cache_path(video, cache_dir):
    import hashlib
    base = os.path.splitext(os.path.basename(video))[0]
    tag = hashlib.sha1(os.path.normcase(video).encode('utf-8')).hexdigest()[:8]
    return os.path.join(cache_dir, f'{base}.{tag}.features.npz')


def _side_path(video, kind, cache_dir=None):
    """Per-video file next to the analysis cache: <name>.<tag>.<kind>"""
    return _cache_path(os.path.abspath(video), cache_dir or default_cache_dir()).replace(
        '.features.npz', '.' + kind)


def _stamp(video):
    st = os.stat(video)
    return np.asarray([st.st_size, st.st_mtime, FEATURES_VERSION, SAMPLE_FPS], np.float64)


def _save_embeddings(video, a, cache_dir=None):
    path = _side_path(video, 'embed.npz', cache_dir)
    try:
        np.savez_compressed(path, stamp=_stamp(video), emb=a['emb'], emb_t=a['emb_t'],
                            scale=np.float32(EMBED_SCALE))
    except OSError as e:
        print(f'Embeddings not written: {e}', flush=True)


def load_embeddings(video, cache_dir=None, compute_missing=False, cancel=None):
    """(emb float32 (n, 1024), centre times in s) or None. compute_missing runs a sound-only
    pass for videos analyzed before embeddings were stored."""
    video = os.path.abspath(video)
    path = _side_path(video, 'embed.npz', cache_dir)
    if os.path.isfile(path):
        try:
            z = np.load(path)
            if np.array_equal(z['stamp'], _stamp(video)):
                return z['emb'].astype(np.float32) / float(z['scale']), z['emb_t']
        except Exception:
            pass
    if not compute_missing:
        return None
    a = extract_audio_features(video, cancel, embeddings=True)
    if a is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_embeddings(video, a, cache_dir)
    return a['emb'].astype(np.float32) / EMBED_SCALE, a['emb_t']


def analyze(video, cache_dir=None, use_cache=True, progress=None, cancel=None):
    """Full analysis as a dict of equally long arrays on the SAMPLE_FPS grid + meta.

    progress(share 0..1, text) is called from this thread.
    """
    video = os.path.abspath(video)
    cache_dir = cache_dir or default_cache_dir()
    cache = _cache_path(video, cache_dir)
    stamp = _stamp(video)
    if use_cache and os.path.isfile(cache):
        try:
            z = np.load(cache)
            if 'stamp' in z.files and np.array_equal(z['stamp'], stamp):
                an = {k: z[k] for k in z.files if k != 'stamp'}
                an['meta'] = json.loads(str(an['meta']))
                an['meta']['from_cache'] = True
                return an
        except Exception:
            pass

    duration, fps, has_audio = probe(video)
    t0 = time.time()
    audio = {}
    audio_err = []

    def run_audio():
        try:
            audio['res'] = extract_audio_features(video, cancel, embeddings=True)
        except Exception as e:  # sound is optional, the picture is not
            audio_err.append(e)

    t_audio = threading.Thread(target=run_audio, daemon=True)
    if has_audio:
        t_audio.start()
    rep = (lambda p: progress(p * 0.97, 'Bild und Ton werden analysiert …')) if progress else None
    an = extract_video_features(video, duration, progress=rep, cancel=cancel)
    if has_audio:
        t_audio.join()
    if cancel is not None and cancel.is_set():
        raise Cancelled()

    n = len(an['diff'])
    t_grid = np.arange(n) / SAMPLE_FPS
    a = audio.get('res')
    for k in AUDIO_KEYS:
        if a is not None:
            an[k] = np.interp(t_grid, a['t'], a[k]).astype(np.float32)
        else:
            an[k] = np.zeros(n, np.float32)
    an['meta'] = {
        'video': video, 'duration': duration, 'fps': fps, 'sample_fps': SAMPLE_FPS,
        'has_audio': a is not None, 'audio_error': str(audio_err[0]) if audio_err else '',
        'analysis_sec': round(time.time() - t0, 2),
    }
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, stamp=stamp, meta=json.dumps(an['meta']),
                            **{k: an[k] for k in VIDEO_KEYS + AUDIO_KEYS})
        if a is not None and 'emb' in a:
            _save_embeddings(video, a, cache_dir)
    except OSError as e:
        print(f'Cache not written: {e}', flush=True)
    if progress:
        progress(1.0, 'Fertig')
    return an


# ---------- Decision ----------

@dataclass
class Settings:
    motion_keep: bool = True
    motion_level: float = -1.0  # percentile of the motion curve, 0..100; < 0 = automatic
    vocal_keep: bool = True
    vocal_level: float = 30.0  # YAMNet vocal probability in %
    speech_mode: str = 'remove'  # ignore | keep | remove
    speech_level: float = 45.0
    music_keep: bool = False
    music_level: float = 50.0
    smooth_s: float = 2.0
    merge_gap_s: float = 3.0
    min_len_s: float = 4.0
    pad_s: float = 0.0
    snap_s: float = 1.5
    decision: str = 'classic'  # classic | extend: classic picks the scenes, then each grows to its action
    strictness: float = 50.0  # extend: 0 = grow far, 100 = grow little
    scene_gap_s: float = 30.0  # extend: how long the action may calm down inside one scene

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        d = dict(d or {})
        if d.get('decision') in ('score', 'blocks'):  # earlier experimental modes
            d['decision'] = 'extend'
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self):
        return asdict(self)


def detect_cuts(diff, min_diff=0.08, min_peak=3.0, min_peak_busy=1.5, k=3, min_gap_s=0.5):
    """A cut is an isolated spike: big difference to the previous frame, standing out from the
    frames on both sides – clearly from the calmer side, at least somewhat from the busier one.
    Fast motion gives high differences too, just not isolated ones, and motion that stops or
    starts abruptly only stands out from one side (pmvscan.js's 'diff vs. recent median' alone
    fires constantly on long, busy shots)."""
    diff = np.asarray(diff, np.float64)
    n = len(diff)
    if n < 2 * k + 2:
        return np.zeros(0, np.int64)
    padded = np.pad(diff, (k, k), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    left = np.median(win[:, :k], axis=1)
    right = np.median(win[:, k + 1:], axis=1)
    calm = np.maximum(np.minimum(left, right), 1e-4)
    busy = np.maximum(np.maximum(left, right), 1e-4)
    cand = np.flatnonzero((diff >= min_diff) & (diff >= min_peak * calm) & (diff >= min_peak_busy * busy))
    cuts = []
    min_gap = int(min_gap_s * SAMPLE_FPS)
    for i in cand:
        if i == 0 or i >= n - 1:
            continue
        if cuts and i - cuts[-1] < min_gap:
            if diff[i] > diff[cuts[-1]]:
                cuts[-1] = int(i)
            continue
        cuts.append(int(i))
    return np.asarray(cuts, np.int64)


def rolling_nanmedian(x, win):
    half = win // 2
    padded = np.pad(x, (half, half), mode='edge')
    view = np.lib.stride_tricks.sliding_window_view(padded, win)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN windows → NaN
        out = np.nanmedian(view, axis=1)
    return out[:len(x)]


def rolling_mean(x, win):
    if win <= 1:
        return np.asarray(x, np.float64)
    half = win // 2
    padded = np.pad(np.asarray(x, np.float64), (half, win - 1 - half), mode='edge')
    return np.convolve(padded, np.ones(win) / win, mode='valid')


def otsu_threshold(values, bins=256):
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


MOTION_BAND = 0.2  # hysteresis width on the log motion curve (±10 %)


def motion_log_curve(motion, cuts, smooth_s):
    """Log motion, cut frames masked, smoothed (median), gaps filled with the value before."""
    m = np.asarray(motion, np.float64).copy()
    for c in cuts:
        m[max(0, c - 1):c + 2] = np.nan
    m[0] = np.nan
    logm = rolling_nanmedian(np.log(np.maximum(m, 1e-4)), int(smooth_s * SAMPLE_FPS) | 1)
    for i in range(len(logm)):
        if not np.isfinite(logm[i]):
            logm[i] = logm[i - 1] if i else np.log(1e-4)
    return logm


def hysteresis(curve, level, band=4.0):
    hi, lo = level + band / 2, level - band / 2
    on = np.zeros(len(curve), bool)
    state = False
    for i, v in enumerate(curve):
        if state and v < lo:
            state = False
        elif not state and v > hi:
            state = True
        on[i] = state
    return on


def mask_to_runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def fill_gaps(mask, max_gap):
    out = np.asarray(mask, bool).copy()
    runs = mask_to_runs(out)
    for (s0, e0), (s1, _) in zip(runs, runs[1:]):
        if s1 - e0 <= max_gap:
            out[e0:s1] = True
    return out


def snap(idx, cuts, max_dist):
    if len(cuts) == 0:
        return idx
    j = int(np.argmin(np.abs(cuts - idx)))
    return int(cuts[j]) if abs(int(cuts[j]) - idx) <= max_dist else idx


def add_span(segs, a, b):
    """Union of sorted (start, end) spans with [a, b)."""
    out = []
    for s0, s1 in sorted(list(segs) + [(a, b)]):
        if out and s0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], s1))
        else:
            out.append((s0, s1))
    return out


def cut_span(segs, a, b):
    """Sorted (start, end) spans minus [a, b)."""
    out = []
    for s0, s1 in segs:
        if s1 <= a or s0 >= b:
            out.append((s0, s1))
            continue
        if s0 < a:
            out.append((s0, a))
        if s1 > b:
            out.append((b, s1))
    return out


def apply_overrides(segs, overrides, duration):
    """Manual corrections [(start, end, 'keep' | 'drop'), ...] on top of the automatic segments;
    later ones win. Pieces shorter than 0.1 s are dropped."""
    for a, b, mode in overrides or []:
        a, b = max(0.0, float(a)), min(float(duration), float(b))
        if b <= a:
            continue
        segs = add_span(segs, a, b) if mode == 'keep' else cut_span(segs, a, b)
    return [(round(a, 2), round(b, 2)) for a, b in segs if b - a >= 0.1]


def set_override(overrides, a, b, mode):
    """New correction list: mode 'keep' / 'drop' adds [a, b) (older ones give way there),
    None only removes corrections inside [a, b)."""
    out = []
    for o_a, o_b, o_m in overrides:
        if o_b <= a or o_a >= b:
            out.append((o_a, o_b, o_m))
            continue
        if o_a < a:
            out.append((o_a, a, o_m))
        if o_b > b:
            out.append((b, o_b, o_m))
    if mode is not None:
        out.append((round(a, 2), round(b, 2), mode))
    return out


def spans_minus(segs, others, min_len=0.0):
    """Parts of segs not covered by others, pieces shorter than min_len dropped."""
    out = sorted(segs)
    for a, b in others:
        out = cut_span(out, a, b)
    return [(a, b) for a, b in out if b - a >= min_len]


def edit_to_overrides(exported, final, min_len=0.3):
    """Corrections that turn the exported keep-segments into the final cut: what was removed
    becomes 'drop', what was added 'keep'. Differences below min_len are frame rounding."""
    drops = [(a, b, 'drop') for a, b in spans_minus(exported, final, min_len)]
    keeps = [(a, b, 'keep') for a, b in spans_minus(final, exported, min_len)]
    return sorted(drops + keeps)


# A video can be cut in both modes and the two cuts mean different things (classic: the
# highlights, combined: whole scenes), so labels and corrections are kept per mode:
# <name>.labels.<mode>.json, <name>.overrides.<mode>.json. The older files without a mode
# (labels.json, overrides.json) are still read: labels for the mode stored in them (none =
# combined, the first reviewed cuts were scene cuts), corrections for both modes until that
# mode gets its own.
MODES = ('classic', 'extend')


def mode_of(decision):
    return 'classic' if decision == 'classic' else 'extend'


def _overrides_path(video, cache_dir=None, decision=None):
    return _side_path(video, 'overrides.json' if decision is None else f'overrides.{mode_of(decision)}.json',
                      cache_dir)


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)


def save_export_record(video, segments, full_length, target='', decision=None, cache_dir=None):
    """What went out to a timeline (DaVinci, XML, EDL) – the reference for reading the edit back.
    decision: the mode it was made with – it tells what the edit made from it means."""
    _write_json(_side_path(video, 'export.json', cache_dir), {
        'segments': [[round(a, 3), round(b, 3)] for a, b in segments], 'full_length': bool(full_length),
        'target': target, 'decision': decision, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})


def load_export_record(video, cache_dir=None):
    d = _read_json(_side_path(video, 'export.json', cache_dir))
    if not isinstance(d, dict):
        return None
    d['segments'] = [(float(a), float(b)) for a, b in d.get('segments', [])]
    return d


def save_labels(video, final, reviewed, source, note='', decision='extend', cache_dir=None):
    """Ground truth for evaluation / training. reviewed=True: the whole video was looked at, so
    everything inside final is wanted and everything outside is not. reviewed=False: only the
    corrections (overrides) count. decision: the mode of the export that was edited – an edit of
    a classic export keeps the highlights, one of a combined export whole scenes."""
    _write_json(_side_path(video, f'labels.{mode_of(decision)}.json', cache_dir), {
        'final': [[round(a, 3), round(b, 3)] for a, b in final], 'reviewed': bool(reviewed),
        'source': source, 'note': note, 'decision': mode_of(decision), 'time': time.strftime('%Y-%m-%d %H:%M:%S')})


def _read_labels(video, decision, cache_dir=None):
    mode = mode_of(decision)
    d = _read_json(_side_path(video, f'labels.{mode}.json', cache_dir))
    if not isinstance(d, dict):
        d = _read_json(_side_path(video, 'labels.json', cache_dir))
        if not isinstance(d, dict) or mode_of(d.get('decision') or 'extend') != mode:
            return None
    d['decision'] = mode
    return d


def load_labels(video, cache_dir=None, decision='extend'):
    """The cut read back for this mode (see MODES), or None."""
    d = _read_labels(video, decision, cache_dir)
    if d is not None:
        d['final'] = [(float(a), float(b)) for a, b in d.get('final', [])]
    return d


def correct_labels(video, a, b, mode, duration, decision='extend', cache_dir=None):
    """A correction made in the app after a fully reviewed read-back also changes that mode's
    final cut (newest wins, like the overrides). Removing a correction leaves it as it is."""
    d = _read_labels(video, decision, cache_dir)
    if d is None or not d.get('reviewed') or mode not in ('keep', 'drop'):
        return False
    final = apply_overrides([tuple(x) for x in d.get('final', [])], [(a, b, mode)], duration)
    d['final'] = [[round(x, 3), round(y, 3)] for x, y in final]
    d['corrected'] = time.strftime('%Y-%m-%d %H:%M:%S')
    _write_json(_side_path(video, f'labels.{mode_of(decision)}.json', cache_dir), d)
    return True


def save_video_settings(video, s, cache_dir=None):
    """The sliders as last set for this one video."""
    _write_json(_side_path(video, 'settings.json', cache_dir), s.to_dict())


def load_video_settings(video, cache_dir=None):
    d = _read_json(_side_path(video, 'settings.json', cache_dir))
    return Settings.from_dict(d) if isinstance(d, dict) else None


def load_overrides(video, cache_dir=None, decision='extend'):
    """Manual corrections of this mode (the older file without a mode until it has its own)."""
    for path in (_overrides_path(video, cache_dir, decision), _overrides_path(video, cache_dir)):
        try:
            with open(path, encoding='utf-8') as f:
                return [(float(a), float(b), str(m)) for a, b, m in json.load(f)]
        except OSError:
            continue
        except (ValueError, TypeError):
            return []
    return []


def save_overrides(video, overrides, cache_dir=None, decision='extend'):
    path = _overrides_path(video, cache_dir, decision)
    if not overrides and not os.path.isfile(_overrides_path(video, cache_dir)):
        try:
            os.remove(path)
        except OSError:
            pass
        return
    # Empty but an old file without a mode exists: "[]" keeps it from coming back for this mode
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([[round(a, 2), round(b, 2), m] for a, b, m in overrides], f)


# Combined mode ("kombiniert"): the classic decision picks WHICH scenes (it is good at that, but
# keeps only their busiest seconds), then every picked piece grows to the action around it –
# the stretch where the motion stays rhythmic (see motion_rhythm) or vocal sounds go on. Talk
# in the middle of that stretch does not end it. Picks without rhythm around them are dropped.
# Measured on the first fully reviewed video: classic Highlights touched 14 of 20 wanted scenes
# but covered only 2–48 % of each.
EXTEND_SMOOTH_S = 30.0  # vocal sounds are averaged over this long
EXTEND_VOCAL = 0.4  # vocal sounds above this share of vocal_level keep a scene going

# Rhythm: how strongly the motion repeats itself 0.5–5 times a second (max. autocorrelation of
# the detrended log motion in 8 s windows, averaged over 60 s). A correlation does not depend on
# picture size, brightness or how busy a production is, so unlike the motion level it means the
# same in every video: in both reviewed films wanted scenes sit at ≈ 0.27–0.29, the rest at
# ≈ 0.22–0.23 (AUC 0.77 / 0.78). The motion level only separated per video, which made the
# combined mode keep ~45 % of every short clip, whatever was in it.
RHYTHM_WIN_S = 8.0
RHYTHM_LAGS = (2, 20)  # frames at SAMPLE_FPS: 0.2–2 s per repetition
RHYTHM_DETREND = 5  # frames; removes the slow ups and downs, keeps the beat
RHYTHM_SMOOTH_S = 60.0
RHYTHM_PICK_MARGIN = 0.01  # a pick needs at least threshold + this much rhythm to count
# A long, clearly rhythmic stretch is a scene even without a pick: the classic pick follows the
# motion level of the video itself, so a calmer scene in a busy film can get none (one reviewed
# video: 13 min at rhythm 0.33 without a single Highlights pick). Measured on 6 reviewed videos
# (Highlights, strictness 50): 76.0 → 79.6 % decided right, no video worse, story-only short
# clips unchanged.
RHYTHM_SEED_MARGIN = 0.05
RHYTHM_SEED_S = 120.0
RHYTHM_SCALE = (0.1, 0.5)  # rhythm shown as 0..100 between these values (same for every video)


def motion_rhythm(motion):
    """Rhythm per frame on the SAMPLE_FPS grid (0 = irregular, up to ~0.6 = clear beat)."""
    x = np.log(np.maximum(np.asarray(motion, np.float64), 1e-4))
    x = x - rolling_mean(x, RHYTHM_DETREND)
    n, w, hop = len(x), int(RHYTHM_WIN_S * SAMPLE_FPS), 5
    if n <= w:
        return np.zeros(n)
    starts = np.arange(0, n - w, hop)
    segs = np.lib.stride_tricks.sliding_window_view(x, w)[starts]
    segs = segs - segs.mean(axis=1, keepdims=True)
    energy = (segs ** 2).sum(axis=1) + 1e-12
    best = np.full(len(starts), -1.0)
    for lag in range(RHYTHM_LAGS[0], RHYTHM_LAGS[1] + 1):
        best = np.maximum(best, (segs[:, :-lag] * segs[:, lag:]).sum(axis=1) / energy)
    per_frame = np.interp(np.arange(n), starts + w // 2, best)
    return rolling_mean(per_frame, int(RHYTHM_SMOOTH_S * SAMPLE_FPS))


def extend_level(strictness):
    """Strictness 0..100 → rhythm a scene needs to keep growing: 0.20 (grow far) .. 0.28 (little)."""
    return 0.20 + min(max(float(strictness), 0.0), 100.0) / 100.0 * 0.08


def rhythm_display(r):
    lo, hi = RHYTHM_SCALE
    return np.clip((np.asarray(r, np.float64) - lo) / (hi - lo) * 100.0, 0, 100)


def grow_to_regions(seed, region):
    """seed plus every run of region that touches a seed frame."""
    out = np.asarray(seed, bool).copy()
    for a, b in mask_to_runs(region):
        if out[a:b].any():
            out[a:b] = True
    return out


def _to_segments(keep, cuts, s, duration):
    """Runs of keep → (start, end) seconds: edges snapped to cuts, min length, padding."""
    snap_d = int(s.snap_s * SAMPLE_FPS)
    segs = []
    for a, b in mask_to_runs(keep):
        a, b = snap(a, cuts, snap_d), snap(b, cuts, snap_d)
        if b - a < s.min_len_s * SAMPLE_FPS:
            continue
        t0 = max(0.0, a / SAMPLE_FPS - s.pad_s)
        t1 = min(duration, b / SAMPLE_FPS + s.pad_s)
        if segs and t0 <= segs[-1][1]:
            segs[-1] = (segs[-1][0], max(segs[-1][1], t1))
        else:
            segs.append((t0, t1))
    return [(round(a, 2), round(b, 2)) for a, b in segs]


def decide(an, s, overrides=None):
    """Keep-segments (seconds) for one analysis and one Settings object, plus manual corrections
    (see apply_overrides) that always win over the automatic decision.

    Returns dict: segments, curves (0..100 on the SAMPLE_FPS grid), masks, cuts, auto_motion_level, stats.
    """
    n = len(an['diff'])
    duration = float(an['meta']['duration']) or n / SAMPLE_FPS
    cuts = detect_cuts(an['diff'])
    win = max(1, int(s.smooth_s * SAMPLE_FPS))

    # Motion level = percentile ("the busiest 100-level % of this video"), compared on the log
    # curve itself: a band in percentile units would be razor thin where most frames sit
    logm = motion_log_curve(an['motion'], cuts, s.smooth_s)
    otsu = otsu_threshold(logm) if n else 0.0
    auto_level = float(np.mean(logm < otsu) * 100.0) if n else 50.0
    level = auto_level if s.motion_level < 0 else float(s.motion_level)
    thr = otsu if s.motion_level < 0 else float(np.quantile(logm, min(max(level, 0.0), 100.0) / 100.0))
    # Still pictures (title cards, black) never count as motion, whatever the percentile says
    still = rolling_mean(an['motion'], win) < 0.002
    # Display scale 0..100 (1st..99th percentile of this video)
    lo_v, hi_v = (np.quantile(logm, [0.01, 0.99]) if n else (0.0, 1.0))
    scale = 100.0 / max(hi_v - lo_v, 1e-6)
    curves = {
        'motion': np.clip((logm - lo_v) * scale, 0, 100),
        'vocal': rolling_mean(an['vocal'], win) * 100.0,
        'speech': rolling_mean(an['speech'], win) * 100.0,
        'music': rolling_mean(an['music'], win) * 100.0,
    }
    masks = {
        'motion': hysteresis(logm, thr, band=MOTION_BAND) & ~still,
        'vocal': hysteresis(curves['vocal'], s.vocal_level),
        'speech': hysteresis(curves['speech'], s.speech_level),
        'music': hysteresis(curves['music'], s.music_level),
    }
    keep = np.zeros(n, bool)
    if s.motion_keep:
        keep |= masks['motion']
    if s.vocal_keep:
        keep |= masks['vocal']
    if s.music_keep:
        keep |= masks['music']
    if s.speech_mode == 'keep':
        keep |= masks['speech']
    keep = fill_gaps(keep, int(s.merge_gap_s * SAMPLE_FPS))
    if s.speech_mode == 'remove':
        keep &= ~masks['speech']
    segs = _to_segments(keep, cuts, s, duration)

    region_line = None
    if s.decision == 'extend' and n:
        # Seeds: the finished classic pieces (min length applied, so no blip grows a scene)
        # Rhythm depends only on the motion: computed once per analysis, not per slider move
        if 'rhythm' not in an or len(an['rhythm']) != n:
            an['rhythm'] = motion_rhythm(an['motion'])
        rhythm = an['rhythm']
        rthr = extend_level(s.strictness)
        seed = np.zeros(n, bool)
        for a, b in segs:
            a, b = int(round(a * SAMPLE_FPS)), int(round(b * SAMPLE_FPS))
            if b > a and rhythm[a:b].mean() >= rthr + RHYTHM_PICK_MARGIN:
                seed[a:b] = True
        for a, b in mask_to_runs(rhythm > rthr + RHYTHM_SEED_MARGIN):
            if b - a >= RHYTHM_SEED_S * SAMPLE_FPS:
                seed[a:b] = True
        region = rhythm > rthr
        if an['meta'].get('has_audio') and s.vocal_keep:
            vocal_ctx = rolling_mean(an['vocal'], int(EXTEND_SMOOTH_S * SAMPLE_FPS)) * 100.0
            region |= vocal_ctx > s.vocal_level * EXTEND_VOCAL
        region = fill_gaps(region, int(s.scene_gap_s * SAMPLE_FPS))
        keep = grow_to_regions(seed, region)
        segs = _to_segments(keep, cuts, s, duration)
        curves['region'] = rhythm_display(rhythm)
        region_line = float(rhythm_display(rthr))

    if overrides:
        segs = apply_overrides(segs, overrides, duration)
    kept = sum(b - a for a, b in segs)
    return {
        'segments': segs, 'curves': curves, 'masks': masks, 'keep': keep, 'cuts': cuts,
        'auto_motion_level': auto_level, 'motion_level': level, 'overrides': list(overrides or []),
        'motion_line': float(np.clip((thr - lo_v) * scale, 0, 100)),  # threshold on the display scale
        'region_line': region_line,  # combined mode: where the rhythm still counts as action
        'stats': {'kept_sec': kept, 'duration': duration, 'share': kept / max(duration, 1e-6),
                  'segments': len(segs), 'cuts': int(len(cuts))},
    }


def full_length_parts(segments, duration):
    """The whole video as (start, end, keep) parts, cut at every segment edge."""
    parts = []
    t = 0.0
    for a, b in segments:
        if a > t + 1e-3:
            parts.append((t, a, False))
        parts.append((a, b, True))
        t = b
    if duration > t + 1e-3:
        parts.append((t, duration, False))
    return parts


# ---------- Export ----------

def export_davinci(video, segs, fps, out_dir, kind='xml', full_length=False, duration=None):
    """kind: xml (FCP7, DaVinci File → Import → Timeline) or edl. Returns the written path.

    full_length: the whole video, cut at every segment edge, clips at their source time.
    """
    from timeline_export import export_edl_cmx, export_xml_xmeml
    base = os.path.splitext(os.path.basename(video))[0]
    suffix = '_scenecut_full' if full_length else '_scenecut'
    if full_length:
        segs = [(a, b) for a, b, _keep in full_length_parts(segs, duration if duration else probe(video)[0])]
    if kind == 'edl':
        export_edl_cmx(video, segs, fps, out_dir, at_source_time=full_length)
        return os.path.join(out_dir, f'{base}{suffix}.edl')
    export_xml_xmeml(video, segs, fps, out_dir, at_source_time=full_length)
    return os.path.join(out_dir, f'{base}{suffix}.xml')


def _nvenc_ok():
    r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=256x256:d=0.1',
                        '-c:v', 'hevc_nvenc', '-f', 'null', '-'],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    return r.returncode == 0


def export_mp4(video, segs, out_dir, progress=None, cancel=None):
    """Re-encode every segment frame-accurately (NVENC HEVC, otherwise x264), then join them."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    final = os.path.join(out_dir, f'{base}_scenecut.mp4')
    venc = ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', '23'] if _nvenc_ok() else \
        ['-c:v', 'libx264', '-preset', 'medium', '-crf', '20']
    tmp_dir = os.path.join(out_dir, f'.{base}_parts')
    os.makedirs(tmp_dir, exist_ok=True)
    parts = []
    total = sum(b - a for a, b in segs) or 1.0
    done = 0.0
    try:
        for i, (a, b) in enumerate(segs):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            part = os.path.join(tmp_dir, f'{i:04d}.mp4')
            r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{a:.3f}', '-i', video,
                                '-t', f'{b - a:.3f}', '-map', '0:v:0', '-map', '0:a:0?', *venc,
                                '-c:a', 'aac', '-b:a', '192k', part],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
            if r.returncode != 0:
                msg = r.stderr.decode(errors='replace').strip().splitlines()
                raise RuntimeError(f'Segment {i + 1}: ' + (msg[-1] if msg else 'ffmpeg error'))
            parts.append(part)
            done += b - a
            if progress:
                progress(done / total * 0.97, f'Segment {i + 1}/{len(segs)} gerendert')
        lst = os.path.join(tmp_dir, 'list.txt')
        with open(lst, 'w', encoding='utf-8') as f:
            for p in parts:
                f.write("file '" + p.replace('\\', '/').replace("'", "'\\''") + "'\n")
        r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                            '-c', 'copy', '-movflags', '+faststart', final],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError('Zusammenfügen fehlgeschlagen: ' + r.stderr.decode(errors='replace')[-300:])
    finally:
        for p in os.listdir(tmp_dir) if os.path.isdir(tmp_dir) else []:
            try:
                os.remove(os.path.join(tmp_dir, p))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    if progress:
        progress(1.0, 'Fertig')
    return final


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('video')
    ap.add_argument('--out', default=None, help='output folder (default: next to the video)')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--motion-level', type=float, default=-1.0,
                    help='motion percentile 0..100 to count as action (default: automatic)')
    ap.add_argument('--no-vocal', action='store_true')
    ap.add_argument('--speech', choices=('ignore', 'keep', 'remove'), default='remove')
    ap.add_argument('--decision', choices=('classic', 'extend'), default='classic',
                    help='extend: classic picks the scenes, each grows to the action around it')
    ap.add_argument('--strictness', type=float, default=50.0, help='extend: 0 = grow far .. 100 = grow little')
    ap.add_argument('--export', default='', help='comma list of xml, edl, mp4')
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    out_dir = args.out or os.path.dirname(video)
    t0 = time.time()
    an = analyze(video, use_cache=not args.no_cache,
                 progress=lambda p, _t: print(f'PROGRESS:{int(p * 100)}', flush=True))
    s = Settings(motion_level=args.motion_level, vocal_keep=not args.no_vocal, speech_mode=args.speech,
                 decision=args.decision, strictness=args.strictness)
    res = decide(an, s)
    st = res['stats']
    print(f"Analysis {an['meta']['analysis_sec']} s (this run {time.time() - t0:.1f} s) | sound: "
          f"{'yes' if an['meta']['has_audio'] else 'no'} | cuts {st['cuts']} | segments {st['segments']} | "
          f"kept {st['kept_sec'] / 60:.1f} of {st['duration'] / 60:.1f} min ({st['share'] * 100:.0f} %) | "
          f"motion level {res['motion_level']:.0f} (auto {res['auto_motion_level']:.0f})", flush=True)
    for kind in [k.strip() for k in args.export.split(',') if k.strip()]:
        if kind == 'mp4':
            path = export_mp4(video, res['segments'], out_dir,
                              progress=lambda p, _t: print(f'PROGRESS:{int(p * 100)}', flush=True))
        else:
            path = export_davinci(video, res['segments'], an['meta']['fps'], out_dir, kind)
        print(f'Exported: {path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
