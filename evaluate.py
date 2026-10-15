"""Measure the automatic decision against your own corrections.

Every video in the analysis cache that has corrections (overrides) or labels read back from
DaVinci is decided again WITHOUT the corrections, and compared with them:

- wanted:   kept by you for sure (labels of a fully reviewed video, or 'always keep')
- accepted: kept automatically and not corrected – probably fine, but nobody can be sure
            (only for videos that were not marked as fully reviewed)
- unwanted: removed by you ('always cut', or outside the reviewed final cut)

Usage:
    python evaluate.py                      # classic vs. combined with the app's current settings
    python evaluate.py --sweep              # combined mode at several strictness levels
    python evaluate.py --preset Highlights  # start from an app preset instead of the saved settings
    python evaluate.py --settings other.json
    python evaluate.py --kurzclips          # short clips cut from the fully reviewed videos

--kurzclips: there are no reviewed short clips yet, so they are made from the reviewed films:
10 and 20 min excerpts (half overlapping) plus compilations of only the wanted (all action) or
only the unwanted parts (all story). The signals are real, the answer is known. Judged by how
much of each clip is decided right, grouped by how much of it is wanted.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import fast_scan as fs


def load_cached(npz):
    z = np.load(npz)
    an = {k: z[k] for k in z.files if k != 'stamp'}
    an['meta'] = json.loads(str(an['meta']))
    return an


def mask_from_spans(spans, n):
    m = np.zeros(n, bool)
    for a, b in spans:
        m[int(round(a * fs.SAMPLE_FPS)):int(round(b * fs.SAMPLE_FPS))] = True
    return m


def ground_truth(an, video, s, cache_dir=None):
    """(wanted, accepted, unwanted) frame masks and whether the video was fully reviewed."""
    n = len(an['diff'])
    labels = fs.load_labels(video, cache_dir)
    overrides = fs.load_overrides(video, cache_dir)
    if labels and labels.get('reviewed'):
        wanted = mask_from_spans(labels['final'], n)
        return wanted, np.zeros(n, bool), ~wanted, True
    keep_o = mask_from_spans([(a, b) for a, b, m in overrides if m == 'keep'], n)
    drop_o = mask_from_spans([(a, b) for a, b, m in overrides if m == 'drop'], n)
    wanted = keep_o & ~drop_o
    unwanted = drop_o & ~keep_o
    # Not fully reviewed (also a DaVinci read-back without review): its changes are in overrides
    classic = fs.Settings.from_dict({**s.to_dict(), 'decision': 'classic'})
    auto = mask_from_spans(fs.decide(an, classic)['segments'], n)
    accepted = auto & ~wanted & ~unwanted
    return wanted, accepted, unwanted, False


def evaluate(an, s, gt):
    wanted, accepted, unwanted, _rev = gt
    n = len(an['diff'])
    kept = mask_from_spans(fs.decide(an, s)['segments'], n)
    labeled = wanted | accepted | unwanted

    def share(m):
        return float(kept[m].mean() * 100) if m.any() else float('nan')
    return {
        'wanted': share(wanted), 'accepted': share(accepted),
        'unwanted_removed': 100.0 - share(unwanted) if unwanted.any() else float('nan'),
        'kept_min': kept.sum() / fs.SAMPLE_FPS / 60, 'new_min': (kept & ~labeled).sum() / fs.SAMPLE_FPS / 60,
        'sec': {k: m.sum() / fs.SAMPLE_FPS for k, m in (('wanted', wanted), ('accepted', accepted),
                                                          ('unwanted', unwanted))},
    }


SHORT_MINUTES = (10, 20)
SHORT_BUCKETS = (('fast keine Action (0–20 % gewollt)', 0.0, 0.2), ('gemischt (20–80 %)', 0.2, 0.8),
                 ('fast nur Action (80–100 %)', 0.8, 1.01))


def _sub_analysis(an, idx):
    """Analysis made of the frames idx (sorted); where pieces are joined there is a cut."""
    sub = {k: an[k][idx].copy() for k in fs.VIDEO_KEYS + fs.AUDIO_KEYS}
    joins = np.flatnonzero(np.diff(idx) > 1) + 1
    sub['diff'][joins] = np.maximum(sub['diff'][joins], 0.3)
    sub['meta'] = {**an['meta'], 'duration': len(idx) / fs.SAMPLE_FPS}
    return sub


def short_clips(videos):
    """(name, analysis, wanted mask) for every made-up short clip of the fully reviewed videos."""
    out = []
    for name, an, (wanted, _a, _u, reviewed) in videos:
        if not reviewed:
            continue
        n = len(wanted)
        for mins in SHORT_MINUTES:
            ln = int(mins * 60 * fs.SAMPLE_FPS)
            for s in range(0, n - ln + 1, ln // 2):
                idx = np.arange(s, s + ln)
                out.append((f'{name[:12]} {mins} min ab {s / fs.SAMPLE_FPS / 60:.0f}', _sub_analysis(an, idx), wanted[idx]))
            for tag, sel in (('nur Action', wanted), ('nur Story', ~wanted)):
                pool = np.flatnonzero(sel)
                for k in range(0, len(pool) - ln + 1, ln):
                    idx = pool[k:k + ln]
                    out.append((f'{name[:12]} {tag} {mins} min #{k // ln + 1}', _sub_analysis(an, idx), wanted[idx]))
    return out


def report_short(clips, configs, verbose=False):
    print(f'{len(clips)} Kurzclips aus den komplett geprüften Videos. Spalten je Gruppe: '
          'behalten (Ø) | richtig entschieden (Ø)\n')
    print(' ' * 16 + ''.join(f'{b[0]:>38s}' for b in SHORT_BUCKETS))
    for cname, s in configs:
        rows = []
        for name, an, wanted in clips:
            kept = mask_from_spans(fs.decide(an, s)['segments'], len(wanted))
            rows.append((name, wanted.mean(), kept.mean(), (kept == wanted).mean()))
            if verbose:
                print(f'    {name:34s} gewollt {rows[-1][1] * 100:3.0f} %  behalten {rows[-1][2] * 100:3.0f} %  '
                      f'richtig {rows[-1][3] * 100:3.0f} %')
        cells = []
        for _b, lo, hi in SHORT_BUCKETS:
            sel = [r for r in rows if lo <= r[1] < hi]
            cells.append(f'{np.mean([r[2] for r in sel]) * 100:5.0f} % | {np.mean([r[3] for r in sel]) * 100:3.0f} % (n={len(sel):2d})'
                         if sel else '–')
        print(f'  {cname:14s}' + ''.join(f'{c:>38s}' for c in cells))
    return 0


def fmt(v):
    return '   –  ' if v != v else f'{v:5.0f}%'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--settings', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       'scenecut_settings.json'))
    ap.add_argument('--cache', default=None)
    ap.add_argument('--sweep', action='store_true', help='combined mode at strictness 20 … 80')
    ap.add_argument('--preset', default=None, help='app preset as the base (Highlights, Standard, Großzügig)')
    ap.add_argument('--kurzclips', action='store_true', help='short clips cut from the fully reviewed videos')
    ap.add_argument('--einzeln', action='store_true', help='with --kurzclips: one line per clip')
    args = ap.parse_args()

    try:
        with open(args.settings, encoding='utf-8') as f:
            base = fs.Settings.from_dict(json.load(f).get('settings'))
    except (OSError, ValueError):
        base = fs.Settings()
    cache = args.cache or fs.default_cache_dir()

    videos = []
    for npz in sorted(glob.glob(os.path.join(cache, '*.features.npz'))):
        has_side = any(os.path.isfile(npz.replace('.features.npz', '.' + k)) for k in ('overrides.json', 'labels.json'))
        if not has_side:
            continue
        an = load_cached(npz)
        video = an['meta']['video']
        videos.append((os.path.basename(video), an, ground_truth(an, video, base, cache)))
    if not videos:
        print('Keine Videos mit Korrekturen im Cache:', cache)
        return 1

    if args.preset:
        from scenecut_app import PRESETS
        base = fs.Settings.from_dict({**fs.Settings().to_dict(), **PRESETS[args.preset][0]})
    configs = [('klassisch', fs.Settings.from_dict({**base.to_dict(), 'decision': 'classic'}))]
    levels = (20, 30, 40, 50, 60, 70, 80) if args.sweep else (base.strictness,)
    for lv in levels:
        configs.append((f'kombiniert {lv:.0f}', fs.Settings.from_dict({**base.to_dict(), 'decision': 'extend',
                                                                          'strictness': lv})))

    print(f'Einstellungen: {args.preset or args.settings}')
    if args.kurzclips:
        return report_short(short_clips(videos), configs, args.einzeln)
    print('Spalten: gewollt behalten | akzeptiert behalten | ungewollt entfernt | behalten gesamt | '
          'davon ungeprüft neu\n')
    totals = {name: {'w': [0, 0], 'a': [0, 0], 'u': [0, 0]} for name, _ in configs}
    for name, an, gt in videos:
        wanted, accepted, unwanted, rev = gt
        print(f'{name[:60]}  ({"komplett geprüft" if rev else "nur Korrekturen"}; gewollt '
              f'{wanted.sum() / 600:.1f} min, akzeptiert {accepted.sum() / 600:.1f} min, '
              f'ungewollt {unwanted.sum() / 600:.1f} min)')
        for cname, s in configs:
            r = evaluate(an, s, gt)
            print(f'  {cname:14s} {fmt(r["wanted"])} {fmt(r["accepted"])} {fmt(r["unwanted_removed"])}'
                  f'   {r["kept_min"]:5.1f} min  {r["new_min"]:5.1f} min')
            t = totals[cname]
            for key, rk, m in (('w', 'wanted', wanted), ('a', 'accepted', accepted), ('u', 'unwanted_removed', unwanted)):
                if m.any():
                    t[key][0] += r[rk] / 100 * m.sum()
                    t[key][1] += m.sum()
        print()
    print('Gesamt (zeitgewichtet):')
    for cname, _s in configs:
        t = totals[cname]
        vals = [t[k][0] / t[k][1] * 100 if t[k][1] else float('nan') for k in ('w', 'a', 'u')]
        print(f'  {cname:14s} ' + ' '.join(fmt(v) for v in vals))
    return 0


if __name__ == '__main__':
    sys.exit(main())
