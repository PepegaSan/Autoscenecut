#!/usr/bin/env python3
"""Import SceneCut XML into running DaVinci Resolve and verify timeline clips."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'Davinci API start'))

from unittest.mock import MagicMock

sys.modules.setdefault('analyzer_nvidia', MagicMock())
sys.modules.setdefault('algo_snapshot', MagicMock())

import autocut_nvidia as ac  # noqa: E402
from davinci_api import connect_resolve  # noqa: E402


def _timeline_report(timeline, label: str, fps: float) -> dict:
    if timeline is None:
        raise RuntimeError(f'{label}: ImportTimelineFromFile returned None')
    name = timeline.GetName()
    start = int(timeline.GetStartFrame() or 0)
    end = int(timeline.GetEndFrame() or 0)
    dur_frames = max(0, end - start)
    items = timeline.GetItemListInTrack('video', 1) if hasattr(timeline, 'GetItemListInTrack') else []
    if not items:
        items = timeline.GetItemsInTrack('video', 1) or []
    clips = []
    for it in items:
        if it is None:
            continue
        cs = int(it.GetStart() or 0)
        cd = int(it.GetDuration() or 0)
        ce = cs + cd
        clips.append({'start': cs, 'end': ce, 'duration': cd})
    clips.sort(key=lambda c: c['start'])
    return {
        'label': label,
        'name': name,
        'timeline_start': start,
        'timeline_end': end,
        'duration_frames': dur_frames,
        'duration_sec': dur_frames / fps if fps > 0 else 0,
        'clip_count': len(clips),
        'clips': clips[:8],
        'clips_truncated': len(clips) > 8,
    }


def _import_xml(media_pool, xml_path: Path, timeline_name: str):
    path = str(xml_path.resolve())
    opts = {'timelineName': timeline_name, 'importSourceClips': True, 'importMultiChannelAudio': True}
    tl = media_pool.ImportTimelineFromFile(path, opts)
    if tl is None:
        tl = media_pool.ImportTimelineFromFile(path)
    return tl


def main() -> int:
    cfg = ac.load_cfg()
    api_path = cfg.get('Settings', 'resolve_api_path', fallback='').strip()
    if not api_path:
        print('SKIP: resolve_api_path fehlt in config_nvidia.ini')
        return 2

    work = _ROOT / 'output' / '_davinci_import_test'
    work.mkdir(parents=True, exist_ok=True)

    # 1) Synthetic short clip + cut-points XML (from prior export test)
    sample = _ROOT / 'output' / '_export_test_run' / 'test_run_export_only_modes' / 'sample.avi'
    if not sample.is_file():
        print('Erzeuge kurzes Testvideo …')
        import cv2
        import numpy as np

        sample.parent.mkdir(parents=True, exist_ok=True)
        w, h, fps, frames = 320, 240, 25.0, 250
        out = cv2.VideoWriter(str(sample), cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        for i in range(frames):
            fr = np.zeros((h, w, 3), dtype=np.uint8)
            import cv2 as cv

            cv.rectangle(fr, (10, 10), (w - 10, h - 10), ((i * 3) % 255, 64, 200), -1)
            out.write(fr)
        out.release()

    fps = 25.0
    seg_res = [
        {'start': 0, 'end': 2, 'final_category': 'action'},
        {'start': 2, 'end': 4, 'final_category': 'action'},
        {'start': 4, 'end': 6, 'final_category': 'story'},
        {'start': 6, 'end': 8, 'final_category': 'vocal'},
        {'start': 8, 'end': 10, 'final_category': 'vocal'},
    ]
    merged = [(0.0, 4.0), (6.0, 8.0)]
    xml_full = work / 'sample_scenecut_full.xml'
    xml_short = work / 'sample_scenecut.xml'
    ac.export_xml_xmeml(str(sample), ac.segments_full_timeline_analysis(seg_res, 10.0), fps, str(work), at_source_time=True)
    ac.export_xml_xmeml(str(sample), merged, fps, str(work), at_source_time=False)
    if not xml_full.is_file() or not xml_short.is_file():
        print('FAIL: XML-Export fehlgeschlagen')
        return 1

    cases = [
        (xml_full, 'Scenecut_TEST_full', 5, 250, 'cut points (volle Länge)'),
        (xml_short, 'Scenecut_TEST_autocut', 2, None, 'Autocut (verkürzt)'),
    ]

    # 2) Optional: real checkpoint video
    ck_path = _ROOT / 'output' / 'last_autocut_checkpoint.json'
    if ck_path.is_file():
        ck = json.loads(ck_path.read_text(encoding='utf-8'))
        vid = (ck.get('video_path') or '').strip()
        if vid and os.path.isfile(vid):
            ck_fps = float(ck.get('fps') or 25.0)
            seg_results = ck.get('segment_results') or []
            dur = ac._video_duration_sec(vid, ck_fps)
            full_segs = ac.segments_full_timeline_analysis(seg_results, dur)
            real_dir = work / 'real'
            real_dir.mkdir(exist_ok=True)
            if ac.export_xml_xmeml(
                vid, full_segs, ck_fps, str(real_dir), at_source_time=True
            ):
                real_xml = real_dir / (Path(vid).stem + '_scenecut_full.xml')
                if real_xml.is_file():
                    cases.append(
                        (
                            real_xml,
                            'Scenecut_TEST_real_full',
                            len(full_segs),
                            int(dur * ck_fps),
                            f'Checkpoint-Video ({Path(vid).name})',
                        )
                    )
                    fps_real = ck_fps
        else:
            print(f'HINWEIS: Checkpoint-Video nicht gefunden: {vid}')
    else:
        fps_real = fps

    print('Verbinde mit DaVinci Resolve (External scripting = Local) …')
    try:
        resolve, project, media_pool, _root = connect_resolve(
            create_scratch_project_name=None,
            auto_launch=False,
            status_callback=lambda m: print(f'  {m}'),
        )
    except Exception as e:
        print(f'FAIL: Resolve-Verbindung: {e}')
        print('Prüfe: Preferences → System → General → External scripting using = Local, Resolve neu starten.')
        return 1

    project_name = project.GetName()
    print(f'Projekt: {project_name}')

    failed = 0
    for xml_path, tl_name, expect_clips, expect_frames, desc in cases:
        use_fps = fps_real if 'real' in tl_name.lower() else fps
        print(f'\n--- Import: {desc} ---')
        print(f'  XML: {xml_path}')
        try:
            timeline = _import_xml(media_pool, xml_path, tl_name)
            project.SetCurrentTimeline(timeline)
            rep = _timeline_report(timeline, tl_name, use_fps)
            print(f"  Timeline: {rep['name']}")
            print(f"  Clips: {rep['clip_count']} (erwartet >= {expect_clips})")
            print(f"  Dauer: {rep['duration_frames']} Frames (~{rep['duration_sec']:.1f}s)")
            if rep['clips']:
                print(f"  Erste Clips (Timeline-Frames): {rep['clips'][:3]}")
            if rep['clip_count'] < expect_clips:
                print('  WARN: weniger Clips als erwartet')
                failed += 1
            if expect_frames is not None:
                tol = max(5, int(expect_frames * 0.02))
                if abs(rep['duration_frames'] - expect_frames) > tol:
                    print(f'  WARN: Dauer abweichend (erwartet ~{expect_frames} ±{tol})')
                    failed += 1
                else:
                    print('  OK: Timeline-Dauer passt')
            time.sleep(0.5)
        except Exception as e:
            print(f'  FAIL: {e}')
            failed += 1

    print(f'\n{"FEHLER" if failed else "ERFOLG"}: {len(cases) - failed}/{len(cases)} Imports ok')
    print('In Resolve: Timelines „Scenecut_TEST_*“ prüfen (Edit-Seite).')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
