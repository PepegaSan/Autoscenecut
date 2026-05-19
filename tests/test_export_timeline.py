#!/usr/bin/env python3
"""Smoke tests for XML/EDL export (autocut vs cut-points vs no-cuts). No Whisper/GPU required."""

from __future__ import annotations

import configparser
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Avoid heavy ML imports when loading autocut_nvidia
sys.modules.setdefault('analyzer_nvidia', MagicMock())
sys.modules.setdefault('algo_snapshot', MagicMock())

import autocut_nvidia as ac  # noqa: E402


def _make_test_video(path: Path, *, fps: float = 25.0, frames: int = 250) -> None:
    """~10 s @ 25 fps synthetic clip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    # .avi + XVID is reliable on Windows without extra codecs
    if path.suffix.lower() != '.avi':
        path = path.with_suffix('.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(path.resolve()), fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f'VideoWriter failed: {path}')
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), ((i * 3) % 255, 64, 200), -1)
        out.write(frame)
    out.release()
    return path


def _parse_xml_clips(xml_path: Path) -> list[tuple[int, int, int, int]]:
    """Return (timeline_start, timeline_end, in_frame, out_frame) per clipitem."""
    root = ET.parse(xml_path).getroot()
    clips = []
    for ci in root.iter('clipitem'):
        clips.append(
            (
                int(ci.findtext('start', '0')),
                int(ci.findtext('end', '0')),
                int(ci.findtext('in', '0')),
                int(ci.findtext('out', '0')),
            )
        )
    return clips


def _parse_edl_events(edl_path: Path) -> list[tuple[str, str, str, str]]:
    events = []
    for line in edl_path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^\d{3}\s+AX\s+V\s+C\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if m:
            events.append(m.groups())
    return events


def test_ntsc_timeline_vs_media_frames():
    fps = 29.97002997002997
    # 5:00 on Resolve timeline (timebase 30 NTSC) vs true media frame at 300s
    assert ac._edit_frames_from_sec(300, 30, 'TRUE') == 9000
    assert ac._media_frames_from_sec(300, fps) == 8991
    assert ac._edit_frames_from_sec(20, 30, 'TRUE') == 600
    assert ac._media_frames_from_sec(20, fps) == 599


def test_segment_helpers():
    sr = [{'start': 0, 'end': 20}, {'start': 20, 'end': 40}, {'start': 40, 'end': 60}]
    segs = ac.segments_full_timeline_analysis(sr, 60.0)
    assert segs == [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)], segs

    merged = [(0.0, 30.0), (50.0, 60.0)]
    fallback = ac.segments_full_timeline_from_merged(merged, 60.0)
    assert fallback == [(0.0, 30.0), (30.0, 50.0), (50.0, 60.0)], fallback


def test_xml_autocut_vs_full(tmp_path: Path):
    vid = _make_test_video(tmp_path / 'sample')
    fps = 25.0
    merged = [(0.0, 4.0), (6.0, 10.0)]

    assert ac.export_xml_xmeml(str(vid), merged, fps, str(tmp_path), at_source_time=False)
    assert ac.export_xml_xmeml(str(vid), merged, fps, str(tmp_path), at_source_time=True)

    base = vid.stem
    short_xml = tmp_path / f'{base}_scenecut.xml'
    full_xml = tmp_path / f'{base}_scenecut_full.xml'
    assert short_xml.is_file() and full_xml.is_file()

    short_clips = _parse_xml_clips(short_xml)
    full_sr = [{'start': 0, 'end': 2}, {'start': 2, 'end': 4}, {'start': 4, 'end': 6},
               {'start': 6, 'end': 8}, {'start': 8, 'end': 10}]
    full_segs = ac.segments_full_timeline_analysis(full_sr, 10.0)
    sub = tmp_path / 'sub'
    sub.mkdir()
    assert ac.export_xml_xmeml(str(vid), full_segs, fps, str(sub), at_source_time=True)
    full_clips = _parse_xml_clips(sub / f'{base}_scenecut_full.xml')

    # Autocut: condensed timeline starts at 0
    assert short_clips[0][0] == 0
    assert short_clips[-1][1] == short_clips[-1][0] + (short_clips[-1][3] - short_clips[-1][2])

    # Full @ 25fps: timeline position matches source in/out
    for (t0, t1, fi, fo), (s, e) in zip(full_clips, full_segs):
        assert fi == ac._media_frames_from_sec(s, fps)
        assert fo == ac._media_frames_from_sec(e, fps)
        assert t0 == fi and t1 == fo, (t0, t1, fi, fo, s, e)


def test_edl_full_record_matches_source(tmp_path: Path):
    vid = _make_test_video(tmp_path / 'sample')
    fps = 25.0
    segs = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert ac.export_edl_cmx(str(vid), segs, fps, str(tmp_path), at_source_time=True)
    edl = tmp_path / f'{vid.stem}_scenecut_full.edl'
    events = _parse_edl_events(edl)
    assert len(events) == 3
    for ev in events:
        src_in, src_out, rec_in, rec_out = ev
        assert src_in == rec_in and src_out == rec_out, ev


def test_run_export_only_modes(tmp_path: Path):
    vid = _make_test_video(tmp_path / 'sample')
    fps = 25.0
    merged = [(0.0, 4.0), (6.0, 8.0)]
    seg_res = [
        {'start': 0, 'end': 2, 'final_category': 'action'},
        {'start': 2, 'end': 4, 'final_category': 'action'},
        {'start': 4, 'end': 6, 'final_category': 'story'},
        {'start': 6, 'end': 8, 'final_category': 'vocal'},
        {'start': 8, 'end': 10, 'final_category': 'vocal'},
    ]

    def cfg(engine: str) -> configparser.ConfigParser:
        c = configparser.ConfigParser()
        c['Settings'] = {'export_engine': engine}
        return c

    out = tmp_path / 'out'
    out.mkdir()

    assert ac.run_export_only(
        str(vid), merged, fps, str(out), cfg('DaVinci: Export Timeline (XML, cut points – full length)'),
        segment_results=seg_res,
    )
    base = vid.stem
    assert (out / f'{base}_scenecut_full.xml').is_file()
    clips = _parse_xml_clips(out / f'{base}_scenecut_full.xml')
    assert len(clips) == 5

    out2 = tmp_path / 'out2'
    out2.mkdir()
    assert ac.run_export_only(
        str(vid), merged, fps, str(out2), cfg('DaVinci: Export Timeline (XML, no cuts)'),
        segment_results=seg_res,
    )
    assert (out2 / f'{base}_scenecut.xml').is_file()
    clips_nc = _parse_xml_clips(out2 / f'{base}_scenecut.xml')
    assert len(clips_nc) == 1

    out3 = tmp_path / 'out3'
    out3.mkdir()
    assert ac.run_export_only(
        str(vid), merged, fps, str(out3), cfg('DaVinci: Export Timeline (XML)'),
        segment_results=seg_res,
    )
    clips_ac = _parse_xml_clips(out3 / f'{base}_scenecut.xml')
    assert len(clips_ac) == 2


def main() -> int:
    tests = [
        test_ntsc_timeline_vs_media_frames,
        test_segment_helpers,
        test_xml_autocut_vs_full,
        test_edl_full_record_matches_source,
        test_run_export_only_modes,
    ]
    failed = 0
    work_root = _ROOT / 'output' / '_export_test_run'
    work_root.mkdir(parents=True, exist_ok=True)
    for fn in tests:
        name = fn.__name__
        try:
            if fn in (test_segment_helpers, test_ntsc_timeline_vs_media_frames):
                fn()
            else:
                case_dir = work_root / name
                if case_dir.exists():
                    import shutil
                    shutil.rmtree(case_dir, ignore_errors=True)
                case_dir.mkdir(parents=True)
                fn(case_dir)
            print(f'PASS  {name}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {name}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
