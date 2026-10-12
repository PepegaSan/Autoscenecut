"""FCP7 XML (xmeml) and CMX EDL timeline files for DaVinci Resolve (File → Import → Timeline).

Taken unchanged from the former autocut_nvidia.py.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


def _video_w_h(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    return w, h


def _xml_rate_tags(fps):
    """FCP7-style timebase + ntsc flag for common frame rates."""
    f = float(fps)
    if abs(f - 23.976023976023978) < 0.01 or abs(f - 24.0) < 0.01:
        return 24, 'FALSE'
    if abs(f - 25.0) < 0.01:
        return 25, 'FALSE'
    if abs(f - 29.97002997002997) < 0.05 or abs(f - 30.0) < 0.05:
        return 30, 'TRUE'
    if abs(f - 50.0) < 0.01:
        return 25, 'FALSE'
    if abs(f - 59.94005994005994) < 0.05 or abs(f - 60.0) < 0.05:
        return 30, 'TRUE'
    tb = max(1, min(60, int(round(f))))
    return tb, 'FALSE'


def _is_ntsc_fps(fps):
    f = float(fps)
    return abs(f - 29.97002997002997) < 0.05 or abs(f - 59.94005994005994) < 0.05


def _media_frames_from_sec(sec, fps):
    """Frame index in the source file (same convention as OpenCV int(sec * fps) seek)."""
    f = float(fps)
    if abs(f - 29.97002997002997) < 0.05:
        return int(float(sec) * 30000.0 / 1001.0)
    if abs(f - 23.976023976023978) < 0.01:
        return int(float(sec) * 24000.0 / 1001.0)
    if abs(f - 59.94005994005994) < 0.05:
        return int(float(sec) * 60000.0 / 1001.0)
    return int(float(sec) * f)


def _edit_frames_from_sec(sec, timebase, ntsc):
    """FCP7 timeline edit units; NTSC uses timebase 30 so 5:00 = frame 9000 in Resolve."""
    if ntsc == 'TRUE':
        return int(round(float(sec) * int(timebase)))
    return _media_frames_from_sec(sec, timebase)


def _frames_to_tc(total_frames, timebase):
    tb = max(1, int(timebase))
    if total_frames < 0:
        total_frames = 0
    ff = total_frames % tb
    t = total_frames // tb
    s = t % 60
    t //= 60
    m = t % 60
    h = t // 60
    return f'{h:02d}:{m:02d}:{s:02d}:{ff:02d}'


def export_xml_xmeml(vid, segs, fps, out_dir, *, at_source_time=False):
    """Final Cut Pro 7 XML (xmeml) — DaVinci Resolve: File → Import → Timeline."""
    if not segs:
        return False
    os.makedirs(out_dir, exist_ok=True)
    abs_vid = os.path.abspath(vid)
    base_name = os.path.splitext(os.path.basename(abs_vid))[0]
    xml_name = f'{base_name}_scenecut_full.xml' if at_source_time else f'{base_name}_scenecut.xml'
    xml_path = os.path.join(out_dir, xml_name)
    pathurl = Path(abs_vid).as_uri()
    w, h = _video_w_h(abs_vid)
    tb, ntsc = _xml_rate_tags(fps)

    timeline_f = 0
    total_end = 0
    clip_elems = []
    ntsc_edit = ntsc == 'TRUE'
    for s, e in segs:
        fi = _media_frames_from_sec(s, fps)
        fo = _media_frames_from_sec(e, fps)
        if fo <= fi:
            continue
        if at_source_time and ntsc_edit:
            t0 = _edit_frames_from_sec(s, tb, ntsc)
            t1 = _edit_frames_from_sec(e, tb, ntsc)
            total_end = max(total_end, t1)
        elif at_source_time:
            t0, t1 = fi, fo
            total_end = max(total_end, t1)
        else:
            if ntsc_edit:
                t0 = timeline_f
                t1 = timeline_f + (_edit_frames_from_sec(e, tb, ntsc) - _edit_frames_from_sec(s, tb, ntsc))
            else:
                t0 = timeline_f
                t1 = timeline_f + (fo - fi)
            timeline_f = t1
            total_end = timeline_f
        clip_elems.append((t0, t1, fi, fo))

    if not clip_elems:
        return False

    root = ET.Element('xmeml', {'version': '5'})
    seq = ET.SubElement(root, 'sequence', {'id': 'seq_autocut'})
    ET.SubElement(seq, 'name').text = 'Scenecut_full' if at_source_time else 'Scenecut'
    ET.SubElement(seq, 'duration').text = str(total_end)
    rate = ET.SubElement(seq, 'rate')
    ET.SubElement(rate, 'timebase').text = str(tb)
    ET.SubElement(rate, 'ntsc').text = ntsc

    media = ET.SubElement(seq, 'media')
    video = ET.SubElement(media, 'video')
    fmt = ET.SubElement(video, 'format')
    sc = ET.SubElement(fmt, 'samplecharacteristics')
    r2 = ET.SubElement(sc, 'rate')
    ET.SubElement(r2, 'timebase').text = str(tb)
    ET.SubElement(r2, 'ntsc').text = ntsc
    ET.SubElement(sc, 'width').text = str(w)
    ET.SubElement(sc, 'height').text = str(h)

    track = ET.SubElement(video, 'track')
    for i, (t0, t1, fi, fo) in enumerate(clip_elems, start=1):
        ci = ET.SubElement(track, 'clipitem', {'id': f'clip-{i}'})
        ET.SubElement(ci, 'name').text = base_name
        ET.SubElement(ci, 'duration').text = str(fo - fi)
        cr = ET.SubElement(ci, 'rate')
        ET.SubElement(cr, 'timebase').text = str(tb)
        ET.SubElement(cr, 'ntsc').text = ntsc
        ET.SubElement(ci, 'start').text = str(t0)
        ET.SubElement(ci, 'end').text = str(t1)
        ET.SubElement(ci, 'in').text = str(fi)
        ET.SubElement(ci, 'out').text = str(fo)
        f_el = ET.SubElement(ci, 'file', {'id': f'file-{i}'})
        ET.SubElement(f_el, 'name').text = os.path.basename(abs_vid)
        ET.SubElement(f_el, 'pathurl').text = pathurl

    ET.indent(root, space='  ')
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        ET.tostring(root, encoding='unicode'),
    ]
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    label = 'XML (FCP7 xmeml, volle Länge)' if at_source_time else 'XML (FCP7 xmeml)'
    print(f'{label} geschrieben: {xml_path}', flush=True)
    return True


def export_edl_cmx(vid, segs, fps, out_dir, *, at_source_time=False):
    """CMX-style EDL; comments carry full source path for Resolve."""
    if not segs:
        return False
    os.makedirs(out_dir, exist_ok=True)
    abs_vid = os.path.abspath(vid)
    base_name = os.path.splitext(os.path.basename(abs_vid))[0]
    edl_name = f'{base_name}_scenecut_full.edl' if at_source_time else f'{base_name}_scenecut.edl'
    edl_path = os.path.join(out_dir, edl_name)
    tb, _ = _xml_rate_tags(fps)
    ntsc = abs(float(fps) - 29.97002997002997) < 0.05 or abs(float(fps) - 59.94005994005994) < 0.05
    fcm = 'DROP FRAME' if ntsc else 'NON-DROP FRAME'

    title = 'Scenecut_full_' if at_source_time else 'Scenecut_'
    lines = ['TITLE: ' + title + base_name.replace(' ', '_'), f'FCM: {fcm}', '']
    ev = 1
    timeline_f = 0
    ntsc_edit = ntsc
    for s, e in segs:
        fi = _media_frames_from_sec(s, fps)
        fo = _media_frames_from_sec(e, fps)
        if fo <= fi:
            continue
        src_in = _frames_to_tc(fi, tb)
        src_out = _frames_to_tc(fo, tb)
        if at_source_time and ntsc_edit:
            rec_in = _frames_to_tc(_edit_frames_from_sec(s, tb, 'TRUE'), tb)
            rec_out = _frames_to_tc(_edit_frames_from_sec(e, tb, 'TRUE'), tb)
        elif at_source_time:
            rec_in, rec_out = src_in, src_out
        else:
            if ntsc_edit:
                dur = _edit_frames_from_sec(e, tb, 'TRUE') - _edit_frames_from_sec(s, tb, 'TRUE')
            else:
                dur = fo - fi
            rec_in = _frames_to_tc(timeline_f, tb)
            rec_out = _frames_to_tc(timeline_f + dur, tb)
            timeline_f += dur
        lines.append(f'{ev:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}')
        lines.append(f'* FROM CLIP NAME: {os.path.basename(abs_vid)}')
        lines.append(f'* SCENECUT_SOURCE: {abs_vid.replace(chr(92), "/")}')
        lines.append('')
        ev += 1

    with open(edl_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    label = 'EDL (volle Länge)' if at_source_time else 'EDL'
    print(f'{label} geschrieben: {edl_path}', flush=True)
    return True

