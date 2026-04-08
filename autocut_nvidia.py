import sys
import os
import shutil
import time
import cv2
import subprocess
import configparser
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from analyzer_nvidia import analyze_segment

CREATE_NO_WINDOW = 0x08000000
CFG_PATH = 'config_nvidia.ini'
CONTROL_PATH = os.path.join('output', 'autocut_control.json')


def load_cfg():
    cfg = configparser.ConfigParser()
    cfg.read(CFG_PATH, encoding='utf-8')
    return cfg


def _read_pause_state():
    try:
        with open(CONTROL_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return bool(data.get('paused', False))
    except Exception:
        return False


def wait_if_paused():
    paused_once = False
    while _read_pause_state():
        if not paused_once:
            print('PAUSED: Processing paused by user. Waiting for resume...', flush=True)
            paused_once = True
        time.sleep(0.4)
    if paused_once:
        print('RESUMED: Processing resumed.', flush=True)


def _davinci_resolve_script_import_ok(py_exe, api_path):
    if not py_exe or not os.path.isfile(py_exe) or not api_path or not os.path.isdir(api_path):
        return False
    api_lit = json.dumps(api_path)
    code = f'import sys; sys.path.insert(0, {api_lit}); import DaVinciResolveScript'
    try:
        r = subprocess.run(
            [py_exe, '-c', code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=25,
            creationflags=CREATE_NO_WINDOW,
        )
        return r.returncode == 0
    except Exception:
        return False


def _interpreter_major_minor(py_exe):
    try:
        r = subprocess.run(
            [py_exe, '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        if r.returncode != 0:
            return None
        parts = (r.stdout or '').strip().split()
        if len(parts) < 2:
            return None
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def pick_davinci_worker_python(cfg):
    """
    Resolve's DaVinciResolveScript.pyd matches one CPython ABI per Resolve version. On current Windows
    Resolve builds this is almost always Python 3.12. Using 3.9/3.11 can pass a naive import test but
    then fail to talk to Resolve correctly — set davinci_python_path to 3.12 (e.g. f4python\\3.12).
    """
    api_path = cfg.get('Settings', 'resolve_api_path', fallback='').strip()
    override = cfg.get('Settings', 'davinci_python_path', fallback='').strip()
    if override and os.path.isfile(override):
        if _davinci_resolve_script_import_ok(override, api_path):
            print(f'INFO: DaVinci-Worker nutzt INI-Override: {override}', flush=True)
            return override
        print(
            'WARNUNG: davinci_python_path aus INI kann DaVinciResolveScript nicht laden — Auto-Erkennung.',
            flush=True,
        )

    envp = (os.environ.get('DAVINCI_SCRIPT_PYTHON') or '').strip()
    if envp and os.path.isfile(envp) and _davinci_resolve_script_import_ok(envp, api_path):
        print(f'INFO: DaVinci-Worker nutzt DAVINCI_SCRIPT_PYTHON: {envp}', flush=True)
        return envp

    candidates = []
    if sys.platform == 'win32':
        # Resolve ships a CPython build under f4python — often works when store python.org does not.
        for base in (
            os.environ.get('ProgramFiles', r'C:\Program Files'),
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
        ):
            for tail in (
                os.path.join('Blackmagic Design', 'DaVinci Resolve', 'f4python', '3.12', 'bin', 'python.exe'),
                os.path.join('Blackmagic Design', 'DaVinci Resolve', 'f4python', '3.12', 'python.exe'),
            ):
                p = os.path.join(base, tail)
                if os.path.isfile(p):
                    candidates.append(p)
        for ver in ('3.12', '3.11', '3.10'):
            try:
                r = subprocess.run(
                    ['py', f'-{ver}', '-c', 'import sys; print(sys.executable)'],
                    capture_output=True,
                    text=True,
                    timeout=12,
                    creationflags=CREATE_NO_WINDOW,
                )
                if r.returncode == 0:
                    p = (r.stdout or '').strip().strip('"')
                    if p and os.path.isfile(p):
                        candidates.append(p)
            except Exception:
                pass
        local = os.environ.get('LOCALAPPDATA', '')
        for v, folder in (
            ('312', 'Python312'),
            ('311', 'Python311'),
            ('310', 'Python310'),
        ):
            for p in (
                os.path.join(local, 'Programs', 'Python', folder, 'python.exe'),
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), folder, 'python.exe'),
                rf'C:\Python{v}\python.exe',
                rf'C:\Program Files\Python{v}\python.exe',
            ):
                if os.path.isfile(p):
                    candidates.append(p)
    else:
        for name in ('python3.12', 'python3.11', 'python3.10', 'python3', 'python'):
            p = shutil.which(name)
            if p:
                candidates.append(p)

    if not getattr(sys, 'frozen', False):
        candidates.append(sys.executable)

    seen = set()

    def try_candidates(version_filter_312):
        """On Windows, prefer 3.12 first — other versions often misbehave with Resolve despite import OK."""
        for exe in candidates:
            if exe in seen:
                continue
            if sys.platform == 'win32' and version_filter_312:
                ver = _interpreter_major_minor(exe)
                if ver != (3, 12):
                    continue
            seen.add(exe)
            if not _davinci_resolve_script_import_ok(exe, api_path):
                continue
            if sys.platform == 'win32' and version_filter_312:
                print(f'INFO: DaVinci-Worker-Python (auto, Windows bevorzugt 3.12): {exe}', flush=True)
            else:
                print(f'INFO: DaVinci-Worker-Python (auto): {exe}', flush=True)
            return exe
        return None

    if sys.platform == 'win32':
        picked = try_candidates(version_filter_312=True)
        if picked:
            return picked
        seen.clear()
        print(
            'WARNUNG: Kein Python 3.12 mit importierbarem DaVinciResolveScript — Fallback alle Versionen.',
            flush=True,
        )

    for exe in candidates:
        if exe in seen:
            continue
        seen.add(exe)
        if _davinci_resolve_script_import_ok(exe, api_path):
            if sys.platform == 'win32':
                print(
                    f'WARNUNG: DaVinci-Worker nutzt {exe} (nicht 3.12). Wenn Resolve nicht reagiert, '
                    'davinci_python_path auf Python 3.12 setzen (z. B. f4python\\3.12\\bin\\python.exe).',
                    flush=True,
                )
            else:
                print(f'INFO: DaVinci-Worker-Python (auto): {exe}', flush=True)
            return exe

    if getattr(sys, 'frozen', False):
        print(
            'EXPORT_FAILED: Kein python.exe gefunden, das DaVinciResolveScript laden kann.\n'
            'Die Scenecut-EXE ist kein Python — sie kann den DaVinci-Worker nicht starten.\n'
            'Lösung: In der GUI (Export-Tab) „Custom Python Worker Path“ setzen — typisch z.B.\n'
            '  …\\DaVinci Resolve\\f4python\\3.12\\bin\\python.exe\n'
            'oder ein Python 3.12, das mit resolve_api_path (Modules) import DaVinciResolveScript schafft.\n'
            'Alternativ Umgebungsvariable DAVINCI_SCRIPT_PYTHON auf diese python.exe.',
            flush=True,
        )
        return None

    fallback = sys.executable if os.path.isfile(sys.executable) else 'python'
    print(
        'WARNUNG: Kein Python mit importierbarem DaVinciResolveScript gefunden — nutze Fallback. '
        'Installiere Python 3.12 (Windows Store oder python.org) oder setze davinci_python_path / DAVINCI_SCRIPT_PYTHON.',
        flush=True,
    )
    return fallback


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


def _frames_from_sec(sec, fps):
    return int(round(float(sec) * float(fps)))


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


def export_xml_xmeml(vid, segs, fps, out_dir):
    """Final Cut Pro 7 XML (xmeml) — DaVinci Resolve: File → Import → Timeline."""
    if not segs:
        return False
    os.makedirs(out_dir, exist_ok=True)
    abs_vid = os.path.abspath(vid)
    base_name = os.path.splitext(os.path.basename(abs_vid))[0]
    xml_path = os.path.join(out_dir, f'{base_name}_scenecut.xml')
    pathurl = Path(abs_vid).as_uri()
    w, h = _video_w_h(abs_vid)
    tb, ntsc = _xml_rate_tags(fps)

    timeline_f = 0
    total_end = 0
    clip_elems = []
    for s, e in segs:
        fi = _frames_from_sec(s, fps)
        fo = _frames_from_sec(e, fps)
        if fo <= fi:
            continue
        dur = fo - fi
        clip_elems.append((timeline_f, timeline_f + dur, fi, fo))
        timeline_f += dur
        total_end = timeline_f

    if not clip_elems:
        return False

    root = ET.Element('xmeml', {'version': '5'})
    seq = ET.SubElement(root, 'sequence', {'id': 'seq_autocut'})
    ET.SubElement(seq, 'name').text = 'Scenecut'
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
    print(f'XML (FCP7 xmeml) geschrieben: {xml_path}', flush=True)
    return True


def export_edl_cmx(vid, segs, fps, out_dir):
    """CMX-style EDL; comments carry full source path for Resolve."""
    if not segs:
        return False
    os.makedirs(out_dir, exist_ok=True)
    abs_vid = os.path.abspath(vid)
    base_name = os.path.splitext(os.path.basename(abs_vid))[0]
    edl_path = os.path.join(out_dir, f'{base_name}_scenecut.edl')
    tb, _ = _xml_rate_tags(fps)
    ntsc = abs(float(fps) - 29.97002997002997) < 0.05 or abs(float(fps) - 59.94005994005994) < 0.05
    fcm = 'DROP FRAME' if ntsc else 'NON-DROP FRAME'

    lines = ['TITLE: Scenecut_' + base_name.replace(' ', '_'), f'FCM: {fcm}', '']
    ev = 1
    timeline_f = 0
    for s, e in segs:
        fi = _frames_from_sec(s, fps)
        fo = _frames_from_sec(e, fps)
        if fo <= fi:
            continue
        dur = fo - fi
        src_in = _frames_to_tc(fi, tb)
        src_out = _frames_to_tc(fo, tb)
        rec_in = _frames_to_tc(timeline_f, tb)
        rec_out = _frames_to_tc(timeline_f + dur, tb)
        lines.append(f'{ev:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}')
        lines.append(f'* FROM CLIP NAME: {os.path.basename(abs_vid)}')
        lines.append(f'* SCENECUT_SOURCE: {abs_vid.replace(chr(92), "/")}')
        lines.append('')
        timeline_f += dur
        ev += 1

    with open(edl_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(f'EDL geschrieben: {edl_path}', flush=True)
    return True


def clamp_int(v, low=0, high=100):
    return max(low, min(high, int(round(v))))


def wpm_threshold_from_calibrated_words(word_count, ref_sec):
    """WPM floor equivalent to '≥ N words in ref_sec' (e.g. 12 words / 20 s → 36 WPM). Independent of analysis segment length."""
    if ref_sec <= 0:
        ref_sec = 20.0
    return float(word_count) * 60.0 / ref_sec


def decide_category(metrics, cfg):
    min_story = cfg.getint('Thresholds', 'min_story_score', fallback=56)
    min_action = cfg.getint('Thresholds', 'min_action_score', fallback=58)
    min_vocal = cfg.getint('Thresholds', 'min_vocal_score', fallback=32)

    vocal_story_penalty_factor = cfg.getfloat('Thresholds', 'vocal_story_penalty_factor', fallback=0.70)
    vocal_speech_penalty_factor = cfg.getfloat('Thresholds', 'vocal_speech_penalty_factor', fallback=0.28)
    action_story_penalty_factor = cfg.getfloat('Thresholds', 'action_story_penalty_factor', fallback=0.52)
    action_speech_penalty_factor = cfg.getfloat('Thresholds', 'action_speech_penalty_factor', fallback=0.22)

    story = float(metrics.get('story_score', 0))
    action_raw = float(metrics.get('action_score', 0))
    vocal_raw = float(metrics.get('sexual_vocal_score', 0))
    speech = float(metrics.get('speech_percent', 0))
    silence = float(metrics.get('silence_percent', 0))
    music = float(metrics.get('music_percent', 0))
    moan = float(metrics.get('moan_percent', 0))
    breath = float(metrics.get('breath_percent', 0))
    scream = float(metrics.get('scream_percent', 0))
    words = int(metrics.get('word_count', 0))
    human_vocal = float(metrics.get('human_vocal_percent', 0))

    ref = max(1, cfg.getint('Settings', 'calibration_segment_seconds', fallback=20))
    wpm = metrics.get('wpm')
    if wpm is None:
        seg = max(5, min(60, cfg.getint('Settings', 'interval_seconds', fallback=20)))
        wpm = float(words) * 60.0 / float(seg) if seg else 0.0
    else:
        wpm = float(wpm)
    wpm_min = lambda n: wpm_threshold_from_calibrated_words(n, ref)

    vocal_penalty = (story * vocal_story_penalty_factor) + (speech * vocal_speech_penalty_factor)
    action_penalty = (story * action_story_penalty_factor) + (speech * action_speech_penalty_factor)

    vocal_effective = clamp_int(vocal_raw - vocal_penalty)
    action_effective = clamp_int(action_raw - action_penalty)

    dialogue_lock = (
        story >= min_story and
        speech >= 20 and
        wpm >= wpm_min(12) and
        music < 60 and
        scream < 42 and
        moan < 18 and
        breath < 24
    )

    hard_dialogue_lock = (
        story >= max(min_story + 4, 60) and
        wpm >= wpm_min(16) and
        speech >= 18 and
        scream < 48 and
        vocal_effective < (min_vocal + 8) and
        action_effective < (min_action + 8)
    )

    soft_dialogue_bias = (
        (
            story >= 42 and
            wpm >= wpm_min(10) and
            speech >= 16 and
            scream < 18 and
            moan < 12 and
            breath < 12 and
            vocal_effective < min_vocal
        )
        or
        (
            story >= 56 and
            wpm >= wpm_min(24) and
            speech >= 6 and
            scream < 12 and
            moan < 8 and
            breath < 8 and
            vocal_effective < max(10, min_vocal - 8)
        )
    )

    if soft_dialogue_bias:
        action_effective = clamp_int(action_effective - 8)

    dialogue = dialogue_lock or hard_dialogue_lock
    vocal_signal = max(vocal_effective, clamp_int(human_vocal * 0.50))

    vocal = (
        vocal_signal >= min_vocal and
        (moan >= 12 or breath >= 14 or scream >= 18 or vocal_effective >= (min_vocal + 8)) and
        silence < 85 and
        not dialogue and
        not (speech >= 80 and wpm >= wpm_min(16) and vocal_effective < (min_vocal + 12)) and
        not (wpm >= wpm_min(35) and moan < 15 and vocal_effective < (min_vocal + 8))
    )

    action = (
        action_effective >= min_action and
        not dialogue and
        not (story >= 42 and wpm >= wpm_min(10) and speech >= 16 and action_effective < (min_action + 6)) and
        (vocal_effective < min_vocal or scream >= 40)
    )

    if hard_dialogue_lock:
        action = False
        vocal = False

    music_cat = music >= 55 and action_effective >= int(min_action * 0.70) and speech < 45 and not dialogue
    silence_cat = (silence >= 70 and action_effective < 15 and story < 20) or (story < 15 and action_raw < 15 and vocal_raw < 15)

    if dialogue:
        final = 'dialogue'
    elif vocal:
        final = 'vocal'
    elif action:
        final = 'action'
    elif music_cat:
        final = 'music'
    elif silence_cat:
        final = 'silence'
    else:
        final = 'dialogue' if soft_dialogue_bias or (story >= 40 and wpm >= wpm_min(10) and speech >= 16) else ('action' if action_effective >= vocal_signal else 'vocal')

    return {
        'dialogue': dialogue,
        'action': action,
        'vocal': vocal,
        'music': music_cat,
        'silence': silence_cat,
        'final_category': final,
        'dialogue_lock': dialogue_lock,
        'hard_dialogue_lock': hard_dialogue_lock,
        'soft_dialogue_bias': soft_dialogue_bias,
        'vocal_effective_score': vocal_effective,
        'action_effective_score': action_effective,
        'vocal_signal_score': vocal_signal,
        'vocal_penalty': round(vocal_penalty, 2),
        'action_penalty': round(action_penalty, 2),
    }


def should_keep(flags, cfg):
    return should_keep_category(flags['final_category'], cfg)


def should_keep_category(final_category, cfg):
    """Reusable when re-filtering from checkpoint (only final_category string per segment)."""
    return any([
        cfg.getboolean('Categories', 'keep_action', fallback=False) and final_category == 'action',
        cfg.getboolean('Categories', 'keep_dialogue', fallback=False) and final_category == 'dialogue',
        cfg.getboolean('Categories', 'keep_vocal', fallback=False) and final_category == 'vocal',
        cfg.getboolean('Categories', 'keep_music', fallback=False) and final_category == 'music',
        cfg.getboolean('Categories', 'keep_silence', fallback=False) and final_category == 'silence',
    ])


def merge_adjacent_keep_segments(keep):
    if not keep:
        return []
    merged = [keep[0]]
    for cur in keep[1:]:
        if cur[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], cur[1]))
        else:
            merged.append(cur)
    return merged


def probe_video_bitrate_kbps(video_path):
    """
    Schätzt die Video-Stream-Bitrate in kb/s (kilobits/s). ffprobe liefert bit_rate i. d. R. in bit/s.
    Kein exakter VBR-„Durchschnitt“, aber nah an der Quell-Metadaten-Bitrate; sonst grobe Schätzung.
    """
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        r = subprocess.run(
            [
                'ffprobe',
                '-v',
                'error',
                '-select_streams',
                'v:0',
                '-show_entries',
                'stream=bit_rate,avg_bitrate',
                '-show_entries',
                'format=bit_rate,duration,size',
                '-of',
                'json',
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=90,
            creationflags=CREATE_NO_WINDOW,
        )
        if r.returncode != 0:
            return None
        j = json.loads(r.stdout or '{}')
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError, ValueError):
        return None

    def _bps_to_kbps(bps_str):
        if not bps_str:
            return None
        try:
            b = int(bps_str)
            if b <= 0:
                return None
            return max(1, b // 1000)
        except (TypeError, ValueError):
            return None

    streams = j.get('streams') or []
    if streams:
        st0 = streams[0]
        for key in ('bit_rate', 'avg_bitrate'):
            kb = _bps_to_kbps(st0.get(key))
            if kb and 200 <= kb <= 800000:
                return kb

    fmt = j.get('format') or {}
    fb = _bps_to_kbps(fmt.get('bit_rate'))
    if fb and 200 <= fb <= 800000:
        return max(200, fb - 256)

    try:
        dur = float(fmt.get('duration') or 0)
        size = int(fmt.get('size') or 0)
    except (TypeError, ValueError):
        dur, size = 0.0, 0
    if dur > 0.5 and size > 100000:
        total_kbps = int((size * 8) / dur / 1000)
        guess = max(200, total_kbps - 256)
        if guess <= 800000:
            return guess
    return None


def export_target_video_kbps(cfg, source_video):
    """
    export_bitrate_mode: default | match_source | manual
    default: FFmpeg nutzt NVENC nur mit Preset / AMD-Build nutzt CRF — kein Ziel-kbps.
    """
    mode = (cfg.get('Settings', 'export_bitrate_mode', fallback='default') or 'default').strip().lower()
    if mode in ('match_source', 'match', 'source'):
        kbps = probe_video_bitrate_kbps(source_video)
        if kbps:
            print(f'INFO: Video-Zielbitrate ~{kbps} kb/s (Quelle/ffprobe).', flush=True)
            return kbps
        print(
            'WARNUNG: export_bitrate_mode=match_source, Bitrate nicht ermittelbar — Preset/Standard-Encoding.',
            flush=True,
        )
        return None
    if mode in ('manual', 'fixed'):
        raw = (cfg.get('Settings', 'export_manual_video_kbps', fallback='') or '').strip()
        try:
            v = int(raw)
            if v < 200:
                print('WARNUNG: export_manual_video_kbps zu niedrig (<200), ignoriert.', flush=True)
                return None
            return min(800000, v)
        except ValueError:
            print('WARNUNG: export_manual_video_kbps ist keine Zahl.', flush=True)
            return None
    return None


def write_autocut_checkpoint(vid, merged, fps, step, out_dir, cfg_snapshot, segment_results=None):
    """After analysis, before export — enables retry / manual FFmpeg without re-running Whisper.

    segment_results: list of {start, end, final_category} for all analyzed segments — used on retry
    to rebuild the timeline from current category switches without re-running analysis.
    """
    out = Path('output')
    out.mkdir(parents=True, exist_ok=True)
    ck_path = out / 'last_autocut_checkpoint.json'
    payload = {
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'video_path': os.path.abspath(vid),
        'out_dir': os.path.abspath(out_dir),
        'merged_segments': [[float(s), float(e)] for s, e in merged],
        'fps': float(fps),
        'interval_seconds': int(step),
        **cfg_snapshot,
    }
    if segment_results is not None:
        payload['segment_results'] = segment_results
    with open(ck_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'CHECKPOINT:{ck_path.resolve()}', flush=True)
    return ck_path


def export_ffmpeg(vid, segs, preset, out_dir, cfg):
    target_kbps = export_target_video_kbps(cfg, vid)
    if target_kbps:
        br = f' -rc:v vbr -b:v {target_kbps}k -maxrate {int(target_kbps * 1.25)}k -bufsize {int(target_kbps * 2)}k'
        print(f'INFO: NVENC mit Ziel-Bitrate (VBR): {target_kbps} kb/s.', flush=True)
    else:
        br = ''
    total = len(segs)
    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, 'concat_list.txt')
    final_out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(vid))[0]}_scenecut_export.mp4")

    enc_ok = True
    with open(list_path, 'w', encoding='utf-8') as f:
        for idx, (s, e) in enumerate(segs):
            wait_if_paused()
            temp_out = os.path.join(out_dir, f'temp_{idx}.mp4')
            cmd = f'ffmpeg -y -ss {s} -to {e} -i "{vid}" -c:v hevc_nvenc -preset {preset}{br} -c:a aac "{temp_out}"'
            r = subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
            )
            bad = r.returncode != 0 or not os.path.isfile(temp_out) or os.path.getsize(temp_out) == 0
            if bad:
                enc_ok = False
                print(f'FEHLER: Segment-Encode {idx + 1}/{total} fehlgeschlagen.', flush=True)
            f.write(f"file '{temp_out}'\n")
            print(f'PROGRESS:{55 + int(((idx + 1) / total) * 35)}', flush=True)
            print(f'Rendered segment {idx+1} of {total}', flush=True)

    if not enc_ok:
        print(
            'HINWEIS: temp_*.mp4 und concat_list.txt bleiben im Ausgabeordner (nicht gelöscht). Concat nicht ausgeführt.',
            flush=True,
        )
        return False

    wait_if_paused()
    print('Concatenating final video...', flush=True)
    r2 = subprocess.run(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" -c copy "{final_out}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    concat_ok = r2.returncode == 0 and os.path.isfile(final_out) and os.path.getsize(final_out) > 0
    if concat_ok:
        for idx in range(total):
            try:
                os.remove(os.path.join(out_dir, f'temp_{idx}.mp4'))
            except OSError:
                pass
        try:
            os.remove(list_path)
        except OSError:
            pass
        print(f'FFmpeg Export finished. File saved to: {final_out}', flush=True)
        return True

    print(
        f'FEHLER: Concat fehlgeschlagen. temp_0..{total - 1}.mp4 und concat_list.txt bleiben in: {out_dir}',
        flush=True,
    )
    return False


def render_davinci(vid, segs, api_path, fps, out_dir, cfg, resolve_py=None):
    py_exe = resolve_py if resolve_py else pick_davinci_worker_python(cfg)
    if not py_exe or not os.path.isfile(py_exe):
        print('EXPORT_FAILED: DaVinci AUTO-RENDER abgebrochen (kein Worker-Python).', flush=True)
        return False
    if getattr(sys, 'frozen', False):
        try:
            if os.path.normcase(os.path.abspath(py_exe)) == os.path.normcase(os.path.abspath(sys.executable)):
                print(
                    'EXPORT_FAILED: Worker-Python darf nicht die Scenecut-EXE sein — bitte davinci_python_path setzen.',
                    flush=True,
                )
                return False
        except OSError:
            pass
    abs_vid = os.path.abspath(vid).replace('\\', '/')
    safe_out_dir = os.path.abspath(out_dir).replace('\\', '/')
    safe_api_path = (api_path or '').replace('\\', '/')

    base_name = os.path.splitext(os.path.basename(abs_vid))[0]
    custom_name = f"{base_name}_scenecut_export"

    rate_kbps = export_target_video_kbps(cfg, vid)

    data = {
        "vid": abs_vid,
        "segs": segs,
        "api_path": safe_api_path,
        "fps": fps,
        "out_dir": safe_out_dir,
        "custom_name": custom_name,
        "resolve_data_rate_kbps": rate_kbps,
    }

    data_file = os.path.abspath("davinci_job.json").replace('\\', '/')
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    worker_code = """import sys
import json
import time
import os

with open(r'{data_file}', 'r', encoding='utf-8') as f:
    data = json.load(f)

if data.get('api_path') and data['api_path'] not in sys.path:
    sys.path.append(data['api_path'])

try:
    import DaVinciResolveScript as dvr
except Exception as e:
    print('FEHLER: DaVinciResolveScript konnte nicht importiert werden:', e, flush=True)
    sys.exit(1)

try:
    res = None
    for attempt in range(40):
        res = dvr.scriptapp('Resolve')
        if res:
            break
        print('Warte auf DaVinci Resolve (läuft und Scripting aktiv?) ...', attempt + 1, '/40', flush=True)
        time.sleep(0.75)
    if not res:
        print('FEHLER: DaVinci Resolve API antwortet nicht. Resolve starten, Projekt öffnen, '
              'Einstellungen → System → Allgemein → "Externe Skriptsteuerung" aktivieren.', flush=True)
        sys.exit(1)

    pm = res.GetProjectManager()
    proj = pm.GetCurrentProject() if pm else None
    if not proj:
        print('FEHLER: Kein offenes DaVinci-Projekt. Bitte ein Projekt öffnen und erneut starten.', flush=True)
        sys.exit(1)

    mp = proj.GetMediaPool()
    if not mp:
        print('FEHLER: Media Pool nicht verfügbar.', flush=True)
        sys.exit(1)

    time.sleep(1.5)
    clips = mp.ImportMedia([data['vid']])
    if not clips:
        print('FEHLER: Video konnte nicht in den Media Pool importiert werden (Pfad, Codec oder Datei gesperrt?).', flush=True)
        sys.exit(1)

    clip = clips[0]
    clip_props = clip.GetClipProperty() or {{}}
    res_text = clip_props.get('Resolution', '')
    fps_val = clip_props.get('FPS', '')
    width, height = None, None

    try:
        if res_text:
            w_h = str(res_text).lower().replace(' ', '').split('x', 1)
            width, height = int(w_h[0]), int(w_h[1])
            proj.SetSetting('timelineResolutionWidth', str(width))
            proj.SetSetting('timelineResolutionHeight', str(height))
    except Exception as e:
        print('INFO: Auflösung aus Clip-Metadaten nicht übernommen:', e, flush=True)

    try:
        if fps_val:
            proj.SetSetting('timelineFrameRate', str(fps_val))
            proj.SetSetting('timelinePlaybackFrameRate', str(fps_val))
    except Exception as e:
        print('INFO: FPS aus Clip-Metadaten nicht übernommen:', e, flush=True)

    tl_name = 'Autocut_Render_' + str(int(time.time()))
    tl = mp.CreateEmptyTimeline(tl_name)
    if not tl:
        print('FEHLER: Konnte keine leere Timeline erstellen (Name vergeben?).', flush=True)
        sys.exit(1)

    proj.SetCurrentTimeline(tl)

    try:
        for track_type in ('video', 'audio'):
            idx = 1
            while idx <= 20:
                items = tl.GetItemListInTrack(track_type, idx)
                if items is None:
                    break
                if items:
                    tl.DeleteClips(items, False)
                idx += 1
    except Exception as e:
        print('INFO: Timeline-Leerung übersprungen:', e, flush=True)

    append_list = []
    for s, e in data['segs']:
        append_list.append({{
            'mediaPoolItem': clip,
            'startFrame': int(s * data['fps']),
            'endFrame': int(e * data['fps']),
        }})

    if not append_list:
        print('FEHLER: Keine Segmente zum Anhängen.', flush=True)
        sys.exit(1)

    ok_append = mp.AppendToTimeline(append_list)
    if ok_append is False:
        print('FEHLER: AppendToTimeline fehlgeschlagen (Zeitbereich/FPS passen nicht zum Clip?).', flush=True)
        sys.exit(1)

    proj.SetCurrentTimeline(tl)
    proj.DeleteAllRenderJobs()
    time.sleep(0.35)

    if not proj.LoadRenderPreset('AutoCutPreset'):
        print("WARNUNG: Render-Preset 'AutoCutPreset' nicht gefunden — Resolve-Standardeinstellungen.", flush=True)

    render_settings = {{
        'SelectAllFrames': True,
        'TargetDir': data['out_dir'],
        'CustomName': data['custom_name'],
    }}
    if width and height:
        render_settings['ResolutionWidth'] = width
        render_settings['ResolutionHeight'] = height

    dr = data.get('resolve_data_rate_kbps')
    if dr is not None:
        try:
            render_settings['DataRate'] = str(int(dr))
            print('INFO: Deliver DataRate (kb/s) =', int(dr), '(API; Preset muss „Restrict“/Bitrate erlauben)', flush=True)
        except (TypeError, ValueError):
            pass

    proj.SetRenderSettings(render_settings)
    time.sleep(0.25)

    job_ok = False
    for jtry in range(8):
        rj = proj.AddRenderJob()
        if rj is not False:
            job_ok = True
            break
        print('WARNUNG: AddRenderJob meldete False, erneuter Versuch', jtry + 1, '/8', flush=True)
        time.sleep(0.6)
    if not job_ok:
        print('FEHLER: AddRenderJob blieb False (Render-Queue, Preset, Zielordner oder Resolve-UI prüfen).', flush=True)
        sys.exit(1)

    proj.StartRendering()

    max_render_wait = 4 * 3600
    waited = 0
    while proj.IsRenderingInProgress():
        time.sleep(3)
        waited += 3
        if waited > max_render_wait:
            print('WARNUNG: Render-Wartezeit überschritten (4h) — beende Warteschleife.', flush=True)
            break

    print('DaVinci Render-Job beendet. Zielordner:', data['out_dir'], flush=True)

except Exception as e:
    print('DaVinci System-Fehler beim Rendern:', e, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
""".format(data_file=data_file)

    worker_file = os.path.abspath("davinci_worker.py")
    with open(worker_file, "w", encoding="utf-8") as f:
        f.write(worker_code)

    print('Starte isolierten DaVinci-Worker...', flush=True)
    try:
        wr = subprocess.run([py_exe, '-u', worker_file])
    except OSError as e:
        print(f'FEHLER: DaVinci-Worker konnte nicht gestartet werden: {e}', flush=True)
        return False

    if wr.returncode != 0:
        print(
            f'DaVinci-Worker Exit-Code: {wr.returncode}. '
            f'Behalten: {data_file} und {worker_file} (manuell wiederholbar).',
            flush=True,
        )
        return False

    for path in (data_file, worker_file):
        try:
            os.remove(path)
        except OSError:
            pass
    return True


def run_export_only(vid, merged, fps, out_dir, cfg, resolve_py=None):
    """Uses current config: export_engine, ffmpeg preset, Resolve paths, XML/EDL."""
    engine = cfg.get('Settings', 'export_engine', fallback='FFmpeg: H.265 (Hardware NVENC)')
    api_path = cfg.get('Settings', 'resolve_api_path', fallback='')
    preset = cfg.get('Settings', 'ffmpeg_nvenc_preset', fallback='p4')
    if 'AUTO-RENDER' in engine:
        return render_davinci(vid, merged, api_path, fps, out_dir, cfg, resolve_py=resolve_py)
    if 'EDL' in engine:
        return export_edl_cmx(vid, merged, fps, out_dir)
    if 'XML' in engine:
        return export_xml_xmeml(vid, merged, fps, out_dir)
    return export_ffmpeg(vid, merged, preset, out_dir, cfg)


def retry_export_from_checkpoint():
    cfg = load_cfg()
    ck_path = Path('output') / 'last_autocut_checkpoint.json'
    if not ck_path.is_file():
        print(
            'NO_CHECKPOINT: output/last_autocut_checkpoint.json fehlt — zuerst vollen Autocut mit Analyse laufen lassen.',
            flush=True,
        )
        sys.exit(3)
    with open(ck_path, 'r', encoding='utf-8') as f:
        ck = json.load(f)
    vid = (ck.get('video_path') or '').strip()
    if not vid or not os.path.isfile(vid):
        print(f'VIDEO_MISSING: Datei nicht gefunden: {vid}', flush=True)
        sys.exit(4)

    seg_res = ck.get('segment_results')
    if seg_res:
        keep = []
        for s in seg_res:
            cat = (s.get('final_category') or 'dialogue').strip()
            try:
                t0, t1 = float(s['start']), float(s['end'])
            except (KeyError, TypeError, ValueError):
                continue
            if should_keep_category(cat, cfg):
                keep.append((t0, t1))
        keep.sort(key=lambda x: x[0])
        merged = merge_adjacent_keep_segments(keep)
        if not merged:
            print(
                'REEXPORT_NO_SEGMENTS: Mit den aktuellen Kategorie-Schaltern (INI) bleibt kein Segment übrig.',
                flush=True,
            )
            sys.exit(6)
        print(
            f'RETRY_EXPORT: category_refilter=on source_segments={len(seg_res)} merged_after_filter={len(merged)}',
            flush=True,
        )
    else:
        raw_segs = ck.get('merged_segments') or []
        merged = [(float(a), float(b)) for a, b in raw_segs]
        if not merged:
            print('CHECKPOINT_EMPTY: merged_segments ist leer.', flush=True)
            sys.exit(5)
        print('RETRY_EXPORT: category_refilter=off (alter Checkpoint ohne segment_results)', flush=True)

    fps = float(ck.get('fps') or 30.0)

    out_cfg = cfg.get('Settings', 'output_path', fallback='').strip()
    if out_cfg and os.path.isdir(out_cfg):
        out_dir = os.path.abspath(out_cfg)
    else:
        out_dir = os.path.abspath(ck.get('out_dir') or os.path.dirname(vid))

    engine = cfg.get('Settings', 'export_engine', fallback='FFmpeg: H.265 (Hardware NVENC)')
    print(f'RETRY_EXPORT: engine={engine}', flush=True)
    print(f'RETRY_EXPORT: video={vid}', flush=True)
    print(f'RETRY_EXPORT: out_dir={out_dir}', flush=True)
    print(f'RETRY_EXPORT: segments={len(merged)}', flush=True)

    export_ok = run_export_only(vid, merged, fps, out_dir, cfg)
    print('PROGRESS:100', flush=True)
    if export_ok:
        print('Process successfully finished!', flush=True)
    else:
        print('EXPORT_FAILED: Export-Wiederholung fehlgeschlagen.', flush=True)
        sys.exit(2)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--retry-export':
        retry_export_from_checkpoint()
        return
    if len(sys.argv) < 2:
        return
    vid = sys.argv[1]
    cfg = load_cfg()
    step = max(5, min(60, cfg.getint('Settings', 'interval_seconds', fallback=20)))
    engine = cfg.get('Settings', 'export_engine', fallback='FFmpeg: H.265 (Hardware NVENC)')
    api_path = cfg.get('Settings', 'resolve_api_path', fallback='')

    out_dir = cfg.get('Settings', 'output_path', fallback='').strip()
    if not out_dir or not os.path.isdir(out_dir):
        out_dir = os.path.dirname(os.path.abspath(vid))

    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cap.release()

    total_steps = max(1, len(range(0, dur, step)))
    csv_rows = []
    keep = []
    segment_results = []

    for idx, t in enumerate(range(0, dur, step), start=1):
        wait_if_paused()
        metrics = analyze_segment(vid, t, step)
        flags = decide_category(metrics, cfg)
        keep_flag = should_keep(flags, cfg)
        seg_end = min(t + step, dur)
        segment_results.append(
            {'start': float(t), 'end': float(seg_end), 'final_category': flags['final_category']}
        )

        if keep_flag:
            keep.append((t, seg_end))

        csv_rows.append({
            'segment_idx': idx,
            'time_sec': t,
            'final_category': flags['final_category'],
            'story': int(metrics.get('story_score', 0)),
            'action_raw': int(metrics.get('action_score', 0)),
            'action_eff': flags['action_effective_score'],
            'action_pen': flags['action_penalty'],
            'vocal_raw': int(metrics.get('sexual_vocal_score', 0)),
            'vocal_eff': flags['vocal_effective_score'],
            'vocal_sig': flags['vocal_signal_score'],
            'vocal_pen': flags['vocal_penalty'],
            'speech': int(metrics.get('speech_percent', 0)),
            'words': int(metrics.get('word_count', 0)),
            'moan': int(metrics.get('moan_percent', 0)),
            'breath': int(metrics.get('breath_percent', 0)),
            'scream': int(metrics.get('scream_percent', 0)),
            'dlg_lock': flags['dialogue_lock'],
            'hard_dlg': flags['hard_dialogue_lock'],
            'soft_dlg': flags['soft_dialogue_bias']
        })

        line = (
            f"SEGMENT {idx}/{total_steps} | final={flags['final_category']} | "
            f"story={metrics.get('story_score', 0)} | action_raw={metrics.get('action_score', 0)} | "
            f"action_eff={flags['action_effective_score']} | vocal_raw={metrics.get('sexual_vocal_score', 0)} | "
            f"vocal_eff={flags['vocal_effective_score']} | vocal_sig={flags['vocal_signal_score']} | "
            f"speech={metrics.get('speech_percent', 0)} | words={metrics.get('word_count', 0)} | "
            f"moan={metrics.get('moan_percent', 0)} | breath={metrics.get('breath_percent', 0)} | scream={metrics.get('scream_percent', 0)} | "
            f"dlg_lock={flags['dialogue_lock']} | hard_dlg={flags['hard_dialogue_lock']} | soft_dlg={flags['soft_dialogue_bias']} | keep={keep_flag}"
        )
        print(line, flush=True)
        print(f"PROGRESS:{int((idx / total_steps) * 55)}", flush=True)

    

    if not keep:
        print('No valid scenes found.', flush=True)
        print('PROGRESS:100', flush=True)
        return

    merged = merge_adjacent_keep_segments(keep)

    print('Analysis complete. Starting Export...', flush=True)
    resolve_py = None
    if 'AUTO-RENDER' in engine:
        resolve_py = pick_davinci_worker_python(cfg)
    snap_py = (resolve_py or '') if 'AUTO-RENDER' in engine else cfg.get('Settings', 'davinci_python_path', fallback='').strip()
    cfg_snapshot = {
        'export_engine': engine,
        'ffmpeg_nvenc_preset': cfg.get('Settings', 'ffmpeg_nvenc_preset', fallback='p4'),
        'export_bitrate_mode': cfg.get('Settings', 'export_bitrate_mode', fallback='default'),
        'export_manual_video_kbps': cfg.get('Settings', 'export_manual_video_kbps', fallback=''),
        'resolve_api_path': api_path,
        'davinci_python_path': snap_py,
    }
    write_autocut_checkpoint(vid, merged, fps, step, out_dir, cfg_snapshot, segment_results=segment_results)

    export_ok = run_export_only(vid, merged, fps, out_dir, cfg, resolve_py=resolve_py)

    print('PROGRESS:100', flush=True)
    if export_ok:
        print('Process successfully finished!', flush=True)
    else:
        print(
            'EXPORT_FAILED: Analyse + Checkpoint gespeichert (output/last_autocut_checkpoint.json). '
            'Temp-Dateien ggf. im Ausgabeordner — nicht gelöscht.',
            flush=True,
        )
        sys.exit(2)


if __name__ == '__main__':
    main()