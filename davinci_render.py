"""Prepare DaVinci's Deliver page with a bitrate limit (H.264, Resolve's own encoder).

Resolve 21 rejects SetRenderSettings({'VideoQuality': ...}) for every codec although it is
documented, so the limit goes through a render preset: save the current settings as a temporary
preset, export it, write the bitrate into the XML, import it as "Scenecut H264 <kbps>k".
H.264 NVIDIA / H.265 NVIDIA renders fail on Resolve 21 here, so the preset uses the native encoder.
"""
import os
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

from davinci_api import ResolveError, _ffprobe_json, to_forward

AUDIO_KBPS = 128
MIN_KBPS, MAX_KBPS = 300, 20000
# H.264 needs about this many times the bits of these codecs for the same picture
CODEC_FACTOR = {'hevc': 1.3, 'av1': 1.4, 'vp9': 1.3}
_TMP_PRESET = '_scenecut_tmp'


def source_video_kbps(video_path):
    """(video bitrate in kbit/s, codec name) of the first video stream; bitrate 0 if unknown."""
    p = _ffprobe_json(['-v', 'error', '-show_entries', 'format=bit_rate:stream=codec_type,codec_name,bit_rate',
                       '-of', 'json', str(video_path)]) or {}
    streams = p.get('streams') or []
    video = next((s for s in streams if s.get('codec_type') == 'video'), {})
    try:
        kbps = int(video.get('bit_rate') or 0) // 1000
        if not kbps:  # MKV/WebM carry no per-stream rate: total minus audio
            audio = sum(int(s.get('bit_rate') or 128000) for s in streams if s.get('codec_type') == 'audio')
            kbps = max(0, int((p.get('format') or {}).get('bit_rate') or 0) - audio) // 1000
    except (TypeError, ValueError):
        kbps = 0
    return kbps, video.get('codec_name') or ''


def auto_kbps(video_path):
    """H.264 bitrate that keeps about the source quality, or None if the source rate is unknown."""
    kbps, codec = source_video_kbps(video_path)
    if not kbps:
        return None
    kbps *= CODEC_FACTOR.get(codec, 1.0)
    return int(min(MAX_KBPS, max(MIN_KBPS, round(kbps / 50) * 50)))


def _decode_map(hexstr):
    """encoder_command_param_map: count, then UTF-16BE key/value strings, each length-prefixed."""
    b = bytes.fromhex(hexstr)
    pos, out = 4, {}

    def read():
        nonlocal pos
        n = struct.unpack('>I', b[pos:pos + 4])[0]
        s = b[pos + 4:pos + 4 + n].decode('utf-16-be')
        pos += 4 + n
        return s
    for _ in range(struct.unpack('>I', b[:4])[0]):
        key = read()
        out[key] = read()
    return out


def _encode_map(m):
    def pack(s):
        e = s.encode('utf-16-be')
        return struct.pack('>I', len(e)) + e
    return (struct.pack('>I', len(m)) + b''.join(pack(k) + pack(v) for k, v in m.items())).hex()


def _patch_preset(src, dst, kbps, audio_kbps):
    tree = ET.parse(src)
    root = tree.getroot()
    for tag, val in (('RecordFormatType', 'mp4'), ('RecordFormatSubType', 'avc1'), ('RecordQuality', '4')):
        el = root.find(tag)
        if el is not None:
            el.text = val
    extra = root.find('ExtraInfoMap')
    for el in extra if extra is not None else []:
        key, val = el.find('DbKey'), el.find('DbVal')
        if key is None or val is None:
            continue
        if key.text == 'h264_datarate':
            val.text = str(kbps)
        elif key.text == 'aud_rate':
            val.text = str(audio_kbps)
        elif key.text == 'encoder_command_param_map' and val.text:
            m = _decode_map(val.text)
            m.update(rc='vbr-bitrate', bitrate=str(kbps))
            val.text = _encode_map(m)
    tree.write(dst, encoding='UTF-8', xml_declaration=True)


def use_bitrate_preset(resolve, project, kbps, audio_kbps=AUDIO_KBPS):
    """Load (creating it first if needed) an H.264 preset limited to kbps. Returns its name."""
    name = f'Scenecut H264 {kbps}k'
    if name in (project.GetRenderPresetList() or []):
        if not project.LoadRenderPreset(name):
            raise ResolveError(f'Render-Preset „{name}“ ließ sich nicht laden.')
        return name
    project.SetCurrentRenderMode(1)  # single clip
    if not project.SetCurrentRenderFormatAndCodec('mp4', 'H264'):
        raise ResolveError('DaVinci kennt den Codec MP4/H.264 nicht.')
    if _TMP_PRESET in (project.GetRenderPresetList() or []):
        project.DeleteRenderPreset(_TMP_PRESET)
    tmp = tempfile.mkdtemp(prefix='scenecut_preset_')
    try:
        if not project.SaveAsNewRenderPreset(_TMP_PRESET):
            raise ResolveError('DaVinci konnte kein Render-Preset speichern.')
        try:
            exported = resolve.ExportRenderPreset(_TMP_PRESET, to_forward(os.path.join(tmp, 'export')))
        finally:
            project.DeleteRenderPreset(_TMP_PRESET)
        # Resolve treats the export path as a folder and writes <name>.xml into it
        src = next((os.path.join(d, f) for d, _s, files in os.walk(tmp) for f in files if f.endswith('.xml')), None)
        if not exported or src is None:
            raise ResolveError('DaVinci konnte das Render-Preset nicht exportieren.')
        dst = os.path.join(tmp, name + '.xml')  # file name = preset name
        _patch_preset(src, dst, kbps, audio_kbps)
        if not resolve.ImportRenderPreset(to_forward(dst)) or name not in (project.GetRenderPresetList() or []):
            raise ResolveError(f'DaVinci hat das Render-Preset „{name}“ nicht importiert.')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not project.LoadRenderPreset(name):
        raise ResolveError(f'Render-Preset „{name}“ ließ sich nicht laden.')
    return name


def prepare_render(resolve, project, video_path, kbps=None, target_dir=None, width=None, height=None):
    """Deliver page ready for the current timeline: bitrate preset, target folder, file name,
    resolution. kbps=None: derived from the source. Returns (preset name, kbps)."""
    kbps = kbps or auto_kbps(video_path)
    if not kbps:
        raise ResolveError('Bitrate der Quelle unbekannt – bitte eine Bitrate eintragen.')
    preset = use_bitrate_preset(resolve, project, int(kbps))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    settings = {'SelectAllFrames': True, 'TargetDir': to_forward(target_dir or os.path.dirname(video_path)),
                'CustomName': f'{stem}_scenecut'}
    if width and height:
        settings.update(FormatWidth=int(width), FormatHeight=int(height))
    if not project.SetRenderSettings(settings):
        raise ResolveError('DaVinci hat Zielordner/Dateiname nicht übernommen.')
    return preset, int(kbps)
