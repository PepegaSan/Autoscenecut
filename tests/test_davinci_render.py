#!/usr/bin/env python3
"""davinci_render against a small fake of Resolve's preset handling (no Resolve needed):
preset XML gets the bitrate, preset is reused, temp preset removed, auto bitrate per codec."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import davinci_render as dr  # noqa: E402

# Shape of a preset exported by Resolve 21 (trimmed)
TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<SyRecordInfo>
 <FormatWidth>1920</FormatWidth>
 <RecordFormatType>mov</RecordFormatType>
 <RecordFormatSubType>hvc1_nvenc</RecordFormatSubType>
 <RecordQuality>0</RecordQuality>
 <ExtraInfoMap>
  <Element><DbKey>aud_rate</DbKey><DbVal>320</DbVal></Element>
  <Element><DbKey>h264_datarate</DbKey><DbVal>4000</DbVal></Element>
  <Element><DbKey>encoder_command_param_map</DbKey><DbVal>{enc}</DbVal></Element>
 </ExtraInfoMap>
</SyRecordInfo>
"""


class FakeResolve:
    def __init__(self, project):
        self.project = project
        self.imported_xml = []

    def ExportRenderPreset(self, name, path):
        # Resolve makes a folder of the given path and puts <name>.xml inside
        os.makedirs(path, exist_ok=True)
        enc = dr._encode_map({'rc': 'quality', 'bitrate': '80000', 'preset': 'balanced'})
        with open(os.path.join(path, name + '.xml'), 'w', encoding='utf-8') as f:
            f.write(TEMPLATE.format(enc=enc))
        return name in self.project.presets

    def ImportRenderPreset(self, path):
        with open(path, encoding='utf-8') as f:
            self.imported_xml.append(f.read())
        self.project.presets.append(os.path.splitext(os.path.basename(path))[0])
        return True


class FakeProject:
    def __init__(self, presets=()):
        self.presets = list(presets)
        self.loaded = []
        self.render_settings = {}
        self.format_codec = None

    def GetRenderPresetList(self):
        return list(self.presets)

    def LoadRenderPreset(self, name):
        self.loaded.append(name)
        return name in self.presets

    def SaveAsNewRenderPreset(self, name):
        self.presets.append(name)
        return True

    def DeleteRenderPreset(self, name):
        self.presets.remove(name)
        return True

    def SetCurrentRenderMode(self, _mode):
        return True

    def SetCurrentRenderFormatAndCodec(self, fmt, codec):
        self.format_codec = (fmt, codec)
        return True

    def SetRenderSettings(self, settings):
        self.render_settings.update(settings)
        return True


def _extra(root):
    return {el.findtext('DbKey'): el.findtext('DbVal') for el in root.find('ExtraInfoMap')}


def test_map_roundtrip(_tmp):
    m = {'rc': 'vbr-bitrate', 'bitrate': '800', 'tuning': 'quality'}
    assert dr._decode_map(dr._encode_map(m)) == m


def test_creates_preset_with_bitrate(_tmp):
    project = FakeProject(presets=['H.264 Master'])
    resolve = FakeResolve(project)
    name = dr.use_bitrate_preset(resolve, project, 900)
    assert name == 'Scenecut H264 900k', name
    assert project.presets == ['H.264 Master', name], 'temp preset must be gone, new one added'
    assert project.loaded[-1] == name and project.format_codec == ('mp4', 'H264')
    root = ET.fromstring(resolve.imported_xml[0])
    assert root.findtext('RecordFormatType') == 'mp4' and root.findtext('RecordFormatSubType') == 'avc1'
    assert root.findtext('RecordQuality') == '4'
    extra = _extra(root)
    assert extra['h264_datarate'] == '900' and extra['aud_rate'] == '128', extra
    enc = dr._decode_map(extra['encoder_command_param_map'])
    assert enc == {'rc': 'vbr-bitrate', 'bitrate': '900', 'preset': 'balanced'}, enc


def test_reuses_existing_preset(_tmp):
    project = FakeProject(presets=['Scenecut H264 900k'])
    resolve = FakeResolve(project)
    assert dr.use_bitrate_preset(resolve, project, 900) == 'Scenecut H264 900k'
    assert resolve.imported_xml == [] and project.presets == ['Scenecut H264 900k']


def test_auto_kbps(_tmp):
    probes = {
        'hevc.mp4': {'streams': [{'codec_type': 'video', 'codec_name': 'hevc', 'bit_rate': '751586'},
                                 {'codec_type': 'audio', 'bit_rate': '128000'}]},
        'h264.mp4': {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'bit_rate': '2000000'}]},
        # MKV: no per-stream rate, total minus audio
        'x.mkv': {'format': {'bit_rate': '1128000'},
                  'streams': [{'codec_type': 'video', 'codec_name': 'h264'}, {'codec_type': 'audio'}]},
        'tiny.mp4': {'streams': [{'codec_type': 'video', 'codec_name': 'h264', 'bit_rate': '90000'}]},
        'none.mp4': {},
    }
    orig = dr._ffprobe_json
    dr._ffprobe_json = lambda args: probes[args[-1]]
    try:
        got = {k: dr.auto_kbps(k) for k in probes}
    finally:
        dr._ffprobe_json = orig
    assert got == {'hevc.mp4': 1000, 'h264.mp4': 2000, 'x.mkv': 1000, 'tiny.mp4': dr.MIN_KBPS, 'none.mp4': None}, got


def test_prepare_render(_tmp):
    project = FakeProject()
    resolve = FakeResolve(project)
    preset, kbps = dr.prepare_render(resolve, project, r'D:\Videos\My Clip.mp4', 700, None, 852, 478)
    assert (preset, kbps) == ('Scenecut H264 700k', 700)
    s = project.render_settings
    assert s['TargetDir'] == 'D:/Videos' and s['CustomName'] == 'My Clip_scenecut', s
    assert (s['FormatWidth'], s['FormatHeight']) == (852, 478) and s['SelectAllFrames'] is True


def main() -> int:
    import shutil
    tests = [test_map_roundtrip, test_creates_preset_with_bitrate, test_reuses_existing_preset, test_auto_kbps,
             test_prepare_render]
    work = _ROOT / 'output' / '_davinci_render_test'
    failed = 0
    for fn in tests:
        case = work / fn.__name__
        shutil.rmtree(case, ignore_errors=True)
        case.mkdir(parents=True)
        try:
            fn(case)
            print(f'PASS  {fn.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {fn.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
