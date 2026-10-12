#!/usr/bin/env python3
"""davinci_timeline against a small fake of the Resolve object model (no Resolve needed):
new timeline next to existing ones, clip reuse, frame math, nothing deleted."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import davinci_timeline as dt  # noqa: E402


class FakeClip:
    def __init__(self, path, frames=3000, fps='29.97'):
        self.props = {'File Path': str(path), 'Frames': str(frames), 'FPS': fps}
        self.set_calls = []

    def GetClipProperty(self, key):
        return self.props.get(key)

    def SetClipProperty(self, key, val):
        self.set_calls.append((key, val))
        self.props[key] = val
        return True


class FakeFolder:
    def __init__(self, clips=(), subs=()):
        self.clips, self.subs = list(clips), list(subs)

    def GetClipList(self):
        return self.clips

    def GetSubFolderList(self):
        return self.subs


class FakeItem:
    def __init__(self, end):
        self.end = end
        self.color = None

    def GetEnd(self):
        return self.end

    def SetClipColor(self, color):
        self.color = color
        return True


class FakeTimeline:
    def __init__(self, name, settings):
        self.name, self.settings, self.items = name, dict(settings), []
        self.track = {'video': [], 'audio': []}

    def GetItemListInTrack(self, kind, index):
        return self.track[kind] if index == 1 else []

    def GetName(self):
        return self.name

    def GetSetting(self, key):
        return self.settings.get(key)

    def SetSetting(self, key, val):
        self.settings[key] = val
        return True

    def GetStartFrame(self):
        return 108000

    def SetStartTimecode(self, _tc):
        return True


class FakeProject:
    def __init__(self, name, timelines=(), fps='24'):
        self.name = name
        self.settings = {'timelineFrameRate': fps}
        self.timelines = list(timelines)
        self.current = None
        self.pool = FakeMediaPool(self)
        self.deleted = []

    def GetName(self):
        return self.name

    def GetSetting(self, key):
        return self.settings.get(key)

    def SetSetting(self, key, val):
        if key == 'timelineFrameRate' and self.timelines:
            return False  # Resolve locks the rate once timelines exist
        self.settings[key] = val
        return True

    def GetTimelineCount(self):
        return len(self.timelines)

    def GetTimelineByIndex(self, i):
        return self.timelines[i - 1]

    def SetCurrentTimeline(self, tl):
        self.current = tl
        return True

    def GetMediaPool(self):
        return self.pool


class FakeMediaPool:
    def __init__(self, project):
        self.project = project
        self.root = FakeFolder()
        self.imported = []
        self.appends = []

    def GetRootFolder(self):
        return self.root

    def ImportMedia(self, paths):
        self.imported += paths
        clip = FakeClip(paths[0])
        self.root.clips.append(clip)
        return [clip]

    def CreateEmptyTimeline(self, name):
        tl = FakeTimeline(name, self.project.settings)
        self.project.timelines.append(tl)
        return tl

    def DeleteTimelines(self, tls):
        self.project.deleted += tls
        return True

    def AppendToTimeline(self, infos):
        # Like Resolve: without recordFrame the clips go to the end of the timeline, back to back
        tl = self.project.current
        out = []
        for info in infos:
            start = tl.items[-1]['_end'] if tl.items else tl.GetStartFrame()
            placed = dict(info, _start=start, _end=start + info['endFrame'] - info['startFrame'])
            tl.items.append(placed)
            self.appends.append(placed)
            v, a = FakeItem(placed['_end']), FakeItem(placed['_end'])
            tl.track['video'].append(v)
            tl.track['audio'].append(a)
            out.append(v)
        return out


class FakeResolve:
    def __init__(self):
        self.pages = []

    def OpenPage(self, page):
        self.pages.append(page)
        return True


def _patch(project, resolve, fps=29.97002997002997, wh=(1280, 720)):
    @contextmanager
    def no_com():
        yield

    dt.scripting_thread = no_com
    dt.connect_resolve = lambda **kw: (resolve, project, project.pool, project.pool.root)
    dt.probe_source_fps = lambda path: fps
    dt.probe_video_wh = lambda path: wh


def _video(tmp_path: Path) -> Path:
    p = Path(tmp_path) / 'My Clip.mp4'
    p.write_bytes(b'x')
    return p


def test_empty_project(tmp_path: Path):
    vid = _video(tmp_path)
    project, resolve = FakeProject('Edit'), FakeResolve()
    _patch(project, resolve)
    proj, name, n = dt.load_into_timeline(str(vid), [(10.0, 20.0), (30.5, 31.0)], log=lambda m: None)
    assert (proj, name, n) == ('Edit', 'My Clip – Scenecut', 2), (proj, name, n)
    assert project.settings['timelineFrameRate'] == '29.97', project.settings
    assert project.pool.imported == [str(vid).replace('\\', '/')]
    tl = project.timelines[0]
    assert project.current is tl and resolve.pages == ['edit']
    a, b = project.pool.appends
    assert (a['startFrame'], a['endFrame']) == (300, 599), a
    assert (b['startFrame'], b['endFrame']) == (914, 929), b
    assert 'recordFrame' not in a and 'recordFrame' not in b, 'Resolve must place the clips itself (no gaps)'
    assert b['_start'] == a['_end'], 'second scene must follow the first without a gap'
    assert tl.settings.get('useCustomSettings') == '1'
    assert tl.settings.get('timelineResolutionWidth') == '1280'


def test_existing_project_reuses_clip_and_keeps_timelines(tmp_path: Path):
    vid = _video(tmp_path)
    existing = [FakeTimeline('Rough cut', {'timelineFrameRate': '24'}),
                FakeTimeline('My Clip – Scenecut', {'timelineFrameRate': '24'})]
    project, resolve = FakeProject('Movie', timelines=existing, fps='24'), FakeResolve()
    clip = FakeClip(str(vid).replace('\\', '/'), fps='29.97')
    project.pool.root.subs.append(FakeFolder(clips=[clip]))
    _patch(project, resolve)
    logs = []
    _proj, name, n = dt.load_into_timeline(str(vid), [(0.0, 5.0)], log=logs.append)
    assert name == 'My Clip – Scenecut (2)', name
    assert project.pool.imported == [], 'clip already in the pool must not be imported again'
    assert clip.set_calls == [], 'clips the user already has must not be changed'
    assert project.timelines[:2] == existing and len(project.timelines) == 3
    assert project.deleted == []
    assert project.settings['timelineFrameRate'] == '24', 'project rate must stay as it is'
    assert project.timelines[2].settings['timelineFrameRate'] == '29.97', 'new timeline gets the video rate'
    assert project.pool.appends[0]['mediaPoolItem'] is clip


def test_full_length(tmp_path: Path):
    vid = _video(tmp_path)
    project, resolve = FakeProject('Edit'), FakeResolve()
    _patch(project, resolve, fps=25.0)
    _proj, name, n = dt.load_into_timeline(str(vid), [(10.0, 20.0), (30.0, 40.0)], log=lambda m: None,
                                           full_length=True, duration=60.0)
    assert name == 'My Clip – Scenecut (ganzes Video)', name
    parts = [(x['startFrame'], x['endFrame']) for x in project.pool.appends]
    assert parts == [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500)], parts
    assert n == 5
    tl = project.timelines[0]
    want = ['Orange', 'Green', 'Orange', 'Green', 'Orange']
    assert [i.color for i in tl.track['video']] == want and [i.color for i in tl.track['audio']] == want


def test_no_segments(tmp_path: Path):
    try:
        dt.load_into_timeline(str(_video(tmp_path)), [])
    except ValueError:
        return
    raise AssertionError('empty segment list must be refused')


def main() -> int:
    import shutil
    tests = [test_empty_project, test_existing_project_reuses_clip_and_keeps_timelines, test_full_length,
             test_no_segments]
    work = _ROOT / 'output' / '_davinci_timeline_test'
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
