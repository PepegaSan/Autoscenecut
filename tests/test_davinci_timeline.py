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
    project.render_calls = []

    def fake_prepare(_resolve, proj, path, kbps, target_dir, w, h, name=None):
        proj.render_calls.append((path, kbps, target_dir, w, h))
        proj.render_name = name
        return f'Scenecut H264 {kbps or 1000}k', kbps or 1000
    dt.prepare_render = fake_prepare


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


def test_deliver_setup(tmp_path: Path):
    vid = _video(tmp_path)
    project, resolve = FakeProject('Edit'), FakeResolve()
    _patch(project, resolve, wh=(852, 478))
    logs = []
    dt.load_into_timeline(str(vid), [(0.0, 5.0)], log=logs.append, render_kbps=800, render_dir='E:/out')
    assert project.render_calls == [(str(vid), 800, 'E:/out', 852, 478)], project.render_calls
    assert any('Scenecut H264 800k' in m for m in logs), logs

    def broken(*_a):
        raise RuntimeError('no preset')
    dt.prepare_render = broken
    logs.clear()
    _p, _name, n = dt.load_into_timeline(str(vid), [(0.0, 5.0)], log=logs.append)
    assert n == 1 and any('Render-Einstellungen nicht gesetzt' in m for m in logs), 'timeline must survive'

    _patch(project, resolve)
    dt.load_into_timeline(str(vid), [(0.0, 5.0)], log=lambda m: None, deliver=False)
    assert project.render_calls == [], 'deliver=False must leave the Deliver page alone'


def test_excerpt(tmp_path: Path):
    vid = _video(tmp_path)
    project, resolve = FakeProject('Edit'), FakeResolve()
    _patch(project, resolve)
    _proj, name, n = dt.load_into_timeline(str(vid), [(725.0, 790.0), (800.0, 870.4)], log=lambda m: None,
                                           excerpt=(725.0, 870.4))
    assert name == 'My Clip – Ausschnitt 12m05s-14m30s' and n == 2, (name, n)
    assert project.render_name == 'My Clip_scenecut_12m05s-14m30s', 'render must not overwrite the whole cut'
    _proj, name, _n = dt.load_into_timeline(str(vid), [(0.0, 5.0)], log=lambda m: None)
    assert name == 'My Clip – Scenecut' and project.render_name is None, (name, project.render_name)


class EditedItem:
    """Timeline item as the user left it in Resolve."""
    def __init__(self, clip, src_start, frames, color='', enabled=True, start_getter=True):
        self.clip, self.src_start, self.frames = clip, src_start, frames
        self.color, self.enabled = color, enabled
        if not start_getter:  # older Resolve: only GetLeftOffset
            self.GetSourceStartFrame = None

    def GetMediaPoolItem(self):
        return self.clip

    def GetSourceStartFrame(self):
        return self.src_start

    def GetLeftOffset(self):
        return self.src_start

    def GetDuration(self):
        return self.frames

    def GetClipColor(self):
        return self.color

    def GetClipEnabled(self):
        return self.enabled


class EditedTimeline(FakeTimeline):
    def __init__(self, name, tracks):
        super().__init__(name, {})
        self.tracks = tracks

    def GetTrackCount(self, kind):
        return len(self.tracks) if kind == 'video' else 0

    def GetItemListInTrack(self, kind, index):
        return self.tracks[index - 1] if kind == 'video' else []


def test_read_back(tmp_path: Path):
    vid = _video(tmp_path)
    mine = FakeClip(str(vid).replace('\\', '/'), frames=3000)
    other = FakeClip('E:/other.mp4', frames=3000)
    # 25 fps. Full-length timeline after editing: green 0–4 s, green 4–8 s deleted, orange
    # 8–10 s split and only its first half kept, green 10–12 s disabled, then 12–16 s split into
    # two touching clips; on track 2 another video and 20–21 s of this one. Colour doesn't count.
    track1 = [EditedItem(mine, 0, 100, 'Green'), EditedItem(mine, 200, 25, 'Orange'),
              EditedItem(mine, 250, 50, 'Green', enabled=False),
              EditedItem(mine, 300, 50), EditedItem(mine, 350, 50, start_getter=False)]
    track2 = [EditedItem(other, 0, 500), EditedItem(mine, 500, 25)]
    tl = EditedTimeline('My Clip – Scenecut (ganzes Video)', [track1, track2])
    spans, found = dt.timeline_cut(tl, str(vid), 25.0)
    assert spans == [(0.0, 4.0), (8.0, 9.0), (12.0, 16.0), (20.0, 21.0)], spans
    assert found == 6, found

    # A start counted from a timecode (outside the clip) falls back to GetLeftOffset
    odd = EditedItem(mine, 40, 25)
    odd.GetSourceStartFrame = lambda: 86400 * 25
    assert dt.timeline_cut(EditedTimeline('x', [[odd]]), str(vid), 25.0)[0] == [(1.6, 2.6)]

    # Which timeline: the open one if it has the video, otherwise the newest Scenecut one
    project = FakeProject('Edit', timelines=[EditedTimeline('Rough cut', [[EditedItem(other, 0, 10)]]),
                                             EditedTimeline('My Clip – Scenecut', [[EditedItem(mine, 0, 10)]]),
                                             tl])
    project.GetCurrentTimeline = lambda: project.timelines[0]
    assert dt.find_timeline(project, str(vid)) is tl
    project.GetCurrentTimeline = lambda: project.timelines[1]
    assert dt.find_timeline(project, str(vid)) is project.timelines[1]
    # An open excerpt timeline is not the cut of the whole video
    excerpt = EditedTimeline('My Clip – Ausschnitt 12m05s-14m30s', [[EditedItem(mine, 0, 10)]])
    project.GetCurrentTimeline = lambda: excerpt
    assert dt.find_timeline(project, str(vid)) is tl
    project.GetCurrentTimeline = lambda: project.timelines[1]

    # Through read_timeline_cut: read only, Resolve must not be started for it
    calls = []
    _patch(project, FakeResolve(), fps=25.0)

    def connect(**kw):
        calls.append(kw)
        return FakeResolve(), project, project.pool, project.pool.root
    dt.connect_resolve = connect
    proj, name, spans, n = dt.read_timeline_cut(str(vid), log=lambda m: None)
    assert (proj, name, spans, n) == ('Edit', 'My Clip – Scenecut', [(0.0, 0.4)], 1), (proj, name, spans, n)
    assert calls[0]['auto_launch'] is False and calls[0]['create_scratch_project_name'] is None, calls
    project.GetCurrentTimeline = lambda: None
    project.timelines = project.timelines[:1]
    try:
        dt.read_timeline_cut(str(vid), log=lambda m: None)
    except dt.ResolveError:
        pass
    else:
        raise AssertionError('no timeline with the video must raise')


def test_no_segments(tmp_path: Path):
    try:
        dt.load_into_timeline(str(_video(tmp_path)), [])
    except ValueError:
        return
    raise AssertionError('empty segment list must be refused')


def main() -> int:
    import shutil
    tests = [test_empty_project, test_existing_project_reuses_clip_and_keeps_timelines, test_full_length,
             test_deliver_setup, test_excerpt, test_read_back, test_no_segments]
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
