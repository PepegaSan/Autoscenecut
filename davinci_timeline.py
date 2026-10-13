"""Load keep-segments straight into a DaVinci Resolve timeline (no rendering).

Two layouts: only the kept parts back to back, or the whole video cut at every segment edge
with kept parts coloured green and the rest orange (decide in DaVinci).

Uses the open project (a new "Scenecut" project only if Resolve shows the Project Manager),
reuses the clip if it is already in the media pool, and adds a new timeline next to the
existing ones – nothing in the project is deleted or overwritten. Connection handling
(launching Resolve, COM on worker threads, edition paths) comes from davinci_api.py.
"""
import os
import time

from davinci_render import prepare_render
from fast_scan import full_length_parts

from davinci_api import (
    ResolveError,
    connect_resolve,
    format_timeline_framerate_for_resolve,
    override_clip_fps,
    probe_source_fps,
    probe_video_wh,
    scene_source_frames,
    scripting_thread,
    set_project_master_before_import,
    sync_active_timeline_fps,
    to_forward,
)


def _norm(path):
    return os.path.normcase(os.path.normpath(str(path).replace('/', os.sep)))


def find_clip(folder, video_path):
    """Media pool item for this file anywhere below folder, or None."""
    want = _norm(video_path)
    for clip in folder.GetClipList() or []:
        try:
            if _norm(clip.GetClipProperty('File Path') or '') == want:
                return clip
        except Exception:
            continue
    for sub in folder.GetSubFolderList() or []:
        hit = find_clip(sub, video_path)
        if hit is not None:
            return hit
    return None


def timeline_names(project):
    names = set()
    for i in range(1, int(project.GetTimelineCount() or 0) + 1):
        tl = project.GetTimelineByIndex(i)
        if tl is not None:
            names.add(tl.GetName())
    return names


def unique_name(base, taken):
    if base not in taken:
        return base
    n = 2
    while f'{base} ({n})' in taken:
        n += 1
    return f'{base} ({n})'


KEEP_COLOR = 'Green'
DROP_COLOR = 'Orange'


def append_segments(media_pool, clip, segments, fps, clip_frames=0):
    """All segments in one AppendToTimeline call, without recordFrame: Resolve puts them back to
    back itself. (davinci_api.append_scenes_sequential places each at GetEnd()+1, but in
    Resolve 21 GetEnd() is already the first frame after the item – one empty frame per cut.)"""
    infos = []
    for start, end in segments:
        a, b = scene_source_frames(start, end, fps, clip_frames)
        infos.append({'mediaPoolItem': clip, 'startFrame': a, 'endFrame': b})
    items = media_pool.AppendToTimeline(infos) or []
    if len(items) != len(infos):
        raise ResolveError(f'DaVinci hat nur {len(items)} von {len(infos)} Szenen übernommen.')
    return items


def color_parts(timeline, keep_flags):
    """Clip colour per part on video and audio track 1 (green = Scenecut would keep it)."""
    for kind in ('video', 'audio'):
        items = timeline.GetItemListInTrack(kind, 1) or []
        if len(items) != len(keep_flags):
            continue
        for item, keep in zip(items, keep_flags):
            try:
                item.SetClipColor(KEEP_COLOR if keep else DROP_COLOR)
            except Exception:
                pass


def load_into_timeline(video_path, segments, log=print, timeline_name=None, full_length=False, duration=None,
                       deliver=True, render_kbps=None, render_dir=None):
    """Create a timeline from the keep-segments (seconds, source time).

    full_length=False: only the segments, back to back.
    full_length=True: the whole video (duration in seconds, default: clip length), cut at every
    segment edge, kept parts green, the rest orange.
    deliver=True also sets up the Deliver page: H.264 limited to render_kbps (None = derived from
    the source), into render_dir (default: next to the video). Nothing is queued or rendered.
    Call from any thread. Returns (project name, timeline name, number of clips).
    """
    if not segments:
        raise ValueError('Keine Szenen zum Laden.')
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    fps = probe_source_fps(video_path)
    rate = format_timeline_framerate_for_resolve(fps)
    width, height = probe_video_wh(video_path)

    with scripting_thread():
        log('Verbinde mit DaVinci Resolve …')
        resolve, project, media_pool, root = connect_resolve(
            status_callback=log, auto_launch=True, create_scratch_project_name='Scenecut')
        project_name = project.GetName()

        # Project frame rate can only change while the project has no timelines
        if int(project.GetTimelineCount() or 0) == 0:
            set_project_master_before_import(project, rate, fps, width or 1920, height or 1080)

        clip = find_clip(root, video_path)
        if clip is None:
            log('Importiere Video in den Media Pool …')
            clips = media_pool.ImportMedia([to_forward(video_path)])
            if not clips:
                raise ResolveError(f'DaVinci konnte das Video nicht importieren: {video_path}')
            clip = clips[0]
            time.sleep(0.25)
            # Resolve sometimes tags clips with the wrong rate (23.976 instead of 29.97);
            # only for clips imported here, never for clips the user already works with
            override_clip_fps(clip, fps, rate)

        stem = os.path.splitext(os.path.basename(video_path))[0]
        base = timeline_name or (f'{stem} – Scenecut (ganzes Video)' if full_length else f'{stem} – Scenecut')
        name = unique_name(base, timeline_names(project))
        timeline = media_pool.CreateEmptyTimeline(name)
        if timeline is None:
            raise ResolveError('DaVinci konnte keine neue Timeline anlegen.')
        project.SetCurrentTimeline(timeline)
        # Own frame rate + resolution for this timeline, in case the project uses others
        try:
            timeline.SetSetting('useCustomSettings', '1')
        except Exception:
            pass
        sync_active_timeline_fps(project, timeline, rate, fps, skip_if_close=True)
        if width and height:
            for key, val in (('timelineResolutionWidth', width), ('timelineResolutionHeight', height)):
                try:
                    timeline.SetSetting(key, str(val))
                except Exception:
                    pass
        try:
            tl_fps = float(str(timeline.GetSetting('timelineFrameRate')).replace(',', '.'))
        except (TypeError, ValueError):
            tl_fps = fps
        if abs(tl_fps - fps) > 0.06:
            # Source frames below stay correct; Resolve just plays the clip at the timeline rate
            log(f'Hinweis: Timeline läuft mit {tl_fps:g} fps, das Video hat {fps:g} fps.')

        try:
            clip_frames = int(clip.GetClipProperty('Frames') or 0)
        except (TypeError, ValueError):
            clip_frames = 0
        if full_length:
            total = duration if duration else (clip_frames / fps if clip_frames else segments[-1][1])
            # Parts shorter than one frame would turn into overlapping 1-frame clips
            parts = [p for p in full_length_parts(segments, total) if round(p[1] * fps) > round(p[0] * fps)]
            log(f'Lege das ganze Video mit {len(parts) - 1} Schnitten in die Timeline „{name}“ …')
            n = len(append_segments(media_pool, clip, [(a, b) for a, b, _k in parts], fps, clip_frames))
            color_parts(timeline, [k for _a, _b, k in parts])
        else:
            log(f'Lege {len(segments)} Szenen in die Timeline „{name}“ …')
            n = len(append_segments(media_pool, clip, segments, fps, clip_frames))
        project.SetCurrentTimeline(timeline)
        if deliver:
            try:
                preset, kbps = prepare_render(resolve, project, video_path, render_kbps, render_dir, width, height)
                log(f'Deliver vorbereitet: {preset} (Video max. {kbps} kbit/s, Audio 128 kbit/s).')
            except Exception as e:  # the timeline is what counts
                log(f'Hinweis: Render-Einstellungen nicht gesetzt – {e}')
        try:
            resolve.OpenPage('edit')
        except Exception:
            pass
    what = f'{n} Abschnitten (grün = behalten)' if full_length else f'{n} Szenen'
    log(f'Fertig: Timeline „{name}“ mit {what} in Projekt „{project_name}“.')
    return project_name, name, n
