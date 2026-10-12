# Scenecut

Cuts long videos down to the scenes that matter – by motion and sound – and loads the result
straight into a DaVinci Resolve timeline (or writes XML / EDL / MP4).

One pass over picture and sound: about **1 minute per 100 minutes of video** on an RTX-class
GPU, then every slider change shows its result instantly.

![Scenecut](images/scenecut.png)

## Requirements

- Windows 10/11, Python 3.12 (the version DaVinci Resolve's scripting expects)
- FFmpeg on `PATH` (`ffmpeg`, `ffprobe`, and `ffplay` for the preview)
- Optional: NVIDIA GPU – decoding uses NVDEC, MP4 export uses NVENC (falls back to CPU / x264)
- Optional: DaVinci Resolve **Studio** for loading timelines directly
  (*Preferences → System → General → External scripting using = Local*, then restart Resolve)

**Install:** run `install.bat` – it finds (or installs via winget) Python 3.12, creates a
`.venv` in this folder with the packages from `requirements.txt` and checks for FFmpeg.

**Start:** `start.bat` (no console window). Videos dropped onto `start.bat` are added right away.

Manual alternative: `pip install -r requirements.txt`, then `python scenecut_app.py`.

## Using it

1. **Drop videos or folders** anywhere into the window. They are analyzed one after the other;
   a second open is instant (analysis cache in `%LOCALAPPDATA%\Scenecut\cache`).
2. **Pick a preset** – *Highlights*, *Standard* (automatic per video), *Großzügig* – or set the
   categories yourself: motion / action, vocal sounds, dialogue (cut out / ignore / keep), music.
   The *Feinschliff* tab has minimum scene length, gap closing, padding and smoothing.
3. **Read the timeline**: the *Bleibt* lane and the green background show exactly what is kept.
   Click to play from there (ffplay, optionally muted).
4. **Fix outliers yourself**: drag over a range or right-click a stretch → *always keep* /
   *always cut*. These corrections win over all sliders and are saved per video.
5. **Export**
   - *Direkt in DaVinci laden*: new timeline in the open Resolve project (Resolve is started if
     needed; existing timelines are never touched)
   - DaVinci timeline file (FCP7 XML) or EDL
   - finished MP4
   - Timeline content: only the kept parts, or the whole video cut at every scene edge
     (direct load colours kept parts green, the rest orange)

Settings are remembered in `scenecut_settings.json` next to the app.

## How it decides

- **Picture**: one sequential ffmpeg decode into 160×90 grayscale at 10 fps. Per frame the
  difference to the previous one, with camera motion removed by phase correlation – pans count
  less than motion in the picture. Cuts are isolated difference spikes.
- **Sound**: the whole track through YAMNet (`yamnet.onnx`) – speech, vocal sounds, music,
  silence per 0.48 s.
- **Decision**: motion threshold as a percentile of the video itself (automatic split by
  default), hysteresis, gap closing, minimum length, edges snapped to cuts, then the manual
  corrections on top.

## Command line

```bash
python fast_scan.py VIDEO [--motion-level 60] [--speech remove|ignore|keep] [--export xml,edl,mp4]
```

## Files

| File | |
|---|---|
| `scenecut_app.py` | the app (tkinter + tkinterdnd2) |
| `fast_scan.py` | analysis, decision, MP4 export, manual corrections |
| `davinci_timeline.py` | loads segments into a Resolve timeline |
| `davinci_api.py` | Resolve connection (vendored from the Hail Mary project) |
| `timeline_export.py` | FCP7 XML and CMX EDL files |
| `yamnet.onnx`, `yamnet_class_map.csv` | sound classification model |

## Tests

```bash
python tests/test_fast_scan.py
python tests/test_davinci_timeline.py
python tests/test_timeline_export.py
```

No GPU or Resolve needed; the Resolve tests use a small fake of its object model.
