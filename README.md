# Scenecut NVIDIA

Desktop tool (CustomTkinter GUI) to analyze video segments and auto-export only selected scene categories (dialogue, action, vocal, etc.).

## Features
- Drag and drop source video in GUI
- Start / Pause / Stop from the source workflow
- Segment analysis with faster-whisper + OpenCV + YAMNet
- Export via FFmpeg (NVENC), or DaVinci Studio XML/EDL/AUTO-RENDER
- Retry export from checkpoint without full re-analysis

## Requirements
- Windows 10/11
- Python 3.10+
- FFmpeg in PATH (`ffmpeg`, `ffprobe`)
- Optional: NVIDIA GPU + CUDA-compatible stack for acceleration
- Optional: DaVinci Resolve for XML/EDL/AUTO-RENDER workflows

## Install
```bash
pip install -r requirements.txt
```

## Run GUI
```bash
python gui_nvidia.py
```


## Windows executable (optional, for GitHub Releases)

1. Install runtime deps: `pip install -r requirements.txt` (must include **tkinterdnd2**).
2. Install build tools: `pip install -r requirements-build.txt`
3. Run `build_gui_exe.bat` (or `pyinstaller --clean scenecut_gui.spec` from this folder).

Use the **same** `python` / venv for steps 1–3. If PyInstaller uses Python A but `tkinterdnd2` is only installed for Python B, the EXE will crash with `No module named 'tkinterdnd2'`.

Output: `dist/ScenecutNVIDIA.exe`. Distribute that file; on first run it creates `config_nvidia.ini` and an `output/` folder **next to the exe**.

**Drag & drop:** `tkinterdnd2` is bundled via PyInstaller. On some PCs one-file builds can still misbehave (AV, elevation, Explorer). If drag & drop fails, run `python gui_nvidia.py` from source, or switch the `.spec` to a **one-folder** (`COLLECT`) build — see [PyInstaller docs](https://pyinstaller.org).

**FFmpeg:** still must be installed separately and on `PATH` (the exe does not bundle ffmpeg).

## Config
- `config_nvidia.ini` contains safe, anonymized defaults.
- Add local DaVinci/Python paths only on your own machine.
- Keep personal paths out of git.

## Notes
- `yamnet.onnx` and `yamnet_class_map.csv` are included for audio-classification features.
- `output/` is runtime data and excluded from version control.


## Publish Checklist
- Review `LICENSE` author/year
- Create GitHub repo and upload this folder content
- Add screenshots or GIFs to README (optional, recommended)

## Credit
Thanks OpenAi for the Whisper-Modell
Google/Tensorflow for YAMNet-Model
`yamnet_class_map.csv`from this scource: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
`yamnet.onnx` form this source https://huggingface.co/zeropointnine/yamnet-onnx/tree/main and special thanks, without the easy access to an onnx file the project wouldn't be possible for me.