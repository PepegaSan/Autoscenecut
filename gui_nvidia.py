import os
import re
import sys
import time
import json
import shutil
from datetime import datetime
import math
import threading
import subprocess
import configparser
import statistics
import cv2
import torch  # noqa: F401 — in der EXE: Methode A / Scene-Analyse laden analyzer nur lazy; sonst fehlt torch im Bundle
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox, filedialog

from theme_palette import PALETTE_DARK, PALETTE_LIGHT
from tkinterdnd2 import TkinterDnD, DND_FILES


def _is_frozen():
    return bool(getattr(sys, 'frozen', False)) and bool(getattr(sys, '_MEIPASS', None))


def _bundle_dir():
    if _is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _exe_dir():
    if _is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


# Frozen: writable config + output next to .exe; worker subprocess cwd = same dir.
# Dev: everything next to gui_nvidia.py
SCRIPT_DIR = _exe_dir()
BUNDLE_DIR = _bundle_dir()
CFG_PATH = os.path.join(SCRIPT_DIR, 'config_nvidia.ini')


def _ensure_config_file():
    if os.path.isfile(CFG_PATH):
        return
    for name in ('config_nvidia.example.ini', 'config_nvidia.ini'):
        src = os.path.join(BUNDLE_DIR, name)
        if os.path.isfile(src):
            try:
                shutil.copyfile(src, CFG_PATH)
                return
            except OSError:
                pass
# CREATE_NO_WINDOW wird NUR noch für Taskkill verwendet, um DaVinci-Abstürze zu vermeiden
CREATE_NO_WINDOW = 0x08000000

ctk.set_default_color_theme('blue')

BTN_RADIUS = 10
BTN_H = 36
FONT_SECTION = ('Segoe UI Semibold', 14)
FONT_UI = ('Segoe UI', 12)
FONT_HINT = ('Segoe UI', 11)
FONT_BTN = ('Segoe UI Semibold', 10)
FONT_BTN_PRIMARY = ('Segoe UI Black', 11)
FONT_APP_TITLE = ('Segoe UI Semibold', 17)

# Delay writing config_nvidia.ini while sliders are dragged (ms)
SAVE_CFG_DEBOUNCE_MS = 450


def clamp_int(v, low=0, high=100):
    return max(low, min(high, int(round(v))))


def percentile(values, q):
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


class DnD_CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class VideoPlayerWindow(ctk.CTkToplevel):
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.parent_gui = parent
        self.video_path = video_path
        self.title('Scene Selection & AI Analysis')
        self.geometry('980x800')
        self.attributes('-topmost', True)

        _p = parent._pal
        self.configure(fg_color=_p['bg'])

        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.current_frame = 0
        self.is_playing = False
        self.last_update_time = time.time()

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self, fg_color='black')
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.lbl_video = ctk.CTkLabel(self.video_frame, text='')
        self.lbl_video.pack(fill='both', expand=True)

        self.controls = ctk.CTkFrame(self, fg_color=_p['bg'])
        self.controls.grid(row=1, column=0, padx=10, pady=(0, 10), sticky='ew')
        self.controls.columnconfigure(1, weight=1)

        self.lbl_time = ctk.CTkLabel(self.controls, text='00:00 / 00:00', font=FONT_UI, text_color=_p['text'])
        self.lbl_time.grid(row=0, column=0, padx=10, pady=5)

        self.slider = ctk.CTkSlider(
            self.controls,
            from_=0,
            to=max(1, self.total_frames - 1),
            command=self.set_frame,
            progress_color=_p['cyan'],
            button_color=_p['panel'],
            button_hover_color=_p['panel_elev'],
            fg_color=_p['panel_elev'],
        )
        self.slider.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky='ew')
        self.slider.set(0)

        nav = ctk.CTkFrame(self.controls, fg_color=_p['bg'])
        nav.grid(row=1, column=0, columnspan=3, pady=5)
        ctk.CTkButton(nav, text='⏪ -10s', command=lambda: self.jump(-10), **parent._button_kw('ghost', width=60)).pack(side='left', padx=5)
        self.btn_play = ctk.CTkButton(nav, text='▶ Play', command=self.toggle_play, **parent._button_kw('primary', width=100))
        self.btn_play.pack(side='left', padx=10)
        ctk.CTkButton(nav, text='⏩ +10s', command=lambda: self.jump(10), **parent._button_kw('ghost', width=60)).pack(side='left', padx=5)

        self.lbl_result = ctk.CTkLabel(
            self.controls,
            text='Scrub to a scene and analyze it.',
            font=FONT_SECTION,
            text_color=_p['text'],
            justify='left',
        )
        self.lbl_result.grid(row=2, column=0, columnspan=3, pady=10)

        frame_analyze = ctk.CTkFrame(self.controls, fg_color=_p['bg'])
        frame_analyze.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        ctk.CTkButton(frame_analyze, text='Analyze → Story/Dialogue', command=lambda: self.analyze('dialogue'), **parent._button_kw('primary', width=170)).pack(side='left', padx=5)
        ctk.CTkButton(frame_analyze, text='Analyze → Action', command=lambda: self.analyze('action'), **parent._button_kw('warning', width=140)).pack(side='left', padx=5)
        ctk.CTkButton(frame_analyze, text='Analyze → Moan/Breath', command=lambda: self.analyze('vocal'), **parent._button_kw('danger', width=170)).pack(side='left', padx=5)

        self.draw_frame()

    def loop(self):
        if self.is_playing:
            if time.time() - self.last_update_time >= (1.0 / max(1.0, self.fps)):
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                self.last_update_time = time.time()
                self.draw_frame()
            if self.current_frame >= self.total_frames - 1:
                self.toggle_play()
            self.after(5, self.loop)

    def draw_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w = max(10, self.lbl_video.winfo_width())
        h = max(10, int(w * (self.height / max(1, self.width))))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.lbl_video.configure(image=ctk_img)
        self.lbl_video.image = ctk_img
        self.slider.set(self.current_frame)
        sec = int(self.current_frame / max(1.0, self.fps))
        total_sec = int(self.total_frames / max(1.0, self.fps))
        self.lbl_time.configure(text=f'{sec//60:02d}:{sec%60:02d} / {total_sec//60:02d}:{total_sec%60:02d}')

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text='⏸ Pause', **self.parent_gui._button_kw('warning', width=100))
        else:
            self.btn_play.configure(text='▶ Play', **self.parent_gui._button_kw('primary', width=100))
        if self.is_playing:
            self.last_update_time = time.time()
            self.loop()

    def jump(self, delta_sec):
        self.current_frame = max(0, min(self.total_frames - 1, self.current_frame + int(delta_sec * self.fps)))
        if not self.is_playing:
            self.draw_frame()

    def set_frame(self, val):
        self.current_frame = int(val)
        if not self.is_playing:
            self.draw_frame()

    def analyze(self, category):
        if self.is_playing:
            self.toggle_play()
        sec = int(self.current_frame / max(1.0, self.fps))
        _p = self.parent_gui._pal
        self.lbl_result.configure(text=f'Analyzing {category}... please wait.', text_color=_p['accent_warn'])
        self.update()
        from analyzer_nvidia import analyze_segment
        seg_len = max(5, min(60, int(float(self.parent_gui.g('Settings', 'interval_seconds', '20') or 20))))
        m = analyze_segment(self.video_path, sec, seg_len)
        story = m.get('story_score', 0)
        action = m.get('action_score', 0)
        vocal = m.get('sexual_vocal_score', 0)
        speech = m.get('speech_percent', 0)
        res = (
            f"Story/Dialogue: {story} | Action: {action} | Moan/Breath (vocal): {vocal} | Speech: {speech}\n"
            f"Words: {m.get('word_count', 0)} | WPM: {m.get('wpm', '—')} (target {m.get('target_wpm', '—')}) | word_score: {m.get('word_score', '—')}\n"
            f"Moan: {m.get('moan_percent', 0)} | Breath: {m.get('breath_percent', 0)} | Scream: {m.get('scream_percent', 0)}"
        )
        self.lbl_result.configure(text=res, text_color=self.parent_gui._pal['accent_positive'])
        pg = self.parent_gui
        if category == 'dialogue':
            pg.set_entry_value(pg.ed_story, max(0, int(story * 0.85)))
        elif category == 'action':
            pg.set_entry_value(pg.ed_action, max(0, int(action * 0.90)))
        elif category == 'vocal':
            pg.set_entry_value(pg.ed_vocal, max(0, int(vocal * 0.90)))

    def destroy(self):
        self.is_playing = False
        self.cap.release()
        super().destroy()


class PathsSettingsDialog(ctk.CTkToplevel):
    """Paths + DaVinci worker fields (moved from Export tab); persists via parent.save_cfg()."""

    def __init__(self, gui: 'NvidiaGUI'):
        super().__init__(gui)
        self.gui = gui
        p = gui._pal
        self.title('Settings — Paths & DaVinci')
        self.geometry('640x420')
        self.minsize(480, 360)

        self._card = ctk.CTkFrame(self, corner_radius=10, border_width=1)
        self._card.pack(fill='both', expand=True, padx=14, pady=14)

        self._lbl_title = ctk.CTkLabel(self._card, text='Paths & DaVinci', font=FONT_SECTION)
        self._lbl_title.pack(anchor='w', padx=12, pady=(10, 6))

        self._lbl_api = ctk.CTkLabel(self._card, text='DaVinci Resolve API (Modules folder, optional)', font=FONT_UI)
        self._lbl_api.pack(anchor='w', padx=12, pady=(8, 2))
        self._e_api = ctk.CTkEntry(self._card, textvariable=gui._var_resolve_api, font=FONT_UI)
        self._e_api.pack(fill='x', padx=12, pady=(0, 8))

        self._lbl_py = ctk.CTkLabel(
            self._card,
            text='Python worker for AUTO-RENDER (.exe needs f4python 3.12 or compatible)',
            font=FONT_UI,
        )
        self._lbl_py.pack(anchor='w', padx=12, pady=(4, 2))
        self._lbl_py_hint = ctk.CTkLabel(
            self._card,
            text='Standalone EXE is not Python — point to Resolve f4python\\3.12\\bin\\python.exe or any 3.12 that imports DaVinciResolveScript with the API path above.',
            font=FONT_HINT,
            justify='left',
            wraplength=560,
        )
        self._lbl_py_hint.pack(anchor='w', padx=12, pady=(0, 4))
        self._e_py = ctk.CTkEntry(self._card, textvariable=gui._var_davinci_py, font=FONT_UI)
        self._e_py.pack(fill='x', padx=12, pady=(0, 8))

        self._lbl_out = ctk.CTkLabel(self._card, text='Output folder (blank = next to source video)', font=FONT_UI)
        self._lbl_out.pack(anchor='w', padx=12, pady=(4, 2))
        out_row = ctk.CTkFrame(self._card, fg_color='transparent')
        out_row.pack(fill='x', padx=12, pady=(0, 12))
        out_row.grid_columnconfigure(0, weight=1)
        self._e_out = ctk.CTkEntry(out_row, textvariable=gui._var_output_path, font=FONT_UI)
        self._e_out.grid(row=0, column=0, sticky='ew', padx=(0, 8))
        self._btn_browse = ctk.CTkButton(out_row, text='Browse…', command=gui.browse_output_dir, **gui._button_kw('ghost', width=100))
        self._btn_browse.grid(row=0, column=1, sticky='e')

        row_btn = ctk.CTkFrame(self._card, fg_color='transparent')
        row_btn.pack(fill='x', padx=12, pady=(0, 14))
        self._btn_save = ctk.CTkButton(row_btn, text='Save & close', command=self._save_close, **gui._button_kw('primary', width=140))
        self._btn_save.pack(side='right')

        self.protocol('WM_DELETE_WINDOW', self._save_close)
        self._apply_palette()

    def _apply_palette(self):
        p = self.gui._pal
        self.configure(fg_color=p['bg'])
        self._card.configure(fg_color=p['panel_elev'], border_color=p['border'])
        for w in (
            self._lbl_title,
            self._lbl_api,
            self._lbl_py,
            self._lbl_out,
        ):
            w.configure(text_color=p['text'])
        self._lbl_py_hint.configure(text_color=p['muted'])
        for e in (self._e_api, self._e_py, self._e_out):
            e.configure(fg_color=p['panel'], border_color=p['border'], text_color=p['text'])
        self._btn_browse.configure(**self.gui._button_kw('ghost', width=100))
        self._btn_save.configure(**self.gui._button_kw('primary', width=140))

    def _save_close(self):
        self.gui.save_cfg()
        self.destroy()


class NvidiaGUI(DnD_CTk):
    def __init__(self):
        super().__init__()
        self.video_path = ''
        self.current_process = None
        self.is_paused = False
        self.cfg = configparser.ConfigParser()
        _ensure_config_file()
        self.cfg.read(CFG_PATH, encoding='utf-8')

        self._theme_muted_labels = []
        self._theme_primary_labels = []
        self._paths_dialog = None
        self._save_cfg_after_id = None
        _ga = (self.g('Settings', 'gui_appearance', 'dark') or 'dark').strip().lower()
        if _ga == 'light':
            self._pal = dict(PALETTE_LIGHT)
            ctk.set_appearance_mode('Light')
        else:
            self._pal = dict(PALETTE_DARK)
            ctk.set_appearance_mode('Dark')
        self._appearance = ctk.StringVar(value='light' if _ga == 'light' else 'dark')

        self._var_resolve_api = ctk.StringVar(value=self.g('Settings', 'resolve_api_path', ''))
        self._var_davinci_py = ctk.StringVar(value=self.g('Settings', 'davinci_python_path', ''))
        self._var_output_path = ctk.StringVar(value=self.g('Settings', 'output_path', ''))

        self.title('Autocut NVIDIA Control Center')
        self.geometry('820x1200')
        self.minsize(520, 420)
        self.configure(fg_color=self._pal['bg'])

        self._top_bar = ctk.CTkFrame(self, corner_radius=0, height=52, fg_color=self._pal['panel'])
        self._top_bar.pack(fill='x', padx=0, pady=0)
        self._top_bar.pack_propagate(False)
        _tb_inner = ctk.CTkFrame(self._top_bar, fg_color='transparent')
        _tb_inner.pack(fill='both', expand=True, padx=14, pady=8)
        self._lbl_app_title = ctk.CTkLabel(
            _tb_inner,
            text='Autocut NVIDIA',
            font=FONT_APP_TITLE,
            text_color=self._pal['text'],
        )
        self._lbl_app_title.pack(side='left')

        self._btn_settings = ctk.CTkButton(
            _tb_inner,
            text='⚙',
            command=self.open_paths_settings,
            **self._button_kw('icon_secondary', width=44, height=40, font=('Segoe UI', 20)),
        )
        self._btn_settings.pack(side='right')
        self._appearance_seg = ctk.CTkSegmentedButton(
            _tb_inner,
            values=['dark', 'light'],
            variable=self._appearance,
            command=self._on_appearance,
            font=FONT_HINT,
        )
        self._appearance_seg.pack(side='right', padx=(0, 10))

        self._scroll_body = ctk.CTkScrollableFrame(
            self,
            fg_color=self._pal['bg'],
            bg_color=self._pal['bg'],
            scrollbar_fg_color=self._pal['panel'],
            scrollbar_button_color=self._pal['cyan_dim'],
            scrollbar_button_hover_color=self._pal['cyan'],
        )
        self._scroll_body.pack(fill='both', expand=True, padx=12, pady=(0, 12))

        self.tabs = ctk.CTkTabview(
            self._scroll_body,
            fg_color=self._pal['panel'],
            bg_color=self._pal['bg'],
            border_color=self._pal['border'],
            segmented_button_fg_color=self._pal['panel_elev'],
            segmented_button_selected_color=self._pal['cyan_dim'],
            segmented_button_selected_hover_color=self._pal['cyan'],
            segmented_button_unselected_color=self._pal['panel_elev'],
            segmented_button_unselected_hover_color=self._pal['border'],
            text_color=self._pal['text'],
        )
        self.tabs.pack(fill='x', expand=False, padx=8, pady=(0, 8))
        self.tab_source = self.tabs.add('1. Source')
        self.tab_thresholds = self.tabs.add('2. Thresholds')
        self.tab_export = self.tabs.add('3. Categories & Export')

        self.build_source_tab()
        self.build_thresholds_tab()
        self.build_export_tab()

        self.frame_progress = ctk.CTkFrame(self._scroll_body, fg_color=self._pal['bg'])
        self.frame_progress.pack(fill='x', padx=8, pady=(0, 5))
        self.lbl_status = ctk.CTkLabel(self.frame_progress, text='Ready', font=FONT_UI)
        self.lbl_status.pack(anchor='w')
        self.progress = ctk.CTkProgressBar(self.frame_progress)
        self.progress.pack(fill='x', pady=6)
        self.progress.set(0)

        self.log_frame = ctk.CTkFrame(
            self._scroll_body,
            fg_color=self._pal['panel'],
            corner_radius=10,
            border_width=1,
            border_color=self._pal['border'],
        )
        self.log_frame.pack(fill='x', expand=False, padx=8, pady=(0, 8))
        self._lbl_log_title = ctk.CTkLabel(self.log_frame, text='Live Segment Log', font=FONT_SECTION)
        self._lbl_log_title.pack(anchor='w', padx=10, pady=(8, 4))
        self.txt_log = ctk.CTkTextbox(self.log_frame, height=220)
        self.txt_log.pack(fill='x', expand=False, padx=10, pady=(0, 10))
        self.txt_log.insert('end', 'Segment details will appear here during analysis.\n')
        self.txt_log.configure(state='disabled')

        self._action_frame = ctk.CTkFrame(self._scroll_body, fg_color=self._pal['bg'])
        self._action_frame.pack(fill='x', padx=8, pady=(0, 16))
        action_frame = self._action_frame
        self.btn_run = ctk.CTkButton(
            action_frame,
            text='▶ START AUTOCUT',
            command=self.run_process,
            **self._button_kw('primary_emphasis', height=42),
        )
        self.btn_run.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.btn_pause = ctk.CTkButton(
            action_frame,
            text='⏸ PAUSE',
            state='disabled',
            command=self.toggle_pause,
            **self._button_kw('warning', height=42),
        )
        self.btn_pause.pack(side='left', padx=(5, 5))
        self.btn_stop = ctk.CTkButton(
            action_frame,
            text='⏹ STOP',
            state='disabled',
            command=self.stop_process,
            **self._button_kw('danger', height=42),
        )
        self.btn_stop.pack(side='right', padx=(5, 0))

        self._apply_palette()

    def g(self, s, k, f=''):
        return self.cfg.get(s, k, fallback=f)

    def gb(self, s, k, f=False):
        return self.cfg.getboolean(s, k, fallback=f)

    def _button_kw(self, variant='ghost', *, height=BTN_H, font=None, width=None):
        p = self._pal
        kw = dict(
            corner_radius=BTN_RADIUS,
            font=font or FONT_BTN,
            height=height,
            border_width=2,
            border_color=p['btn_rim'],
        )
        if width is not None:
            kw['width'] = width
        if variant == 'ghost':
            kw.update(
                fg_color=p['panel_elev'],
                hover_color=p['border'],
                text_color=p['text'],
            )
        elif variant == 'icon_secondary':
            # Toolbar icon (e.g. ⚙): muted glyph, not full-brightness `text` in dark mode
            kw.update(
                fg_color=p['panel_elev'],
                hover_color=p['border'],
                text_color=p['muted'],
            )
        elif variant == 'primary':
            kw.update(
                fg_color=p['cyan_dim'],
                hover_color=p['cyan'],
                text_color=p['text'],
                border_color=p['primary_border'],
            )
        elif variant == 'primary_emphasis':
            kw.update(
                fg_color=p['cyan_dim'],
                hover_color=p['cyan'],
                text_color=p['text'],
                border_color=p['primary_border'],
                font=font or FONT_BTN_PRIMARY,
            )
        elif variant == 'warning':
            kw.update(
                fg_color=p['gold_dim'],
                hover_color=p['gold'],
                text_color=p['text'],
                border_color=p['gold_dim'],
            )
        elif variant == 'danger':
            kw.update(
                fg_color=p['stop'],
                hover_color='#ef5350',
                text_color='#ffffff',
                border_color='#b71c1c',
            )
        return kw

    def _on_appearance(self, value=None):
        v = (value or self._appearance.get() or 'dark').strip().lower()
        if v == 'light':
            self._pal = dict(PALETTE_LIGHT)
            ctk.set_appearance_mode('Light')
        else:
            self._pal = dict(PALETTE_DARK)
            ctk.set_appearance_mode('Dark')
        self._appearance.set('light' if v == 'light' else 'dark')
        self._apply_palette()
        self.save_cfg()

    def _configure_pause_button(self, *, enabled, paused=False):
        if not enabled:
            self.btn_pause.configure(state='disabled', text='⏸ PAUSE', **self._button_kw('warning', height=42))
            return
        if paused:
            self.btn_pause.configure(state='normal', text='▶ FORTSETZEN', **self._button_kw('primary', height=42))
        else:
            self.btn_pause.configure(state='normal', text='⏸ PAUSE', **self._button_kw('warning', height=42))

    def _apply_palette(self):
        """Recolour palette-driven widgets (required when switching dark/light in CustomTkinter)."""
        p = self._pal
        self.configure(fg_color=p['bg'])
        self._top_bar.configure(fg_color=p['panel'])
        self._lbl_app_title.configure(text_color=p['text'])
        if getattr(self, '_btn_settings', None):
            self._btn_settings.configure(
                **self._button_kw('icon_secondary', width=44, height=40, font=('Segoe UI', 20)),
            )
        self._appearance_seg.configure(
            fg_color=p['panel'],
            selected_color=p['cyan_dim'],
            selected_hover_color=p['cyan'],
            unselected_color=p['panel_elev'],
            unselected_hover_color=p['border'],
            text_color=p['text'],
        )

        self._scroll_body.configure(
            fg_color=p['bg'],
            bg_color=p['bg'],
            scrollbar_fg_color=p['panel'],
            scrollbar_button_color=p['cyan_dim'],
            scrollbar_button_hover_color=p['cyan'],
        )

        self.tabs.configure(
            fg_color=p['panel'],
            bg_color=p['bg'],
            border_color=p['border'],
            segmented_button_fg_color=p['panel_elev'],
            segmented_button_selected_color=p['cyan_dim'],
            segmented_button_selected_hover_color=p['cyan'],
            segmented_button_unselected_color=p['panel_elev'],
            segmented_button_unselected_hover_color=p['border'],
            text_color=p['text'],
        )
        for _tn in ('tab_source', 'tab_thresholds', 'tab_export'):
            _tb = getattr(self, _tn, None)
            if _tb is not None:
                _tb.configure(fg_color=p['bg'])

        self.frame_progress.configure(fg_color=p['bg'])
        if getattr(self, '_action_frame', None):
            self._action_frame.configure(fg_color=p['bg'])

        if getattr(self, '_card_thresholds', None):
            self._card_thresholds.configure(fg_color=p['panel_elev'], border_color=p['border'])
        if getattr(self, '_card_export', None):
            self._card_export.configure(fg_color=p['panel_elev'], border_color=p['border'])

        self.lbl_status.configure(text_color=p['muted'])
        self.progress.configure(progress_color=p['cyan'], fg_color=p['panel_elev'])

        self.log_frame.configure(fg_color=p['panel'], border_color=p['border'])
        self._lbl_log_title.configure(text_color=p['text'])
        self.txt_log.configure(
            fg_color=p['panel'],
            bg_color=p['bg'],
            border_color=p['border'],
            text_color=p['text'],
            scrollbar_button_color=p['cyan_dim'],
            scrollbar_button_hover_color=p['cyan'],
        )

        for w in getattr(self, '_theme_primary_labels', []):
            try:
                w.configure(text_color=p['text'])
            except Exception:
                pass

        _st = str(self.btn_run.cget('state'))
        self.btn_run.configure(state=_st, **self._button_kw('primary_emphasis', height=42))
        self._configure_pause_button(
            enabled=str(self.btn_pause.cget('state')) == 'normal',
            paused=self.is_paused,
        )
        self.btn_stop.configure(state=str(self.btn_stop.cget('state')), **self._button_kw('danger', height=42))

        if getattr(self, 'btn_player', None):
            self.btn_player.configure(state=str(self.btn_player.cget('state')), **self._button_kw('primary', height=35))

        if getattr(self, 'btn_auto_thresholds', None):
            self.btn_auto_thresholds.configure(**self._button_kw('primary', height=38))
        if getattr(self, 'btn_reset_thresholds', None):
            self.btn_reset_thresholds.configure(**self._button_kw('ghost', height=36))

        if getattr(self, 'btn_retry_export', None):
            self.btn_retry_export.configure(state=str(self.btn_retry_export.cget('state')), **self._button_kw('primary', height=38))

        if self.video_path:
            self.drop_zone.configure(fg_color=p['cyan_dim'], text_color=p['text'])
        else:
            self.drop_zone.configure(fg_color=p['panel_elev'], text_color=p['text'])

        for w in self._theme_muted_labels:
            try:
                w.configure(text_color=p['muted'])
            except Exception:
                pass

        for sl in (getattr(self, 'slider_segment', None), getattr(self, 'slider_motion', None)):
            if sl is not None:
                sl.configure(
                    progress_color=p['cyan'],
                    button_color=p['panel'],
                    button_hover_color=p['panel_elev'],
                    fg_color=p['panel_elev'],
                )

        _sw_kw = dict(
            text_color=p['text'],
            progress_color=p['cyan'],
            button_color=p['panel'],
            button_hover_color=p['panel_elev'],
            fg_color=p['panel_elev'],
        )
        for sw in (
            getattr(self, 'sw_yamnet', None),
            getattr(self, 'sw_dia', None),
            getattr(self, 'sw_act', None),
            getattr(self, 'sw_voc', None),
            getattr(self, 'sw_music', None),
            getattr(self, 'sw_silence', None),
        ):
            if sw is not None:
                sw.configure(**_sw_kw)

        for om in (getattr(self, 'opt_whisper', None), getattr(self, 'opt_mode', None), getattr(self, 'opt_bitrate_mode', None)):
            if om is not None:
                om.configure(
                    fg_color=p['panel'],
                    button_color=p['cyan_dim'],
                    button_hover_color=p['cyan'],
                    text_color=p['text'],
                )

        if getattr(self, 'ed_manual_kbps', None):
            self.ed_manual_kbps.configure(fg_color=p['panel'], border_color=p['border'], text_color=p['text'])

        for ent in (
            getattr(self, 'ed_story', None),
            getattr(self, 'ed_action', None),
            getattr(self, 'ed_vocal', None),
            getattr(self, 'ed_vocal_story_penalty', None),
            getattr(self, 'ed_vocal_speech_penalty', None),
            getattr(self, 'ed_action_story_penalty', None),
            getattr(self, 'ed_action_speech_penalty', None),
        ):
            if ent is not None:
                ent.configure(fg_color=p['panel'], border_color=p['border'], text_color=p['text'])

        for lb in (
            getattr(self, 'lbl_segment_val', None),
            getattr(self, 'lbl_motion_val', None),
        ):
            if lb is not None:
                lb.configure(text_color=p['text'])

        pd = getattr(self, '_paths_dialog', None)
        if pd is not None and pd.winfo_exists():
            pd._apply_palette()

    def open_paths_settings(self):
        w = getattr(self, '_paths_dialog', None)
        if w is not None and w.winfo_exists():
            w.focus_force()
            w.lift()
            return
        self._paths_dialog = PathsSettingsDialog(self)

    def append_log(self, line):
        self.txt_log.configure(state='normal')
        self.txt_log.insert('end', line + '\n')
        self.txt_log.see('end')
        self.txt_log.configure(state='disabled')

    def on_seg_len_slider(self, value):
        self.lbl_seg_len.configure(text=f'{int(round(float(value)))} s')

    def add_entry(self, parent, label, value, info_text=""):
        row = ctk.CTkFrame(parent, fg_color='transparent')
        row.pack(fill='x', padx=12, pady=6)
        _rl = ctk.CTkLabel(row, text=label, width=220, anchor='w', text_color=self._pal['text'], font=FONT_UI)
        _rl.pack(side='left')
        self._theme_primary_labels.append(_rl)
        e = ctk.CTkEntry(
            row,
            width=60,
            fg_color=self._pal['panel'],
            border_color=self._pal['border'],
            text_color=self._pal['text'],
        )
        e.insert(0, value)
        e.pack(side='left', padx=8)
        if info_text:
            hl = ctk.CTkLabel(row, text=info_text, text_color=self._pal['muted'], font=FONT_HINT, justify='left')
            hl.pack(side='left', fill='x', expand=True)
            self._theme_muted_labels.append(hl)
        return e

    def browse_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self._var_output_path.set(d)
            self.save_cfg()

    def build_source_tab(self):
        self.drop_zone = ctk.CTkLabel(
            self.tab_source,
            text='📁 Drop NVIDIA Video Here\n(Drag & Drop)',
            corner_radius=10,
            fg_color=self._pal['panel_elev'],
            text_color=self._pal['text'],
            font=FONT_UI,
        )
        self.drop_zone.pack(fill='x', padx=10, pady=12, ipady=30)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.on_drop)
        
        self.btn_player = ctk.CTkButton(
            self.tab_source,
            text='🎬 Open Video Player for Scene Analysis',
            state='disabled',
            command=self.open_player,
            **self._button_kw('primary', height=35),
        )
        self.btn_player.pack(pady=10, fill='x', padx=10)

        settings_frame = ctk.CTkFrame(self.tab_source, fg_color='transparent')
        settings_frame.pack(fill='x', padx=10, pady=20)
        
        _as = ctk.CTkLabel(settings_frame, text='Analysis Settings', font=FONT_SECTION, text_color=self._pal['text'])
        _as.pack(anchor='w', pady=(0, 10))
        self._theme_primary_labels.append(_as)

        # --- NEU: Segment Length Slider ---
        _seg_title = ctk.CTkLabel(settings_frame, text='Segment Length (Interval Seconds):', font=FONT_UI, text_color=self._pal['text'])
        _seg_title.pack(anchor='w', pady=(5, 0))
        self._theme_primary_labels.append(_seg_title)
        seg_slider_frame = ctk.CTkFrame(settings_frame, fg_color='transparent')
        seg_slider_frame.pack(fill='x', pady=(5, 5))
        
        # Zeigt die aktuelle Zahl an (z.B. "20s")
        current_interval = int(self.g('Settings', 'interval_seconds', '20'))
        self.lbl_segment_val = ctk.CTkLabel(seg_slider_frame, text=f"{current_interval}s", width=30, text_color=self._pal['text'])
        self.lbl_segment_val.pack(side='left')
        self._theme_primary_labels.append(self.lbl_segment_val)
        
        # Der Slider von 5 bis 60 in 55 Schritten
        self.slider_segment = ctk.CTkSlider(seg_slider_frame, from_=5, to=60, number_of_steps=55, command=self.on_segment_slider)
        self.slider_segment.set(current_interval)
        self.slider_segment.pack(side='left', fill='x', expand=True, padx=10)
        _i1 = ctk.CTkLabel(
            settings_frame,
            text='Info: 5-10s small Shorts/TikToks. 20s+ for Gameplay/Longform.',
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
        )
        _i1.pack(anchor='w', pady=(0, 15))
        self._theme_muted_labels.append(_i1)
        # ----------------------------------

        # Whisper Model Dropdown
        row_whisper = ctk.CTkFrame(settings_frame, fg_color='transparent')
        row_whisper.pack(fill='x', pady=5)
        top_row = ctk.CTkFrame(row_whisper, fg_color='transparent')
        top_row.pack(fill='x')
        _wm = ctk.CTkLabel(top_row, text='Whisper Model:', width=120, anchor='w', text_color=self._pal['text'], font=FONT_UI)
        _wm.pack(side='left')
        self._theme_primary_labels.append(_wm)
        self.opt_whisper = ctk.CTkOptionMenu(top_row, values=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'])
        self.opt_whisper.set(self.g('Settings', 'whisper_model', 'base'))
        self.opt_whisper.pack(side='left', fill='x', expand=True)
        _i2 = ctk.CTkLabel(
            row_whisper,
            text="Info: 'base'/'small' = very fast. 'medium'/'large' = better for strong accents/bad audio, but slower.",
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
        )
        _i2.pack(anchor='w', pady=(2, 0))
        self._theme_muted_labels.append(_i2)

        # YAMNet Toggle Switch
        self.sw_yamnet = ctk.CTkSwitch(settings_frame, text='Enable YAMNet (Audio Classification)')
        if self.gb('Settings', 'yamnet_enabled', True): 
            self.sw_yamnet.select()
        self.sw_yamnet.pack(anchor='w', pady=(15, 0))
        _i3 = ctk.CTkLabel(
            settings_frame,
            text='Info: Disable if breathing/moaning sounds are not needed (e.g., pure gameplay). Halves analysis time.',
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
        )
        _i3.pack(anchor='w', pady=(2, 0))
        self._theme_muted_labels.append(_i3)

    def build_thresholds_tab(self):
        frame = ctk.CTkFrame(
            self.tab_thresholds,
            fg_color=self._pal['panel_elev'],
            corner_radius=10,
            border_width=1,
            border_color=self._pal['border'],
        )
        self._card_thresholds = frame
        frame.pack(fill='x', padx=10, pady=10)

        _thr_help = ctk.CTkLabel(
            frame,
            text=(
                'Rule of thumb: higher minimums = stricter (keeps less). '
                'Lower minimums = more sensitive (keeps more, including borderline scenes).'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _thr_help.pack(anchor='w', padx=12, pady=(8, 2))
        self._theme_muted_labels.append(_thr_help)
        
        self.ed_story = self.add_entry(
            frame,
            'Min. Story/Dialogue Score (0-100)',
            self.g('Thresholds', 'min_story_score', '56'),
            'Higher: only strong dialogue/story counts. Lower: more speech-like scenes are treated as story.',
        )
        self.ed_action = self.add_entry(
            frame,
            'Min. Action Score (0-100)',
            self.g('Thresholds', 'min_action_score', '58'),
            'Higher: only strong action remains. Lower: keeps calmer action but may include borderline scenes.',
        )
        self.ed_vocal = self.add_entry(
            frame,
            'Min. Moan/Breath Score (0-100)',
            self.g('Thresholds', 'min_vocal_score', '32'),
            'Higher: only strong moan/breath/scream blocks. Lower: catches weaker vocal cues.',
        )
        
        _pf = ctk.CTkLabel(
            frame,
            text='Penalty Factors (reduces score if speech is present):',
            font=FONT_SECTION,
            text_color=self._pal['text'],
        )
        _pf.pack(anchor='w', padx=12, pady=(15, 0))
        self._theme_primary_labels.append(_pf)
        self.ed_vocal_story_penalty = self.add_entry(
            frame,
            'Moan/Breath story penalty factor',
            self.g('Thresholds', 'vocal_story_penalty_factor', '0.70'),
            'Higher: story/dialogue suppresses moan/breath stronger. Lower: moan/breath survives despite dialogue.',
        )
        self.ed_vocal_speech_penalty = self.add_entry(
            frame,
            'Moan/Breath speech penalty factor',
            self.g('Thresholds', 'vocal_speech_penalty_factor', '0.28'),
            'Higher: spoken speech reduces moan/breath score stronger. Lower: speech interferes less.',
        )
        self.ed_action_story_penalty = self.add_entry(
            frame,
            'Action story penalty factor',
            self.g('Thresholds', 'action_story_penalty_factor', '0.45'),
            'Higher: story/dialogue suppresses action stronger. Lower: action survives despite story.',
        )
        self.ed_action_speech_penalty = self.add_entry(
            frame,
            'Action speech penalty factor',
            self.g('Thresholds', 'action_speech_penalty_factor', '0.20'),
            'Higher: spoken speech reduces action score stronger. Lower: speech interferes less.',
        )

        _mt = ctk.CTkLabel(
            frame,
            text='Action Motion Target (Sensitivity)',
            font=FONT_SECTION,
            text_color=self._pal['text'],
        )
        _mt.pack(anchor='w', padx=12, pady=(15, 0))
        self._theme_primary_labels.append(_mt)
        slider_frame = ctk.CTkFrame(frame, fg_color='transparent')
        slider_frame.pack(fill='x', padx=12, pady=(5, 5))
        
        self.lbl_motion_val = ctk.CTkLabel(
            slider_frame,
            text=self.g('Settings', 'action_motion_target', '1.0'),
            width=30,
            text_color=self._pal['text'],
        )
        self.lbl_motion_val.pack(side='left')
        self._theme_primary_labels.append(self.lbl_motion_val)
        
        self.slider_motion = ctk.CTkSlider(slider_frame, from_=0.1, to=10.0, number_of_steps=99, command=self.on_motion_slider)
        self.slider_motion.set(float(self.g('Settings', 'action_motion_target', '1.0')))
        self.slider_motion.pack(side='left', fill='x', expand=True, padx=10)

        info_text = (
            "Motion target: higher value = less sensitive (needs more movement). "
            "Lower value = more sensitive (accepts calmer movement)."
        )
        _im = ctk.CTkLabel(frame, text=info_text, justify='left', text_color=self._pal['muted'], font=FONT_HINT)
        _im.pack(anchor='w', padx=12, pady=(0, 15))
        self._theme_muted_labels.append(_im)

        _sp = ctk.CTkLabel(
            frame,
            text='Method A scene sensitivity',
            font=FONT_SECTION,
            text_color=self._pal['text'],
        )
        _sp.pack(anchor='w', padx=12, pady=(0, 2))
        self._theme_primary_labels.append(_sp)
        self.opt_method_a_scene_profile = ctk.CTkOptionMenu(
            frame,
            values=[
                'Balanced (recommended)',
                'Sensitive (more scene points)',
                'Strict (fewer scene points)',
            ],
        )
        _profile = (self.g('Settings', 'method_a_scene_profile', 'balanced') or 'balanced').strip().lower()
        if _profile.startswith('sens'):
            self.opt_method_a_scene_profile.set('Sensitive (more scene points)')
        elif _profile.startswith('strict'):
            self.opt_method_a_scene_profile.set('Strict (fewer scene points)')
        else:
            self.opt_method_a_scene_profile.set('Balanced (recommended)')
        self.opt_method_a_scene_profile.pack(fill='x', padx=12, pady=(0, 4))
        _sp_hint = ctk.CTkLabel(
            frame,
            text=(
                'Sensitive detects more cuts for calibration (subtle edits). '
                'Strict keeps only stronger cuts (cleaner/faster).'
            ),
            justify='left',
            text_color=self._pal['muted'],
            font=FONT_HINT,
            wraplength=720,
        )
        _sp_hint.pack(anchor='w', padx=12, pady=(0, 12))
        self._theme_muted_labels.append(_sp_hint)

        self.btn_auto_thresholds = ctk.CTkButton(
            frame,
            text='Auto-calibrate thresholds (Method A)',
            command=self.auto_thresholds_method_a,
            **self._button_kw('primary', height=38),
        )
        self.btn_auto_thresholds.pack(fill='x', padx=12, pady=(12, 8))

        self.btn_reset_thresholds = ctk.CTkButton(
            frame,
            text='Reset to Stable Defaults',
            command=self.restore_stable_defaults,
            **self._button_kw('ghost', height=36),
        )
        self.btn_reset_thresholds.pack(fill='x', padx=12, pady=(0, 10))

    def build_export_tab(self):
        frame = ctk.CTkFrame(
            self.tab_export,
            fg_color=self._pal['panel_elev'],
            corner_radius=10,
            border_width=1,
            border_color=self._pal['border'],
        )
        self._card_export = frame
        frame.pack(fill='x', padx=10, pady=10)
        
        # Categories
        self.sw_dia = ctk.CTkSwitch(frame, text='Keep Story/Dialogue')
        self.sw_act = ctk.CTkSwitch(frame, text='Keep Action')
        self.sw_voc = ctk.CTkSwitch(frame, text='Keep Moan/Breath/Scream Scenes')
        self.sw_music = ctk.CTkSwitch(frame, text='Keep Music')
        self.sw_silence = ctk.CTkSwitch(frame, text='Keep Silence')
        if self.gb('Categories', 'keep_dialogue', False): self.sw_dia.select()
        if self.gb('Categories', 'keep_action', True): self.sw_act.select()
        if self.gb('Categories', 'keep_vocal', True): self.sw_voc.select()
        if self.gb('Categories', 'keep_music', False): self.sw_music.select()
        if self.gb('Categories', 'keep_silence', False): self.sw_silence.select()
        for sw in [self.sw_dia, self.sw_act, self.sw_voc, self.sw_music, self.sw_silence]:
            sw.pack(anchor='w', padx=14, pady=6)
        _keep_hint = ctk.CTkLabel(
            frame,
            text=(
                'Keep switches affect export only. Detection/classification still runs for all categories. '
                'Use Thresholds tab if category detection itself should change.'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _keep_hint.pack(anchor='w', padx=14, pady=(2, 8))
        self._theme_muted_labels.append(_keep_hint)

        self.sw_save_csv = ctk.CTkSwitch(
            frame,
            text='Write segment CSV (output/segment_logs/) — compare runs on the same video',
        )
        if self.gb('Settings', 'save_segment_analysis_csv', False):
            self.sw_save_csv.select()
        self.sw_save_csv.pack(anchor='w', padx=14, pady=(4, 10))
        _csv_hint = ctk.CTkLabel(
            frame,
            text=(
                'Each run appends a new timestamped file. Rows include start_sec, scores, keep flags, '
                'and INI thresholds so two CSVs differ only by what you changed (e.g. keep profile).'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _csv_hint.pack(anchor='w', padx=14, pady=(0, 6))
        self._theme_muted_labels.append(_csv_hint)

        # Export Engine
        self.opt_mode = ctk.CTkOptionMenu(
            frame,
            values=[
                'FFmpeg: H.265 (Hardware NVENC)',
                'DaVinci: Export Timeline (XML)',
                'DaVinci: Export Edit Decision List (EDL)',
                'DaVinci: Export Timeline (XML, cut points – full length)',
                'DaVinci: Export Edit Decision List (EDL, cut points – full length)',
                'DaVinci: Export Timeline (XML, no cuts)',
                'DaVinci: Export Edit Decision List (EDL, no cuts)',
                'DaVinci: AUTO-RENDER',
            ],
        )
        self.opt_mode.set(self.g('Settings', 'export_engine', 'FFmpeg: H.265 (Hardware NVENC)'))
        self.opt_mode.pack(fill='x', padx=14, pady=(15, 6))

        _vb = ctk.CTkLabel(
            frame,
            text='Video bitrate (FFmpeg + DaVinci AUTO-RENDER):',
            font=FONT_SECTION,
            text_color=self._pal['text'],
        )
        _vb.pack(anchor='w', padx=14)
        self._theme_primary_labels.append(_vb)
        self.opt_bitrate_mode = ctk.CTkOptionMenu(
            frame,
            values=['default (NVENC preset / no target kbps)', 'match_source (ffprobe → target kbps)', 'manual (fixed kb/s)'],
        )
        _bm = (self.g('Settings', 'export_bitrate_mode', 'default') or 'default').strip().lower()
        if _bm in ('match_source', 'match', 'source'):
            self.opt_bitrate_mode.set('match_source (ffprobe → target kbps)')
        elif _bm in ('manual', 'fixed'):
            self.opt_bitrate_mode.set('manual (fixed kb/s)')
        else:
            self.opt_bitrate_mode.set('default (NVENC preset / no target kbps)')
        self.opt_bitrate_mode.pack(fill='x', padx=14, pady=(0, 4))
        row_man_br = ctk.CTkFrame(frame, fg_color='transparent')
        row_man_br.pack(fill='x', padx=14, pady=(0, 10))
        _mk = ctk.CTkLabel(row_man_br, text='Manual kb/s', width=100, anchor='w', text_color=self._pal['text'], font=FONT_UI)
        _mk.pack(side='left')
        self._theme_primary_labels.append(_mk)
        self.ed_manual_kbps = ctk.CTkEntry(
            row_man_br,
            width=100,
            fg_color=self._pal['panel'],
            border_color=self._pal['border'],
            text_color=self._pal['text'],
        )
        self.ed_manual_kbps.insert(0, self.g('Settings', 'export_manual_video_kbps', '12000'))
        self.ed_manual_kbps.pack(side='left', padx=(8, 0))
        _ex1 = ctk.CTkLabel(
            frame,
            text=(
                'match_source: reads source video bitrate via ffprobe (approximate). '
                'DaVinci: sets API DataRate if your AutoCutPreset uses a restricted bitrate. '
                'Not byte-identical to source — different encoder/GOP still changes the file.'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _ex1.pack(anchor='w', padx=14, pady=(0, 12))
        self._theme_muted_labels.append(_ex1)

        _ex2 = ctk.CTkLabel(
            frame,
            text=(
                'DaVinci API path, worker Python, and output folder: open ⚙ Settings (top right). '
                'Blank output folder → files next to the source video. Values are written to config on Save / Run / Retry Export. '
                '"cut points – full length": full video in Resolve with razor cuts at each analysis segment — nothing removed. '
                '"no cuts": one clip for the whole file (no segment cuts).'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _ex2.pack(anchor='w', padx=14, pady=(0, 10))
        self._theme_muted_labels.append(_ex2)

        # --- DEIN FEHLENDER EXPORT BUTTON ---
        self.btn_retry_export = ctk.CTkButton(
            frame,
            text='🔁 Retry Export (from Checkpoint)',
            command=self.retry_export_click,
            **self._button_kw('primary', height=38),
        )
        self.btn_retry_export.pack(fill='x', padx=14, pady=(15, 5))
        _ex3 = ctk.CTkLabel(
            frame,
            text=(
                'After a full run: change the category switches above (e.g. turn off Keep Moan/Breath). Settings are saved when you click '
                'Retry Export. The timeline is rebuilt from the checkpoint with your new choices — no re-analysis. '
                'Older checkpoints without segment_results: export uses the saved segment list only (same as before).'
            ),
            text_color=self._pal['muted'],
            font=FONT_HINT,
            justify='left',
            wraplength=720,
        )
        _ex3.pack(anchor='w', padx=14, pady=(0, 8))
        self._theme_muted_labels.append(_ex3)

    def restore_stable_defaults(self):
        # Penalty Faktoren
        self.set_entry_value(self.ed_vocal_story_penalty, '0.70')
        self.set_entry_value(self.ed_vocal_speech_penalty, '0.28')
        self.set_entry_value(self.ed_action_story_penalty, '0.45')
        self.set_entry_value(self.ed_action_speech_penalty, '0.20')
        # Motion Target (Sensibilität)
        self.slider_motion.set(1.0) # Oder 2.2, je nachdem was dein finaler Favorit war
        self.lbl_motion_val.configure(text="1.0")
        self.save_cfg()
        messagebox.showinfo("Reset", "Stable penalty and motion defaults restored!")


    def _sync_retry_export_button(self):
        ck = os.path.join(SCRIPT_DIR, 'output', 'last_autocut_checkpoint.json')
        if getattr(self, 'btn_retry_export', None):
            st = 'normal' if os.path.isfile(ck) else 'disabled'
            self.btn_retry_export.configure(state=st, **self._button_kw('primary', height=38))

    def set_entry_value(self, entry, value):
        entry.delete(0, 'end')
        entry.insert(0, str(value))

    def _schedule_save_cfg(self):
        if self._save_cfg_after_id is not None:
            try:
                self.after_cancel(self._save_cfg_after_id)
            except Exception:
                pass
        self._save_cfg_after_id = self.after(SAVE_CFG_DEBOUNCE_MS, self._debounced_save_cfg)

    def _debounced_save_cfg(self):
        self._save_cfg_after_id = None
        self.save_cfg()

    def on_motion_slider(self, value):
        self.lbl_motion_val.configure(text=f"{value:.1f}")
        self._schedule_save_cfg()

    def on_segment_slider(self, value):
        self.lbl_segment_val.configure(text=f"{int(value)}s")
        self._schedule_save_cfg()

    def on_drop(self, event):
        path = event.data.strip('{}')
        if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            self.video_path = path
            self.drop_zone.configure(text=f'✅ Loaded:\n{os.path.basename(path)}', fg_color=self._pal['cyan_dim'], text_color=self._pal['text'])
            self.btn_player.configure(state='normal', **self._button_kw('primary', height=35))
            self.txt_log.configure(state='normal')
            self.txt_log.delete('1.0', 'end')
            self.txt_log.insert('end', 'New video loaded. Segment log appears here on each run (previous runs were cleared).\n')
            self.txt_log.configure(state='disabled')
        else:
            messagebox.showerror('Error', 'Invalid format.')

    def open_player(self):
        if self.video_path:
            VideoPlayerWindow(self, self.video_path)

    def save_cfg(self):
        if self._save_cfg_after_id is not None:
            try:
                self.after_cancel(self._save_cfg_after_id)
            except Exception:
                pass
            self._save_cfg_after_id = None
        if not self.cfg.has_section('Thresholds'): self.cfg.add_section('Thresholds')
        self.cfg['Thresholds']['min_story_score'] = self.ed_story.get()
        self.cfg['Thresholds']['min_action_score'] = self.ed_action.get()
        self.cfg['Thresholds']['min_vocal_score'] = self.ed_vocal.get()
        self.cfg['Thresholds']['vocal_story_penalty_factor'] = self.ed_vocal_story_penalty.get()
        self.cfg['Thresholds']['vocal_speech_penalty_factor'] = self.ed_vocal_speech_penalty.get()
        self.cfg['Thresholds']['action_story_penalty_factor'] = self.ed_action_story_penalty.get()
        self.cfg['Thresholds']['action_speech_penalty_factor'] = self.ed_action_speech_penalty.get()
        
        if not self.cfg.has_section('Categories'): self.cfg.add_section('Categories')
        self.cfg['Categories']['keep_dialogue'] = str(self.sw_dia.get() == 1).lower()
        self.cfg['Categories']['keep_action'] = str(self.sw_act.get() == 1).lower()
        self.cfg['Categories']['keep_vocal'] = str(self.sw_voc.get() == 1).lower()
        self.cfg['Categories']['keep_music'] = str(self.sw_music.get() == 1).lower()
        self.cfg['Categories']['keep_silence'] = str(self.sw_silence.get() == 1).lower()
        
        if not self.cfg.has_section('Settings'): self.cfg.add_section('Settings')
        self.cfg['Settings']['export_engine'] = self.opt_mode.get()
        _lbl = self.opt_bitrate_mode.get()
        if 'match_source' in _lbl:
            self.cfg['Settings']['export_bitrate_mode'] = 'match_source'
        elif 'manual' in _lbl:
            self.cfg['Settings']['export_bitrate_mode'] = 'manual'
        else:
            self.cfg['Settings']['export_bitrate_mode'] = 'default'
        self.cfg['Settings']['export_manual_video_kbps'] = self.ed_manual_kbps.get().strip()
        self.cfg['Settings']['resolve_api_path'] = self._var_resolve_api.get().strip()
        self.cfg['Settings']['davinci_python_path'] = self._var_davinci_py.get().strip()
        self.cfg['Settings']['output_path'] = self._var_output_path.get().strip()
        self.cfg['Settings']['action_motion_target'] = str(round(self.slider_motion.get(), 1))
        
        # --- NEU: Die Werte aus dem Source-Tab speichern ---
        self.cfg['Settings']['whisper_model'] = self.opt_whisper.get()
        self.cfg['Settings']['yamnet_enabled'] = str(self.sw_yamnet.get() == 1).lower()
        self.cfg['Settings']['interval_seconds'] = str(int(self.slider_segment.get()))
        self.cfg['Settings']['gui_appearance'] = self._appearance.get()
        if getattr(self, 'opt_method_a_scene_profile', None) is not None:
            _prof = str(self.opt_method_a_scene_profile.get()).lower()
            if _prof.startswith('sensitive'):
                self.cfg['Settings']['method_a_scene_profile'] = 'sensitive'
            elif _prof.startswith('strict'):
                self.cfg['Settings']['method_a_scene_profile'] = 'strict'
            else:
                self.cfg['Settings']['method_a_scene_profile'] = 'balanced'
        if getattr(self, 'sw_save_csv', None) is not None:
            self.cfg['Settings']['save_segment_analysis_csv'] = str(self.sw_save_csv.get() == 1).lower()
        # --------------------------------------------------
        
        with open(CFG_PATH, 'w', encoding='utf-8') as f:
            self.cfg.write(f)

    def calc_method_a_thresholds(self, samples):
        story_vals = [s['story'] for s in samples]
        action_vals = [s['action_eff'] for s in samples]
        vocal_vals = [s['vocal_sig'] for s in samples]

        story_med = statistics.median(story_vals)
        action_med = statistics.median(action_vals)
        vocal_med = statistics.median(vocal_vals)

        story_p60 = percentile(story_vals, 0.60)
        action_p70 = percentile(action_vals, 0.70)
        vocal_p70 = percentile(vocal_vals, 0.70)

        new_story = clamp_int(max(12, min(95, story_p60 * 0.95)))
        new_action = clamp_int(max(8, min(95, action_p70 * 0.90)))
        new_vocal = clamp_int(max(6, min(95, vocal_p70 * 0.90)))

        return {
            'story_vals': story_vals,
            'action_vals': action_vals,
            'vocal_vals': vocal_vals,
            'story_med': round(story_med, 1),
            'action_med': round(action_med, 1),
            'vocal_med': round(vocal_med, 1),
            'story_p60': round(story_p60, 1),
            'action_p70': round(action_p70, 1),
            'vocal_p70': round(vocal_p70, 1),
            'new_story': new_story,
            'new_action': new_action,
            'new_vocal': new_vocal,
        }

    def auto_thresholds_method_a(self):
        if not self.video_path:
            messagebox.showwarning('No video selected', 'Please load a video first.')
            return
        
        # WICHTIG: Sichert die GUI-Werte inkl. Slider in die INI, BEVOR die Analyse startet
        self.save_cfg()
        
        self.lbl_status.configure(text='Method A auto-analysis started...', text_color=self._pal['accent_warn'])
        self.progress.set(0)
        threading.Thread(target=self._auto_thresholds_method_a_thread, daemon=True).start()

    def _method_a_sample_count(self, duration_sec):
        """Auto-size calibration samples by runtime length."""
        d = max(1, int(duration_sec))
        if d <= 10 * 60:
            return 12
        if d <= 30 * 60:
            return 24
        if d <= 60 * 60:
            return 32
        return 40

    def _stratified_starts(self, duration, seg_len, sample_count):
        max_start = max(0, int(duration - seg_len))
        if max_start <= 0:
            return [0] * sample_count
        starts = []
        for i in range(sample_count):
            lo = int(round((i * max_start) / float(sample_count)))
            hi = int(round(((i + 1) * max_start) / float(sample_count)))
            if hi < lo:
                hi = lo
            start = int(round((lo + hi) * 0.5))
            starts.append(max(0, min(max_start, start)))
        return starts

    def _scene_points_ffmpeg(self, video_path, threshold=0.26):
        """
        Optional scene-change hints via ffmpeg scene score.
        Returns sorted list of pts_time (seconds). Empty on any error.
        """
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-nostats',
            '-i',
            video_path,
            '-vf',
            f"select='gt(scene,{threshold})',showinfo",
            '-f',
            'null',
            '-',
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            return []
        if r.returncode not in (0, 255):
            return []
        pts = []
        for m in re.finditer(r'pts_time:([0-9]+(?:\.[0-9]+)?)', (r.stderr or '')):
            try:
                pts.append(float(m.group(1)))
            except ValueError:
                pass
        if not pts:
            return []
        pts = sorted(set(pts))
        if len(pts) > 500:
            step = max(1, len(pts) // 500)
            pts = pts[::step]
        return pts

    def _scene_guided_starts(self, duration, seg_len, sample_count, scene_points):
        """Stratified bins + prefer scene-change candidates inside each bin."""
        max_start = max(0, int(duration - seg_len))
        if max_start <= 0:
            return [0] * sample_count
        if not scene_points:
            return self._stratified_starts(duration, seg_len, sample_count)

        candidates = []
        for t in scene_points:
            s = int(round(float(t) - seg_len * 0.35))
            if 0 <= s <= max_start:
                candidates.append(s)
        if not candidates:
            return self._stratified_starts(duration, seg_len, sample_count)
        candidates = sorted(set(candidates))

        starts = []
        for i in range(sample_count):
            lo = int(round((i * max_start) / float(sample_count)))
            hi = int(round(((i + 1) * max_start) / float(sample_count)))
            if hi < lo:
                hi = lo
            in_bin = [c for c in candidates if lo <= c <= hi]
            if in_bin:
                start = in_bin[len(in_bin) // 2]
            else:
                start = int(round((lo + hi) * 0.5))
            starts.append(max(0, min(max_start, int(start))))
        return starts

    def _auto_thresholds_method_a_thread(self):
        try:
            from analyzer_nvidia import analyze_segment
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            duration = max(1, int(total_frames / max(1.0, fps)))

            seg_len = max(5, min(60, int(self.cfg.get('Settings', 'interval_seconds', fallback=20))))
            sample_count = self._method_a_sample_count(duration)
            self.after(
                0,
                lambda n=sample_count, d=duration: self.append_log(
                    f'Method A sampling: duration={d}s -> sample_count={n} (stratified)'
                ),
            )
            profile_ui = str(self.opt_method_a_scene_profile.get() if getattr(self, 'opt_method_a_scene_profile', None) else '').lower()
            if profile_ui.startswith('sensitive'):
                scene_profile, scene_thresh = 'sensitive', 0.20
            elif profile_ui.startswith('strict'):
                scene_profile, scene_thresh = 'strict', 0.32
            else:
                scene_profile, scene_thresh = 'balanced', 0.26
            self.after(
                0,
                lambda p=scene_profile, t=scene_thresh: self.append_log(
                    f'Method A scene profile={p} (ffmpeg scene threshold={t})'
                ),
            )
            scene_points = self._scene_points_ffmpeg(self.video_path, threshold=scene_thresh)
            if scene_points:
                starts = self._scene_guided_starts(duration, seg_len, sample_count, scene_points)
                sampling_mode = 'scene_guided_stratified'
                self.after(
                    0,
                    lambda n=len(scene_points): self.append_log(
                        f'Method A: ffmpeg scene points detected={n} (scene-guided sampling active)'
                    ),
                )
            else:
                starts = self._stratified_starts(duration, seg_len, sample_count)
                sampling_mode = 'stratified_only'
                self.after(0, lambda: self.append_log('Method A: no ffmpeg scene points (fallback to stratified sampling)'))

            vocal_story_penalty = float(self.ed_vocal_story_penalty.get().strip() or '0.70')
            vocal_speech_penalty = float(self.ed_vocal_speech_penalty.get().strip() or '0.28')
            action_story_penalty = float(self.ed_action_story_penalty.get().strip() or '0.52')
            action_speech_penalty = float(self.ed_action_speech_penalty.get().strip() or '0.22')

            samples = []
            for idx, sec in enumerate(starts, start=1):
                self.after(0, lambda i=idx, n=sample_count: self.lbl_status.configure(text=f'Method A: analyzing sample {i} of {n}...', text_color=self._pal['text']))
                m = analyze_segment(self.video_path, int(sec), seg_len)
                s = float(m.get('story_score', 0))
                a = float(m.get('action_score', 0))
                raw_v = float(m.get('sexual_vocal_score', 0))
                human_v = float(m.get('human_vocal_percent', 0))
                speech = float(m.get('speech_percent', 0))
                
                vocal_eff = clamp_int(raw_v - ((s * vocal_story_penalty) + (speech * vocal_speech_penalty)))
                vocal_sig = max(vocal_eff, clamp_int(human_v))
                action_eff = clamp_int(a - ((s * action_story_penalty) + (speech * action_speech_penalty)))
                
                samples.append({
                    'sec': int(sec),
                    'story': clamp_int(s),
                    'action_eff': action_eff,
                    'vocal_sig': vocal_sig,
                    'speech': clamp_int(speech),
                    'raw_action': clamp_int(a),
                    'raw_vocal': clamp_int(raw_v),
                })
                self.after(0, self.progress.set, idx / float(sample_count))

            result = self.calc_method_a_thresholds(samples)
            self.after(0, self.set_entry_value, self.ed_story, result['new_story'])
            self.after(0, self.set_entry_value, self.ed_action, result['new_action'])
            self.after(0, self.set_entry_value, self.ed_vocal, result['new_vocal'])
            self.save_cfg()

            try:
                from algo_snapshot import build_algo_run_snapshot, write_algo_run_artifact

                out_algo = os.path.join(SCRIPT_DIR, 'output', 'algo_runs')
                extra_ma = {
                    'method_a': {
                        'sample_count': sample_count,
                        'sampling_mode': sampling_mode,
                        'scene_profile': scene_profile,
                        'ffmpeg_scene_threshold': scene_thresh,
                        'segment_seconds': seg_len,
                        'sample_starts_sec': starts,
                        'ffmpeg_scene_points_count': len(scene_points),
                        'ffmpeg_scene_points_preview': [round(x, 3) for x in scene_points[:80]],
                        'penalty_factors_at_run': {
                            'vocal_story_penalty': vocal_story_penalty,
                            'vocal_speech_penalty': vocal_speech_penalty,
                            'action_story_penalty': action_story_penalty,
                            'action_speech_penalty': action_speech_penalty,
                        },
                        'samples': samples,
                        'threshold_result': {
                            k: result[k]
                            for k in (
                                'story_med',
                                'action_med',
                                'vocal_med',
                                'story_p60',
                                'action_p70',
                                'vocal_p70',
                                'new_story',
                                'new_action',
                                'new_vocal',
                            )
                        },
                        'distribution_vectors': {
                            'story_vals': result.get('story_vals'),
                            'action_vals': result.get('action_vals'),
                            'vocal_vals': result.get('vocal_vals'),
                        },
                    },
                }
                snap = build_algo_run_snapshot(
                    self.cfg,
                    video_path=self.video_path,
                    cfg_path=CFG_PATH,
                    extra=extra_ma,
                )
                write_algo_run_artifact(
                    snap,
                    self.video_path,
                    prefix='method_a',
                    output_root=out_algo,
                )
            except Exception as e:
                self.after(0, lambda m=str(e): self.append_log(f'Method A: algo snapshot not saved ({m})'))

            lines = [
                'Method A finished.',
                f"Samples: {len(samples)} ({sampling_mode})",
                f"Story/Dialogue median={result['story_med']} p60={result['story_p60']} -> threshold={result['new_story']}",
                f"Action median={result['action_med']} p70={result['action_p70']} -> threshold={result['new_action']}",
                f"Moan/Breath median={result['vocal_med']} p70={result['vocal_p70']} -> threshold={result['new_vocal']}",
                '',
                'Per-sample overview:'
            ]
            for i, row in enumerate(samples, start=1):
                lines.append(
                    f"#{i:02d} @ {row['sec']}s | story={row['story']} | action_eff={row['action_eff']} | moan_breath_sig={row['vocal_sig']} | speech={row['speech']} | action_raw={row['raw_action']} | moan_breath_raw={row['raw_vocal']}"
                )
            msg = '\n'.join(lines)
            self.after(0, lambda: self.lbl_status.configure(text='Method A thresholds set successfully', text_color=self._pal['accent_positive']))
            self.after(0, lambda: messagebox.showinfo('Method A Result', msg))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f'Method A error: {e}', text_color=self._pal['accent_error']))
            self.after(0, lambda: messagebox.showerror('Method A Error', str(e)))

    def stop_process(self):
        self._set_paused(False)
        if self.current_process:
            self.lbl_status.configure(text='Stopping...', text_color=self._pal['accent_warn'])
            try:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.current_process.pid)], creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass
            self.finish_run(-9)

    def _control_file_path(self):
        out_dir = os.path.join(SCRIPT_DIR, 'output')
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, 'autocut_control.json')

    def _set_paused(self, paused):
        self.is_paused = bool(paused)
        data = {'paused': self.is_paused, 'updated_at': datetime.now().isoformat(timespec='seconds')}
        try:
            with open(self._control_file_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.append_log(f'Warning: Could not write pause control file: {e}')

    def toggle_pause(self):
        if not self.current_process:
            return
        self._set_paused(not self.is_paused)
        if self.is_paused:
            self._configure_pause_button(enabled=True, paused=True)
            self.lbl_status.configure(text='Pause requested... waiting for safe pause point', text_color=self._pal['accent_warn'])
            self.append_log('PAUSE requested by user.')
        else:
            self._configure_pause_button(enabled=True, paused=False)
            self.lbl_status.configure(text='Resuming...', text_color=self._pal['text'])
            self.append_log('RESUME requested by user.')

    def run_process(self):
        if not self.video_path:
            messagebox.showwarning('No video selected', 'Please load a video first.')
            return
        self.save_cfg()
        self.btn_run.configure(state='disabled', **self._button_kw('primary_emphasis', height=42))
        self.btn_retry_export.configure(state='disabled', **self._button_kw('primary', height=38))
        self._configure_pause_button(enabled=True, paused=False)
        self.btn_stop.configure(state='normal', **self._button_kw('danger', height=42))
        self._set_paused(False)
        self.progress.set(0)
        self.lbl_status.configure(text='Starting analysis...', text_color=self._pal['accent_warn'])
        self.txt_log.configure(state='normal')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.txt_log.insert('end', f'\n--- Run started {ts} ---\n')
        self.txt_log.see('end')
        self.txt_log.configure(state='disabled')
        threading.Thread(target=self.execute_thread, daemon=True).start()

    def retry_export_click(self):
        ck = os.path.join(SCRIPT_DIR, 'output', 'last_autocut_checkpoint.json')
        if not os.path.isfile(ck):
            messagebox.showwarning(
                'No checkpoint',
                'output/last_autocut_checkpoint.json not found. Run a full analysis with at least one kept scene first.',
            )
            return
        self.save_cfg()
        self.btn_run.configure(state='disabled', **self._button_kw('primary_emphasis', height=42))
        self.btn_retry_export.configure(state='disabled', **self._button_kw('primary', height=38))
        self._configure_pause_button(enabled=True, paused=False)
        self.btn_stop.configure(state='normal', **self._button_kw('danger', height=42))
        self._set_paused(False)
        self.progress.set(0)
        self.lbl_status.configure(text='Retry export from checkpoint...', text_color=self._pal['accent_warn'])
        self.txt_log.configure(state='normal')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.txt_log.insert('end', f'\n--- Retry export {ts} (current export settings) ---\n')
        self.txt_log.see('end')
        self.txt_log.configure(state='disabled')
        threading.Thread(target=self.retry_export_thread, daemon=True).start()

    def set_segment_status(self, line):
        m = re.search(r'SEGMENT\s+(\d+)/(\d+)', line)
        if m:
            current, total = m.group(1), m.group(2)
            self.lbl_status.configure(text=f'Analyzing segment {current} of {total}...', text_color=self._pal['text'])
        self.append_log(line)

    def execute_thread(self):
        script = os.path.join(SCRIPT_DIR, 'autocut_nvidia.py')
        if _is_frozen():
            args = [sys.executable, '--autocut-worker', self.video_path]
        else:
            args = [sys.executable, '-u', script, self.video_path]
        try:
            self.current_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=SCRIPT_DIR,
                env=os.environ.copy(),
            )
            for line in iter(self.current_process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('PROGRESS:'):
                    try:
                        self.after(0, self.progress.set, int(line.split(':')[1]) / 100.0)
                    except Exception:
                        pass
                elif line.startswith('SEGMENT '):
                    self.after(0, lambda txt=line: self.set_segment_status(txt))
                elif line.startswith('PAUSED:'):
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt.replace('PAUSED:', '').strip(), text_color=self._pal['accent_warn']))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif line.startswith('RESUMED:'):
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt.replace('RESUMED:', '').strip(), text_color=self._pal['text']))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'Analysis complete. Starting Export' in line:
                    self.after(0, lambda: self.lbl_status.configure(text='Analysis complete. Starting export...', text_color=self._pal['text']))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif (
                    'Rendered segment' in line
                    or 'DaVinci' in line
                    or 'timeline' in line.lower()
                    or 'EDL' in line
                    or 'FFmpeg' in line
                    or 'XML (' in line
                    or 'geschrieben:' in line
                ):
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt[:110], text_color=self._pal['text']))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'EXPORT_FAILED' in line or 'CHECKPOINT:' in line or 'NO_CHECKPOINT' in line:
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'RETRY_EXPORT:' in line or 'VIDEO_MISSING' in line or 'CHECKPOINT_EMPTY' in line:
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'successfully finished' in line.lower():
                    self.after(0, lambda txt=line: self.append_log(txt))
                else:
                    self.after(0, lambda txt=line: self.append_log(txt))
            self.current_process.wait()
            rc = self.current_process.returncode if self.current_process else 0
            self.after(0, lambda c=rc: self.finish_run(c))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f'Error: {e}', text_color=self._pal['accent_error']))
            self.after(0, lambda: self.append_log(f'Error: {e}'))
            self.after(0, lambda: self.finish_run(-1))

    def retry_export_thread(self):
        script = os.path.join(SCRIPT_DIR, 'autocut_nvidia.py')
        if _is_frozen():
            args = [sys.executable, '--autocut-worker', '--retry-export']
        else:
            args = [sys.executable, '-u', script, '--retry-export']
        try:
            self.current_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=SCRIPT_DIR,
                env=os.environ.copy(),
            )
            for line in iter(self.current_process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('PROGRESS:'):
                    try:
                        self.after(0, self.progress.set, int(line.split(':')[1]) / 100.0)
                    except Exception:
                        pass
                elif (
                    line.startswith('PAUSED:')
                    or line.startswith('RESUMED:')
                    or
                    'Rendered segment' in line
                    or 'DaVinci' in line
                    or 'FFmpeg' in line
                    or 'Concat' in line
                    or 'XML (' in line
                    or 'geschrieben:' in line
                ):
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt[:110], text_color=self._pal['text']))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif (
                    'EXPORT_FAILED' in line
                    or 'RETRY_EXPORT:' in line
                    or 'NO_CHECKPOINT' in line
                    or 'VIDEO_MISSING' in line
                    or 'CHECKPOINT_EMPTY' in line
                    or 'REEXPORT_NO_SEGMENTS' in line
                ):
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'successfully finished' in line.lower():
                    self.after(0, lambda txt=line: self.append_log(txt))
                else:
                    self.after(0, lambda txt=line: self.append_log(txt))
            self.current_process.wait()
            rc = self.current_process.returncode if self.current_process else 0
            self.after(0, lambda c=rc: self.finish_run(c))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f'Error: {e}', text_color=self._pal['accent_error']))
            self.after(0, lambda: self.append_log(f'Error: {e}'))
            self.after(0, lambda: self.finish_run(-1))

    def finish_run(self, returncode=0):
        self.btn_run.configure(state='normal', **self._button_kw('primary_emphasis', height=42))
        self._configure_pause_button(enabled=False, paused=False)
        self.btn_stop.configure(state='disabled', **self._button_kw('danger', height=42))
        self._set_paused(False)
        self.current_process = None
        self._sync_retry_export_button()
        if returncode == 2:
            self.lbl_status.configure(
                text='Analysis saved; export failed — checkpoint & temp files kept (see log)',
                text_color=self._pal['accent_warn'],
            )
        elif returncode == -9:
            self.lbl_status.configure(text='Stopped by user', text_color=self._pal['accent_warn'])
        elif returncode == 3:
            self.lbl_status.configure(text='Retry: no checkpoint (run analysis first)', text_color=self._pal['accent_warn'])
        elif returncode == 4:
            self.lbl_status.configure(text='Retry: video from checkpoint not found', text_color=self._pal['accent_error'])
        elif returncode == 5:
            self.lbl_status.configure(text='Retry: checkpoint has no segments', text_color=self._pal['accent_error'])
        elif returncode == 6:
            self.lbl_status.configure(
                text='Retry: no segments left with current category switches — adjust and try again',
                text_color=self._pal['accent_warn'],
            )
        elif returncode not in (0, None):
            self.lbl_status.configure(text=f'Process exited with code {returncode}', text_color=self._pal['accent_error'])
        else:
            self.lbl_status.configure(text='Finished successfully', text_color=self._pal['accent_positive'])


def _run_frozen_autocut_worker():
    if not _is_frozen() or len(sys.argv) < 2 or sys.argv[1] != '--autocut-worker':
        return False
    import autocut_nvidia

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    autocut_nvidia.main()
    return True


if __name__ == '__main__':
    if _run_frozen_autocut_worker():
        sys.exit(0)
    app = NvidiaGUI()
    app.mainloop()