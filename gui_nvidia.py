import os
import re
import sys
import time
import json
import shutil
from datetime import datetime
import math
import random
import threading
import subprocess
import configparser
import statistics
import cv2
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox, filedialog
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

ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')


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

        self.controls = ctk.CTkFrame(self)
        self.controls.grid(row=1, column=0, padx=10, pady=(0, 10), sticky='ew')
        self.controls.columnconfigure(1, weight=1)

        self.lbl_time = ctk.CTkLabel(self.controls, text='00:00 / 00:00', font=('Arial', 12))
        self.lbl_time.grid(row=0, column=0, padx=10, pady=5)

        self.slider = ctk.CTkSlider(self.controls, from_=0, to=max(1, self.total_frames - 1), command=self.set_frame)
        self.slider.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky='ew')
        self.slider.set(0)

        nav = ctk.CTkFrame(self.controls, fg_color='transparent')
        nav.grid(row=1, column=0, columnspan=3, pady=5)
        ctk.CTkButton(nav, text='⏪ -10s', width=60, command=lambda: self.jump(-10)).pack(side='left', padx=5)
        self.btn_play = ctk.CTkButton(nav, text='▶ Play', fg_color='green', command=self.toggle_play)
        self.btn_play.pack(side='left', padx=10)
        ctk.CTkButton(nav, text='⏩ +10s', width=60, command=lambda: self.jump(10)).pack(side='left', padx=5)

        self.lbl_result = ctk.CTkLabel(self.controls, text='Scrub to a scene and analyze it.', font=('Arial', 14, 'bold'), justify='left')
        self.lbl_result.grid(row=2, column=0, columnspan=3, pady=10)

        frame_analyze = ctk.CTkFrame(self.controls, fg_color='transparent')
        frame_analyze.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        ctk.CTkButton(frame_analyze, text='Analyze → Dialogue', fg_color='#4b0082', hover_color='#300052', command=lambda: self.analyze('dialogue')).pack(side='left', padx=5)
        ctk.CTkButton(frame_analyze, text='Analyze → Action', fg_color='#b8860b', hover_color='#8b6508', command=lambda: self.analyze('action')).pack(side='left', padx=5)
        ctk.CTkButton(frame_analyze, text='Analyze → Vocal', fg_color='#c0392b', hover_color='#922b21', command=lambda: self.analyze('vocal')).pack(side='left', padx=5)

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
        self.btn_play.configure(text='⏸ Pause' if self.is_playing else '▶ Play', fg_color='#b8860b' if self.is_playing else 'green')
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
        self.lbl_result.configure(text=f'Analyzing {category}... please wait.', text_color='yellow')
        self.update()
        from analyzer_nvidia import analyze_segment
        seg_len = max(5, min(60, int(float(self.parent_gui.g('Settings', 'interval_seconds', '20') or 20))))
        m = analyze_segment(self.video_path, sec, seg_len)
        story = m.get('story_score', 0)
        action = m.get('action_score', 0)
        vocal = m.get('sexual_vocal_score', 0)
        speech = m.get('speech_percent', 0)
        res = (
            f"Story: {story} | Action: {action} | Vocal: {vocal} | Speech: {speech}\n"
            f"Words: {m.get('word_count', 0)} | WPM: {m.get('wpm', '—')} (target {m.get('target_wpm', '—')}) | word_score: {m.get('word_score', '—')}\n"
            f"Moan: {m.get('moan_percent', 0)} | Breath: {m.get('breath_percent', 0)} | Scream: {m.get('scream_percent', 0)}"
        )
        self.lbl_result.configure(text=res, text_color='lightgreen')
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


class NvidiaGUI(DnD_CTk):
    def __init__(self):
        super().__init__()
        self.video_path = ''
        self.current_process = None
        self.is_paused = False
        self.cfg = configparser.ConfigParser()
        _ensure_config_file()
        self.cfg.read(CFG_PATH, encoding='utf-8')

        self.title('Autocut NVIDIA Control Center')
        self.geometry('820x1200')

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill='both', expand=True, padx=20, pady=15)
        self.tab_source = self.tabs.add('1. Source')
        self.tab_thresholds = self.tabs.add('2. Thresholds')
        self.tab_export = self.tabs.add('3. Categories & Export')

        self.build_source_tab()
        self.build_thresholds_tab()
        self.build_export_tab()

        self.frame_progress = ctk.CTkFrame(self, fg_color='transparent')
        self.frame_progress.pack(fill='x', padx=20, pady=(0, 5))
        self.lbl_status = ctk.CTkLabel(self.frame_progress, text='Ready', font=('Arial', 12))
        self.lbl_status.pack(anchor='w')
        self.progress = ctk.CTkProgressBar(self.frame_progress)
        self.progress.pack(fill='x', pady=6)
        self.progress.set(0)

        self.log_frame = ctk.CTkFrame(self)
        self.log_frame.pack(fill='both', expand=False, padx=20, pady=(0, 8))
        ctk.CTkLabel(self.log_frame, text='Live Segment Log', font=('Arial', 12, 'bold')).pack(anchor='w', padx=10, pady=(8, 4))
        self.txt_log = ctk.CTkTextbox(self.log_frame, height=220)
        self.txt_log.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self.txt_log.insert('end', 'Segment details will appear here during analysis.\n')
        self.txt_log.configure(state='disabled')

        action_frame = ctk.CTkFrame(self, fg_color='transparent')
        action_frame.pack(fill='x', padx=20, pady=(0, 20))
        self.btn_run = ctk.CTkButton(action_frame, text='▶ START AUTOCUT', height=42, fg_color='#28a745', hover_color='#218838', command=self.run_process)
        self.btn_run.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.btn_pause = ctk.CTkButton(
            action_frame,
            text='⏸ PAUSE',
            height=42,
            fg_color='#f39c12',
            hover_color='#d68910',
            state='disabled',
            command=self.toggle_pause,
        )
        self.btn_pause.pack(side='left', padx=(5, 5))
        self.btn_stop = ctk.CTkButton(action_frame, text='⏹ STOP', height=42, fg_color='#dc3545', hover_color='#c82333', state='disabled', command=self.stop_process)
        self.btn_stop.pack(side='right', padx=(5, 0))

    def g(self, s, k, f=''):
        return self.cfg.get(s, k, fallback=f)

    def gb(self, s, k, f=False):
        return self.cfg.getboolean(s, k, fallback=f)

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
        ctk.CTkLabel(row, text=label, width=220, anchor='w').pack(side='left')
        e = ctk.CTkEntry(row, width=60)
        e.insert(0, value)
        e.pack(side='left', padx=8)
        if info_text:
            ctk.CTkLabel(row, text=info_text, text_color='gray', font=('Arial', 11), justify='left').pack(side='left', fill='x', expand=True)
        return e

    def browse_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.ed_out_dir.delete(0, 'end')
            self.ed_out_dir.insert(0, d)
            self.save_cfg()

    def build_source_tab(self):
        self.drop_zone = ctk.CTkLabel(self.tab_source, text='📁 Drop NVIDIA Video Here\n(Drag & Drop)', corner_radius=10, fg_color='#2a2d2e', font=('Arial', 14))
        self.drop_zone.pack(fill='x', padx=10, pady=12, ipady=30)
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind('<<Drop>>', self.on_drop)
        
        self.btn_player = ctk.CTkButton(self.tab_source, text='🎬 Open Video Player for Scene Analysis', state='disabled', command=self.open_player, height=35)
        self.btn_player.pack(pady=10, fill='x', padx=10)

        settings_frame = ctk.CTkFrame(self.tab_source, fg_color='transparent')
        settings_frame.pack(fill='x', padx=10, pady=20)
        
        ctk.CTkLabel(settings_frame, text='Analysis Settings', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))

        # --- NEU: Segment Length Slider ---
        ctk.CTkLabel(settings_frame, text='Segment Length (Interval Seconds):', font=('Arial', 12, 'bold')).pack(anchor='w', pady=(5, 0))
        seg_slider_frame = ctk.CTkFrame(settings_frame, fg_color='transparent')
        seg_slider_frame.pack(fill='x', pady=(5, 5))
        
        # Zeigt die aktuelle Zahl an (z.B. "20s")
        current_interval = int(self.g('Settings', 'interval_seconds', '20'))
        self.lbl_segment_val = ctk.CTkLabel(seg_slider_frame, text=f"{current_interval}s", width=30)
        self.lbl_segment_val.pack(side='left')
        
        # Der Slider von 5 bis 60 in 55 Schritten
        self.slider_segment = ctk.CTkSlider(seg_slider_frame, from_=5, to=60, number_of_steps=55, command=self.on_segment_slider)
        self.slider_segment.set(current_interval)
        self.slider_segment.pack(side='left', fill='x', expand=True, padx=10)
        ctk.CTkLabel(settings_frame, text="Info: 5-10s small Shorts/TikToks. 20s+ for Gameplay/Longform.", text_color='gray', font=('Arial', 11), justify='left').pack(anchor='w', pady=(0, 15))
        # ----------------------------------

        # Whisper Model Dropdown
        row_whisper = ctk.CTkFrame(settings_frame, fg_color='transparent')
        row_whisper.pack(fill='x', pady=5)
        top_row = ctk.CTkFrame(row_whisper, fg_color='transparent')
        top_row.pack(fill='x')
        ctk.CTkLabel(top_row, text='Whisper Model:', width=120, anchor='w').pack(side='left')
        self.opt_whisper = ctk.CTkOptionMenu(top_row, values=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'])
        self.opt_whisper.set(self.g('Settings', 'whisper_model', 'base'))
        self.opt_whisper.pack(side='left', fill='x', expand=True)
        ctk.CTkLabel(row_whisper, text="Info: 'base'/'small' = very fast. 'medium'/'large' = better for strong accents/bad audio, but slower.", text_color='gray', font=('Arial', 11), justify='left').pack(anchor='w', pady=(2, 0))

        # YAMNet Toggle Switch
        self.sw_yamnet = ctk.CTkSwitch(settings_frame, text='Enable YAMNet (Audio Classification)')
        if self.gb('Settings', 'yamnet_enabled', True): 
            self.sw_yamnet.select()
        self.sw_yamnet.pack(anchor='w', pady=(15, 0))
        ctk.CTkLabel(settings_frame, text="Info: Disable if breathing/moaning sounds are not needed (e.g., pure gameplay). Halves analysis time.", text_color='gray', font=('Arial', 11), justify='left').pack(anchor='w', pady=(2, 0))

    def build_thresholds_tab(self):
        frame = ctk.CTkFrame(self.tab_thresholds)
        frame.pack(fill='x', padx=10, pady=10)
        
        self.ed_story = self.add_entry(frame, 'Min. Story Score (0-100)', self.g('Thresholds', 'min_story_score', '56'), "Higher = requires clearer speech. Lower = tolerates mumbling/noise.")
        self.ed_action = self.add_entry(frame, 'Min. Action Score (0-100)', self.g('Thresholds', 'min_action_score', '58'), "Higher = requires stronger visual motion. Lower = reacts to slight camera pans.")
        self.ed_vocal = self.add_entry(frame, 'Min. Vocal Score (0-100)', self.g('Thresholds', 'min_vocal_score', '32'), "Higher = requires distinctly louder breathing/moaning sounds.")
        
        ctk.CTkLabel(frame, text='Penalty Factors (reduces score if speech is present):', font=('Arial', 12, 'bold')).pack(anchor='w', padx=12, pady=(15, 0))
        self.ed_vocal_story_penalty = self.add_entry(frame, 'Vocal story penalty factor', self.g('Thresholds', 'vocal_story_penalty_factor', '0.70'), "Higher = dialogue blocks vocal detection more strongly.")
        self.ed_vocal_speech_penalty = self.add_entry(frame, 'Vocal speech penalty factor', self.g('Thresholds', 'vocal_speech_penalty_factor', '0.28'))
        self.ed_action_story_penalty = self.add_entry(frame, 'Action story penalty factor', self.g('Thresholds', 'action_story_penalty_factor', '0.45'), "Higher = dialogue blocks action detection more strongly.")
        self.ed_action_speech_penalty = self.add_entry(frame, 'Action speech penalty factor', self.g('Thresholds', 'action_speech_penalty_factor', '0.20'))

        ctk.CTkLabel(frame, text='Action Motion Target (Sensitivity)', font=('Arial', 12, 'bold')).pack(anchor='w', padx=12, pady=(15, 0))
        slider_frame = ctk.CTkFrame(frame, fg_color='transparent')
        slider_frame.pack(fill='x', padx=12, pady=(5, 5))
        
        self.lbl_motion_val = ctk.CTkLabel(slider_frame, text=self.g('Settings', 'action_motion_target', '1.0'), width=30)
        self.lbl_motion_val.pack(side='left')
        
        self.slider_motion = ctk.CTkSlider(slider_frame, from_=0.1, to=10.0, number_of_steps=99, command=self.on_motion_slider)
        self.slider_motion.set(float(self.g('Settings', 'action_motion_target', '1.0')))
        self.slider_motion.pack(side='left', fill='x', expand=True, padx=10)

        info_text = "Info: If 'action_raw' gets stuck at 82, increase to make motion detection less sensitive."
        ctk.CTkLabel(frame, text=info_text, justify='left', text_color='gray', font=('Arial', 11)).pack(anchor='w', padx=12, pady=(0, 15))

        btn_auto = ctk.CTkButton(frame, text='Auto-analyze 10 samples (Method A)', height=38, fg_color='#1f6aa5', hover_color='#174d79', command=self.auto_thresholds_method_a)
        btn_auto.pack(fill='x', padx=12, pady=(12, 8))

        btn_reset = ctk.CTkButton(frame, text='Reset to Stable Defaults', fg_color='#6c757d', hover_color='#5a6268', command=self.restore_stable_defaults)
        btn_reset.pack(fill='x', padx=12, pady=(0, 10))

    def build_export_tab(self):
        frame = ctk.CTkFrame(self.tab_export)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Categories
        self.sw_dia = ctk.CTkSwitch(frame, text='Keep Dialogues')
        self.sw_act = ctk.CTkSwitch(frame, text='Keep Action')
        self.sw_voc = ctk.CTkSwitch(frame, text='Keep Vocal Scenes')
        self.sw_music = ctk.CTkSwitch(frame, text='Keep Music')
        self.sw_silence = ctk.CTkSwitch(frame, text='Keep Silence')
        if self.gb('Categories', 'keep_dialogue', False): self.sw_dia.select()
        if self.gb('Categories', 'keep_action', True): self.sw_act.select()
        if self.gb('Categories', 'keep_vocal', True): self.sw_voc.select()
        if self.gb('Categories', 'keep_music', False): self.sw_music.select()
        if self.gb('Categories', 'keep_silence', False): self.sw_silence.select()
        for sw in [self.sw_dia, self.sw_act, self.sw_voc, self.sw_music, self.sw_silence]:
            sw.pack(anchor='w', padx=14, pady=6)
            
        # Export Engine
        self.opt_mode = ctk.CTkOptionMenu(frame, values=['FFmpeg: H.265 (Hardware NVENC)', 'DaVinci: Export Timeline (XML)', 'DaVinci: Export Edit Decision List (EDL)', 'DaVinci: AUTO-RENDER'])
        self.opt_mode.set(self.g('Settings', 'export_engine', 'FFmpeg: H.265 (Hardware NVENC)'))
        self.opt_mode.pack(fill='x', padx=14, pady=15)
        
        # 1. DaVinci API
        ctk.CTkLabel(frame, text='DaVinci Resolve API Path (optional):', font=('Arial', 12, 'bold')).pack(anchor='w', padx=14)
        self.ed_path = ctk.CTkEntry(frame)
        self.ed_path.insert(0, self.g('Settings', 'resolve_api_path', ''))
        self.ed_path.pack(fill='x', padx=14, pady=(0, 10))
        
        # 2. Python Path
        ctk.CTkLabel(frame, text='Custom Python Worker Path:', font=('Arial', 12, 'bold')).pack(anchor='w', padx=14)
        ctk.CTkLabel(frame, text="Info: Only needed if the standalone .exe fails to auto-detect DaVinci's Python.", text_color='gray', font=('Arial', 11), justify='left').pack(anchor='w', padx=14)
        self.ed_py_path = ctk.CTkEntry(frame)
        self.ed_py_path.insert(0, self.g('Settings', 'davinci_python_path', ''))
        self.ed_py_path.pack(fill='x', padx=14, pady=(0, 10))
        
        # 3. Output Folder
        ctk.CTkLabel(frame, text='Output Folder:', font=('Arial', 12, 'bold')).pack(anchor='w', padx=14)
        ctk.CTkLabel(frame, text='Info: If left blank, files are saved in the same directory as the source video.', text_color='gray', font=('Arial', 11), justify='left').pack(anchor='w', padx=14)
        
        out_row = ctk.CTkFrame(frame, fg_color='transparent')
        out_row.pack(fill='x', padx=14, pady=(0, 8))
        self.ed_out_dir = ctk.CTkEntry(out_row)
        self.ed_out_dir.insert(0, self.g('Settings', 'output_path', ''))
        self.ed_out_dir.pack(side='left', fill='x', expand=True)
        btn_browse = ctk.CTkButton(out_row, text='📁 Browse...', width=80, command=self.browse_output_dir)
        btn_browse.pack(side='left', padx=(8, 0))

        # --- DEIN FEHLENDER EXPORT BUTTON ---
        self.btn_retry_export = ctk.CTkButton(frame, text='🔁 Retry Export (from Checkpoint)', height=38, fg_color='#d35400', hover_color='#a04000', command=self.retry_export_click)
        self.btn_retry_export.pack(fill='x', padx=14, pady=(15, 5))

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
            self.btn_retry_export.configure(state='normal' if os.path.isfile(ck) else 'disabled')

    def set_entry_value(self, entry, value):
        entry.delete(0, 'end')
        entry.insert(0, str(value))

    def on_motion_slider(self, value):
        self.lbl_motion_val.configure(text=f"{value:.1f}")

    def on_motion_slider(self, value):
        self.lbl_motion_val.configure(text=f"{value:.1f}")

    def on_segment_slider(self, value):
        self.lbl_segment_val.configure(text=f"{int(value)}s")

    def on_drop(self, event):
        path = event.data.strip('{}')
        if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            self.video_path = path
            self.drop_zone.configure(text=f'✅ Loaded:\n{os.path.basename(path)}', fg_color='#1f538d')
            self.btn_player.configure(state='normal')
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
        self.cfg['Settings']['resolve_api_path'] = self.ed_path.get()
        self.cfg['Settings']['davinci_python_path'] = self.ed_py_path.get()
        self.cfg['Settings']['output_path'] = self.ed_out_dir.get()
        self.cfg['Settings']['action_motion_target'] = str(round(self.slider_motion.get(), 1))
        
        # --- NEU: Die Werte aus dem Source-Tab speichern ---
        self.cfg['Settings']['whisper_model'] = self.opt_whisper.get()
        self.cfg['Settings']['yamnet_enabled'] = str(self.sw_yamnet.get() == 1).lower()
        self.cfg['Settings']['interval_seconds'] = str(int(self.slider_segment.get()))
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
        
        self.lbl_status.configure(text='Method A auto-analysis started...', text_color='yellow')
        self.progress.set(0)
        threading.Thread(target=self._auto_thresholds_method_a_thread, daemon=True).start()

    def _auto_thresholds_method_a_thread(self):
        try:
            from analyzer_nvidia import analyze_segment
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            duration = max(1, int(total_frames / max(1.0, fps)))

            sample_count = 10
            seg_len = max(5, min(60, int(self.cfg.get('Settings', 'interval_seconds', fallback=20))))
            
            if duration <= seg_len + 2:
                starts = [0] * sample_count
            else:
                step = max(1.0, (duration - seg_len) / float(sample_count))
                starts = []
                for i in range(sample_count):
                    center = int(i * step)
                    jitter = int(min(8, step * 0.35))
                    start = center + random.randint(-jitter, jitter) if jitter > 0 else center
                    start = max(0, min(duration - seg_len, start))
                    starts.append(start)

            vocal_story_penalty = float(self.ed_vocal_story_penalty.get().strip() or '0.70')
            vocal_speech_penalty = float(self.ed_vocal_speech_penalty.get().strip() or '0.28')
            action_story_penalty = float(self.ed_action_story_penalty.get().strip() or '0.52')
            action_speech_penalty = float(self.ed_action_speech_penalty.get().strip() or '0.22')

            samples = []
            for idx, sec in enumerate(starts, start=1):
                self.after(0, lambda i=idx, n=sample_count: self.lbl_status.configure(text=f'Method A: analyzing sample {i} of {n}...', text_color='white'))
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

            lines = [
                'Method A finished.',
                f"Samples: {len(samples)}",
                f"Story median={result['story_med']} p60={result['story_p60']} -> threshold={result['new_story']}",
                f"Action median={result['action_med']} p70={result['action_p70']} -> threshold={result['new_action']}",
                f"Vocal median={result['vocal_med']} p70={result['vocal_p70']} -> threshold={result['new_vocal']}",
                '',
                'Per-sample overview:'
            ]
            for i, row in enumerate(samples, start=1):
                lines.append(
                    f"#{i:02d} @ {row['sec']}s | story={row['story']} | action_eff={row['action_eff']} | vocal_sig={row['vocal_sig']} | speech={row['speech']} | action_raw={row['raw_action']} | vocal_raw={row['raw_vocal']}"
                )
            msg = '\n'.join(lines)
            self.after(0, lambda: self.lbl_status.configure(text='Method A thresholds set successfully', text_color='lightgreen'))
            self.after(0, lambda: messagebox.showinfo('Method A Result', msg))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f'Method A error: {e}', text_color='red'))
            self.after(0, lambda: messagebox.showerror('Method A Error', str(e)))

    def stop_process(self):
        self._set_paused(False)
        if self.current_process:
            self.lbl_status.configure(text='Stopping...', text_color='yellow')
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
            self.btn_pause.configure(text='▶ FORTSETZEN', fg_color='#2ecc71', hover_color='#27ae60')
            self.lbl_status.configure(text='Pause requested... waiting for safe pause point', text_color='orange')
            self.append_log('PAUSE requested by user.')
        else:
            self.btn_pause.configure(text='⏸ PAUSE', fg_color='#f39c12', hover_color='#d68910')
            self.lbl_status.configure(text='Resuming...', text_color='white')
            self.append_log('RESUME requested by user.')

    def run_process(self):
        if not self.video_path:
            messagebox.showwarning('No video selected', 'Please load a video first.')
            return
        self.save_cfg()
        self.btn_run.configure(state='disabled')
        self.btn_retry_export.configure(state='disabled')
        self.btn_pause.configure(state='normal', text='⏸ PAUSE', fg_color='#f39c12', hover_color='#d68910')
        self.btn_stop.configure(state='normal')
        self._set_paused(False)
        self.progress.set(0)
        self.lbl_status.configure(text='Starting analysis...', text_color='yellow')
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
        self.btn_run.configure(state='disabled')
        self.btn_retry_export.configure(state='disabled')
        self.btn_pause.configure(state='normal', text='⏸ PAUSE', fg_color='#f39c12', hover_color='#d68910')
        self.btn_stop.configure(state='normal')
        self._set_paused(False)
        self.progress.set(0)
        self.lbl_status.configure(text='Retry export from checkpoint...', text_color='yellow')
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
            self.lbl_status.configure(text=f'Analyzing segment {current} of {total}...', text_color='white')
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
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt.replace('PAUSED:', '').strip(), text_color='orange'))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif line.startswith('RESUMED:'):
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt.replace('RESUMED:', '').strip(), text_color='white'))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'Analysis complete. Starting Export' in line:
                    self.after(0, lambda: self.lbl_status.configure(text='Analysis complete. Starting export...', text_color='white'))
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
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt[:110], text_color='white'))
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
            self.after(0, lambda: self.lbl_status.configure(text=f'Error: {e}', text_color='red'))
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
                    self.after(0, lambda txt=line: self.lbl_status.configure(text=txt[:110], text_color='white'))
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'EXPORT_FAILED' in line or 'RETRY_EXPORT:' in line or 'NO_CHECKPOINT' in line or 'VIDEO_MISSING' in line or 'CHECKPOINT_EMPTY' in line:
                    self.after(0, lambda txt=line: self.append_log(txt))
                elif 'successfully finished' in line.lower():
                    self.after(0, lambda txt=line: self.append_log(txt))
                else:
                    self.after(0, lambda txt=line: self.append_log(txt))
            self.current_process.wait()
            rc = self.current_process.returncode if self.current_process else 0
            self.after(0, lambda c=rc: self.finish_run(c))
        except Exception as e:
            self.after(0, lambda: self.lbl_status.configure(text=f'Error: {e}', text_color='red'))
            self.after(0, lambda: self.append_log(f'Error: {e}'))
            self.after(0, lambda: self.finish_run(-1))

    def finish_run(self, returncode=0):
        self.btn_run.configure(state='normal')
        self.btn_pause.configure(state='disabled', text='⏸ PAUSE', fg_color='#f39c12', hover_color='#d68910')
        self.btn_stop.configure(state='disabled')
        self._set_paused(False)
        self.current_process = None
        self._sync_retry_export_button()
        if returncode == 2:
            self.lbl_status.configure(
                text='Analysis saved; export failed — checkpoint & temp files kept (see log)',
                text_color='orange',
            )
        elif returncode == -9:
            self.lbl_status.configure(text='Stopped by user', text_color='orange')
        elif returncode == 3:
            self.lbl_status.configure(text='Retry: no checkpoint (run analysis first)', text_color='orange')
        elif returncode == 4:
            self.lbl_status.configure(text='Retry: video from checkpoint not found', text_color='red')
        elif returncode == 5:
            self.lbl_status.configure(text='Retry: checkpoint has no segments', text_color='red')
        elif returncode not in (0, None):
            self.lbl_status.configure(text=f'Process exited with code {returncode}', text_color='red')
        else:
            self.lbl_status.configure(text='Finished successfully', text_color='lightgreen')


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