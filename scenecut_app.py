"""Scenecut – simple GUI for fast_scan.

Drop videos anywhere in the window, they get analyzed one after the other (about a minute per
hour of video). The sliders work live on the selected video: the timeline shows right away
what is kept. Click into the timeline to watch from there (ffplay). Settings apply to all
videos and are remembered.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import fast_scan as fs

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # still usable through "Hinzufügen …"
    TkinterDnD = None

APP_NAME = 'Scenecut'
VIDEO_EXT = {'.mp4', '.mkv', '.mov', '.avi', '.wmv', '.m4v', '.webm', '.ts', '.mts', '.m2ts', '.flv', '.mpg', '.mpeg'}

C = {
    'bg': '#0f1117', 'panel': '#171a22', 'panel2': '#1e222c', 'border': '#2b3040',
    'text': '#e8eaf0', 'muted': '#8a90a2', 'accent': '#00c8ff', 'accent_dim': '#0a5f7a',
    'keep': '#3ccf7a', 'keep_bg': '#1f4d35', 'motion': '#00c8ff', 'motion_off': '#153646', 'vocal': '#ffb74d', 'speech': '#c792ea', 'music': '#7f8cff',
    'cut': '#4a5068', 'error': '#ff8a80', 'drop': '#5a2a2a', 'new': '#ffe066', 'gone': '#b0b6c8',
}
FONT = ('Segoe UI', 10)
FONT_B = ('Segoe UI Semibold', 10)
FONT_H = ('Segoe UI Semibold', 13)
FONT_S = ('Segoe UI', 9)

# name: (settings that differ from fs.Settings defaults, tooltip)
PRESETS = {
    'Highlights': ({'motion_level': 80.0, 'vocal_level': 50.0, 'speech_level': 35.0, 'min_len_s': 6.0,
                    'merge_gap_s': 2.0, 'pad_s': 0.5, 'smooth_s': 3.0},
                   'Nur die stärksten Stellen: viel Bewegung oder deutliche Stimme, Dialog raus.'),
    'Standard': ({}, 'Automatische Schwelle je Video, Stimme zählt mit, Dialog raus.'),
    'Großzügig': ({'motion_level': 25.0, 'vocal_level': 20.0, 'speech_level': 65.0, 'min_len_s': 3.0,
                   'merge_gap_s': 8.0, 'pad_s': 1.0},
                  'Fast alles mit etwas Bewegung oder Stimme bleibt, nur klarer Dialog fliegt raus.'),
}
# Chosen separately from the presets: a preset never switches the decision mode
MODE_KEYS = ('decision', 'strictness', 'block_s')


def _app_dir():
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))


# SCENECUT_SETTINGS: separate settings file (tests, second setup)
SETTINGS_PATH = os.environ.get('SCENECUT_SETTINGS') or os.path.join(_app_dir(), 'scenecut_settings.json')


def fmt_time(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def open_in_explorer(path):
    try:
        if os.path.isfile(path):
            subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
        else:
            os.startfile(path)
    except Exception:
        pass


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _e=None):
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, bg=C['panel2'], fg=C['text'], font=FONT_S, padx=8, pady=4,
                 relief='solid', borderwidth=1, wraplength=320, justify='left').pack()

    def hide(self, _e=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class Item:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.status = 'wartet'
        self.progress = 0.0
        self.an = None
        self.res = None
        self.error = ''
        self.overrides = []  # manual corrections: (start, end, 'keep' | 'drop')


class App:
    def __init__(self, root):
        self.root = root
        self.items = []
        self.current = None
        self.events = queue.Queue()
        self.jobs = queue.Queue()
        self.cancel = threading.Event()
        self.busy_export = False
        self._redraw_after = None
        self._loading = True

        cfg = self._load_cfg()
        self.s = fs.Settings.from_dict(cfg.get('settings'))
        self.out_dir = tk.StringVar(value=cfg.get('out_dir', ''))
        self.export_kind = tk.StringVar(value=cfg.get('export_kind', 'davinci'))
        self.timeline_mode = tk.StringVar(value=cfg.get('timeline_mode', 'cut'))
        self.render_kbps = tk.StringVar(value=cfg.get('render_kbps', 'Auto'))
        self.preview_mute = tk.BooleanVar(value=bool(cfg.get('preview_mute', False)))

        root.title(APP_NAME)
        root.configure(bg=C['bg'])
        root.geometry(cfg.get('geometry', '1180x820'))
        root.minsize(1000, 760)
        self._style()
        self._build()
        self._apply_settings_to_widgets()
        self._update_presets()
        self._loading = False

        if TkinterDnD is not None:
            self._register_drop(root)
        threading.Thread(target=self._worker, daemon=True).start()
        self._poll_id = root.after(100, self._poll)
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._restore_session(cfg.get('videos') or [], cfg.get('current'))
        for arg in sys.argv[1:]:
            self.add_paths([arg])

    def _restore_session(self, videos, current):
        """Reopen the videos of the last session that still exist; cached ones load instantly."""
        found = [p for p in videos if os.path.isfile(p)]
        if found:
            self.add_paths(found)
            it = next((it for it in self.items if it.path == current), None)
            if it is not None:
                self.tree.selection_set(str(id(it)))
                self.tree.see(str(id(it)))
        missing = len(videos) - len(found)
        if found or missing:
            msg = f'{len(found)} Video(s) aus der letzten Sitzung geöffnet'
            self.status_var.set(msg + (f', {missing} nicht mehr vorhanden.' if missing else '.'))

    def _register_drop(self, widget):
        """Drop target on every widget, so dropping works anywhere in the window."""
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>', self._on_drop)
        except (tk.TclError, AttributeError):
            pass
        for child in widget.winfo_children():
            self._register_drop(child)

    # ---------- Settings file ----------

    def _load_cfg(self):
        try:
            with open(SETTINGS_PATH, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_cfg(self):
        data = {'settings': self.s.to_dict(), 'out_dir': self.out_dir.get(),
                'export_kind': self.export_kind.get(), 'timeline_mode': self.timeline_mode.get(),
                'render_kbps': self.render_kbps.get(),
                # Session: reopened on the next start (analysis + corrections come from the cache)
                'videos': [it.path for it in self.items],
                'current': self.current.path if self.current is not None else None,
                'preview_mute': bool(self.preview_mute.get()),
                'geometry': self.root.geometry()}
        try:
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
        except OSError:
            pass

    # ---------- Layout ----------

    def _style(self):
        st = ttk.Style(self.root)
        st.theme_use('clam')
        st.configure('.', background=C['bg'], foreground=C['text'], font=FONT, bordercolor=C['border'],
                     focuscolor=C['accent'], troughcolor=C['panel2'])
        st.configure('Panel.TFrame', background=C['panel'])
        st.configure('TLabel', background=C['bg'], foreground=C['text'])
        st.configure('Panel.TLabel', background=C['panel'], foreground=C['text'])
        st.configure('Muted.TLabel', background=C['panel'], foreground=C['muted'], font=FONT_S)
        st.configure('Head.TLabel', background=C['panel'], foreground=C['text'], font=FONT_H)
        st.configure('Value.TLabel', background=C['panel'], foreground=C['accent'], font=FONT_B, width=7)
        st.configure('TCheckbutton', background=C['panel'], foreground=C['text'], font=FONT_B)
        st.configure('Plain.TCheckbutton', font=FONT)
        st.configure('TRadiobutton', background=C['panel'], foreground=C['text'])
        # clam: indicatorbackground = box / dot area, indicatorforeground = the mark in it
        for w in ('TCheckbutton', 'TRadiobutton'):
            st.configure(w, indicatorbackground=C['panel2'], indicatorforeground=C['panel2'],
                         upperbordercolor=C['border'], lowerbordercolor=C['border'])
            st.map(w, background=[('active', C['panel'])],
                   indicatorbackground=[('selected', C['accent']), ('pressed', C['accent_dim'])],
                   indicatorforeground=[('selected', '#ffffff')])
        st.configure('TButton', background=C['panel2'], foreground=C['text'], padding=(12, 6), borderwidth=1)
        st.map('TButton', background=[('active', C['border']), ('disabled', C['panel'])],
               foreground=[('disabled', C['muted'])])
        st.configure('Accent.TButton', background=C['accent_dim'], foreground='#ffffff', font=FONT_B)
        st.map('Accent.TButton', background=[('active', C['accent']), ('disabled', C['panel'])])
        st.configure('TCombobox', fieldbackground=C['panel2'], background=C['panel2'], foreground=C['text'],
                     arrowcolor=C['text'], insertcolor=C['text'], selectbackground=C['accent_dim'],
                     selectforeground='#ffffff', lightcolor=C['panel2'], darkcolor=C['panel2'], padding=(6, 3))
        st.map('TCombobox', fieldbackground=[('disabled', C['panel'])], background=[('active', C['border'])],
               foreground=[('disabled', C['muted'])], arrowcolor=[('disabled', C['muted'])])
        # Drop-down list of the combobox is a plain Tk listbox
        for opt, val in (('background', C['panel2']), ('foreground', C['text']),
                         ('selectBackground', C['accent_dim']), ('selectForeground', '#ffffff')):
            self.root.option_add(f'*TCombobox*Listbox.{opt}', val)
        st.configure('Horizontal.TScale', background=C['panel'], troughcolor=C['panel2'], sliderthickness=16)
        # Cards: one block per category, so options and explanations clearly belong to their slider
        st.configure('Card.TFrame', background=C['panel2'])
        st.configure('Card.TLabel', background=C['panel2'], foreground=C['text'], font=FONT_B)
        st.configure('CardMuted.TLabel', background=C['panel2'], foreground=C['muted'], font=FONT_S)
        st.configure('CardValue.TLabel', background=C['panel2'], foreground=C['accent'], font=FONT_B, width=7)
        st.configure('Card.Horizontal.TScale', background=C['panel2'], troughcolor=C['bg'], sliderthickness=16)
        for w, font in (('Card.TCheckbutton', FONT_B), ('CardPlain.TCheckbutton', FONT), ('Card.TRadiobutton', FONT)):
            st.configure(w, background=C['panel2'], foreground=C['text'], font=font, indicatorbackground=C['bg'],
                         indicatorforeground=C['bg'], upperbordercolor=C['border'], lowerbordercolor=C['border'])
            st.map(w, background=[('active', C['panel2'])],
                   indicatorbackground=[('selected', C['accent']), ('pressed', C['accent_dim'])],
                   indicatorforeground=[('selected', '#ffffff')])
        st.configure('Preset.TButton', padding=(10, 3))
        st.configure('PresetOn.TButton', padding=(10, 3), background=C['accent_dim'], foreground='#ffffff')
        st.map('PresetOn.TButton', background=[('active', C['accent'])])
        st.configure('TNotebook', background=C['panel'], borderwidth=0, tabmargins=(0, 0, 0, 0))
        st.configure('TNotebook.Tab', background=C['panel'], foreground=C['muted'], padding=(10, 5),
                     font=FONT_B, borderwidth=0)
        st.map('TNotebook.Tab', background=[('selected', C['panel2'])], foreground=[('selected', C['text'])])
        st.configure('Horizontal.TProgressbar', background=C['accent'], troughcolor=C['panel2'], thickness=6)
        st.configure('Treeview', background=C['panel'], fieldbackground=C['panel'], foreground=C['text'],
                     rowheight=26, borderwidth=0)
        st.map('Treeview', background=[('selected', C['accent_dim'])])
        st.configure('Treeview.Heading', background=C['panel2'], foreground=C['muted'], relief='flat')

    def _build(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ----- Left: videos + export -----
        left = ttk.Frame(root, style='Panel.TFrame', padding=12)
        left.grid(row=0, column=0, sticky='nsw', padx=(12, 6), pady=(12, 6))
        left.rowconfigure(2, weight=1, minsize=130)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text='Videos', style='Head.TLabel').grid(row=0, column=0, sticky='w')
        hint = 'Videos oder Ordner ins Fenster ziehen' if TkinterDnD else 'Drag & Drop nicht verfügbar (tkinterdnd2 fehlt)'
        ttk.Label(left, text=hint, style='Muted.TLabel').grid(row=1, column=0, sticky='w', pady=(0, 8))
        self.tree = ttk.Treeview(left, columns=('status',), show='tree headings', height=5, selectmode='browse')
        self.tree.heading('#0', text='Datei')
        self.tree.heading('status', text='Status')
        self.tree.column('#0', width=200)
        self.tree.column('status', width=100, anchor='e')
        self.tree.grid(row=2, column=0, sticky='nsew')
        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<Delete>', lambda e: self.remove_selected())
        self.tree.bind('<Double-1>', lambda e: self.retry_selected())
        btns = ttk.Frame(left, style='Panel.TFrame')
        btns.grid(row=3, column=0, sticky='ew', pady=(8, 0))
        ttk.Button(btns, text='Hinzufügen …', command=self.ask_files).pack(side='left')
        ttk.Button(btns, text='Entfernen', command=self.remove_selected).pack(side='left', padx=6)

        ttk.Separator(left).grid(row=4, column=0, sticky='ew', pady=14)
        ex = ttk.Frame(left, style='Panel.TFrame')
        ex.grid(row=5, column=0, sticky='ew')
        ex.columnconfigure(0, weight=1)
        ttk.Label(ex, text='Export', style='Head.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 4))
        for i, (val, text) in enumerate((('davinci', 'Direkt in DaVinci laden (neue Timeline)'),
                                         ('xml', 'DaVinci-Timeline als Datei (XML)'), ('edl', 'EDL (Schnittliste)'),
                                         ('mp4', 'Fertiges MP4 rendern'))):
            ttk.Radiobutton(ex, text=text, value=val, variable=self.export_kind,
                            command=self._export_kind_changed).grid(row=1 + i, column=0, sticky='w', pady=1)
        ttk.Label(ex, text='Inhalt der Timeline', style='Panel.TLabel', font=FONT_B).grid(
            row=5, column=0, sticky='w', pady=(10, 0))
        self.mode_radios = []
        for i, (val, text) in enumerate((('cut', 'Nur die behaltenen Teile'),
                                         ('full', 'Ganzes Video mit Schnitten (behalten = grün)'))):
            rb = ttk.Radiobutton(ex, text=text, value=val, variable=self.timeline_mode, command=self._save_cfg)
            rb.grid(row=6 + i, column=0, sticky='w', pady=1)
            self.mode_radios.append(rb)
        br = ttk.Frame(ex, style='Panel.TFrame')
        br.grid(row=8, column=0, sticky='w', pady=(10, 0))
        ttk.Label(br, text='Max. Bitrate', style='Panel.TLabel', font=FONT_B).pack(side='left')
        self.kbps_box = ttk.Combobox(br, textvariable=self.render_kbps, width=7,
                                     values=('Auto', '600', '800', '1000', '1500', '2500', '4000', 'Aus'))
        self.kbps_box.pack(side='left', padx=6)
        self.kbps_box.bind('<<ComboboxSelected>>', lambda _e: self._save_cfg())
        self.kbps_box.bind('<FocusOut>', lambda _e: self._save_cfg())
        ttk.Label(br, text='kbit/s  (Auto · Aus · Zahl)', style='Muted.TLabel').pack(side='left')
        Tooltip(self.kbps_box, 'Richtet DaVincis Deliver-Seite ein (H.264, Audio 128 kbit/s). '
                               'Auto = etwa die Qualität der Quelle. Aus = Deliver-Seite nicht anfassen.')
        self.target_title = ttk.Label(ex, text='Ziel', style='Panel.TLabel', font=FONT_B)
        self.target_title.grid(row=9, column=0, sticky='w', pady=(10, 0))
        self.out_label = ttk.Label(ex, text='', style='Muted.TLabel', wraplength=290)
        self.out_label.grid(row=10, column=0, sticky='w')
        of = ttk.Frame(ex, style='Panel.TFrame')
        of.grid(row=11, column=0, sticky='w', pady=(4, 0))
        self.btn_out_pick = ttk.Button(of, text='Ordner wählen …', command=self.ask_out_dir)
        self.btn_out_pick.pack(side='left')
        self.btn_out_reset = ttk.Button(of, text='Neben dem Video', command=lambda: self._set_out(''))
        self.btn_out_reset.pack(side='left', padx=6)
        self._set_out(self.out_dir.get(), save=False)

        self.btn_export = ttk.Button(ex, text='Dieses Video exportieren', style='Accent.TButton',
                                     command=lambda: self.start_export(all_items=False))
        self.btn_export.grid(row=12, column=0, sticky='ew', pady=(14, 0))
        act = ttk.Frame(ex, style='Panel.TFrame')
        act.grid(row=13, column=0, sticky='ew', pady=(6, 0))
        act.columnconfigure((0, 1), weight=1)
        self.btn_export_all = ttk.Button(act, text='Alle exportieren', command=lambda: self.start_export(all_items=True))
        self.btn_export_all.grid(row=0, column=0, sticky='ew')
        self.btn_open = ttk.Button(act, text='Ordner öffnen', command=self._open_last, state='disabled')
        self.btn_open.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        self.btn_import = ttk.Button(ex, text='Schnitt aus DaVinci übernehmen', command=self.start_import)
        self.btn_import.grid(row=14, column=0, sticky='ew', pady=(6, 0))
        Tooltip(self.btn_import, 'Liest deine fertige Timeline dieses Videos aus DaVinci (nur lesen). Was du dort '
                                 'gelöscht, gekürzt oder wieder reingeholt hast, wird hier zur Korrektur – und zum '
                                 'Lernmaterial. Es zählt jeder Clip, der noch in der Timeline liegt (egal welche '
                                 'Farbe, auch halbierte); Rausgeschnittenes und Deaktiviertes gilt als raus.')
        self.last_export = None
        self._export_kind_changed(save=False)

        # ----- Right: timeline + sliders -----
        right = ttk.Frame(root, style='Panel.TFrame', padding=(16, 12))
        right.grid(row=0, column=1, sticky='nsew', padx=(6, 12), pady=(12, 6))
        right.columnconfigure(0, weight=1)

        self.title_var = tk.StringVar(value='Kein Video ausgewählt')
        ttk.Label(right, textvariable=self.title_var, style='Head.TLabel').grid(row=0, column=0, sticky='w')
        self.info_var = tk.StringVar(value='Zieh ein Video ins Fenster, um zu starten.')
        ttk.Label(right, textvariable=self.info_var, style='Muted.TLabel').grid(row=1, column=0, sticky='w')

        self.canvas = tk.Canvas(right, height=180, bg=C['panel2'], highlightthickness=0, cursor='hand2')
        self.canvas.grid(row=2, column=0, sticky='ew', pady=(10, 2))
        self.canvas.bind('<Configure>', lambda e: self.redraw())
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Leave>', lambda e: self.canvas.delete('hover'))
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self._drag = None
        hint_row = ttk.Frame(right, style='Panel.TFrame')
        hint_row.grid(row=3, column=0, sticky='ew')
        for color, text in ((C['keep'], 'grün = bleibt drin'), (C['bg'], 'dunkel = fliegt raus'),
                            ('#ffffff', 'weiß umrandet = selbst behalten'), (C['error'], 'rot = selbst raus')):
            tk.Label(hint_row, text='■', fg=color, bg=C['panel'], font=FONT).pack(side='left')
            ttk.Label(hint_row, text=text + '    ', style='Muted.TLabel').pack(side='left')
        # Only in score mode: the bars under "Bleibt"
        self.cmp_legend = []
        for color, text in ((C['new'], 'gelber Strich = neu ggü. klassisch'), (C['gone'], 'grauer = fällt weg')):
            sq = tk.Label(hint_row, text='▬', fg=color, bg=C['panel'], font=FONT)
            lbl = ttk.Label(hint_row, text=text + '    ', style='Muted.TLabel')
            self.cmp_legend.append((sq, lbl))
        how = ttk.Frame(right, style='Panel.TFrame')
        how.grid(row=4, column=0, sticky='ew')
        ttk.Label(how, text='Klick = ab dort abspielen  ·  Ziehen / Rechtsklick = selbst behalten oder rausschneiden',
                  style='Muted.TLabel').pack(side='left')
        ttk.Checkbutton(how, text='Vorschau ohne Ton', variable=self.preview_mute, style='Plain.TCheckbutton',
                        command=self._save_cfg).pack(side='right')

        top = ttk.Frame(right, style='Panel.TFrame')
        top.grid(row=5, column=0, sticky='ew', pady=(6, 6))
        top.columnconfigure(0, weight=1)
        self.summary_var = tk.StringVar(value='')
        tk.Label(top, textvariable=self.summary_var, bg=C['panel'], fg=C['keep'], font=FONT_H,
                 anchor='w').grid(row=0, column=0, sticky='ew')
        presets = ttk.Frame(top, style='Panel.TFrame')
        presets.grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Label(presets, text='Voreinstellung:', style='Muted.TLabel').pack(side='left', padx=(0, 6))
        self.preset_btns = {}
        for name, (_vals, tip) in PRESETS.items():
            b = ttk.Button(presets, text=name, style='Preset.TButton', command=lambda n=name: self.apply_preset(n))
            b.pack(side='left', padx=(0, 4))
            Tooltip(b, tip)
            self.preset_btns[name] = b

        nb = ttk.Notebook(right)
        nb.grid(row=6, column=0, sticky='nsew')
        right.rowconfigure(6, weight=1)
        box = ttk.Frame(nb, style='Panel.TFrame', padding=(0, 8, 0, 0))
        fine = ttk.Frame(nb, style='Panel.TFrame', padding=(0, 8, 0, 0))
        nb.add(box, text='  Was soll bleiben?  ')
        nb.add(fine, text='  Feinschliff  ')
        box.columnconfigure(0, weight=1)
        fine.columnconfigure(0, weight=1)
        self.v = {}

        # How the categories are combined
        self.v['decision'] = tk.StringVar()
        self.v['strictness'] = tk.DoubleVar()
        card = self._card(box, 0)
        ttk.Label(card, text='Entscheidung', style='Card.TLabel').grid(row=0, column=0, sticky='w', padx=(2, 0))
        modes = ttk.Frame(card, style='Card.TFrame')
        modes.grid(row=1, column=0, sticky='w', padx=(20, 0))
        for val, text, tip in (
                ('classic', 'klassisch', 'Bewegung ODER Stimme reicht zum Behalten, jeder Dialog-Treffer schneidet raus.'),
                ('score', 'kombiniert', 'Alles zählt zusammen: Bewegung, Stimme und Lautstärke dafür, anhaltender '
                                        'Dialog und Musik dagegen. Leise Bewegung fällt eher raus, einzelne '
                                        '„Sprache“-Treffer mitten in Action nicht mehr.'),
                ('blocks', 'Action-Blöcke', 'Entscheidet über ganze zusammenhängende Abschnitte statt Sekunde für '
                                            'Sekunde – ruhige Momente mitten in der Action bleiben drin. Stimme '
                                            'zählt am meisten (über 30 s gemittelt). Gewichte aus deinem ersten '
                                            'komplett geprüften Video, noch vorläufig.')):
            rb = ttk.Radiobutton(modes, text=text, value=val, variable=self.v['decision'], style='Card.TRadiobutton',
                                 command=self._changed)
            rb.pack(side='left', padx=(0, 6))
            Tooltip(rb, tip)
        self.strict_scale = self._scale(card, 0, 'strictness', 0, 100, col=1, card=True)
        self.strict_hint = ttk.Label(card, text='', style='CardMuted.TLabel')
        self.strict_hint.grid(row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))

        # One card per category: title + options left, slider + its explanation right
        self.v['motion_keep'] = tk.BooleanVar()
        self.v['motion_auto'] = tk.BooleanVar()
        self.v['motion_level'] = tk.DoubleVar()
        card = self._card(box, 1)
        ttk.Checkbutton(card, text='Bewegung / Action', variable=self.v['motion_keep'], style='Card.TCheckbutton',
                        command=self._changed).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(card, text='Schwelle automatisch', variable=self.v['motion_auto'],
                        style='CardPlain.TCheckbutton', command=self._changed).grid(row=1, column=0, sticky='w',
                                                                                    padx=(22, 0))
        self.motion_scale = self._scale(card, 0, 'motion_level', 0, 95, col=1, card=True)
        self.motion_hint = ttk.Label(card, text='', style='CardMuted.TLabel')
        self.motion_hint.grid(row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))

        self.v['vocal_keep'] = tk.BooleanVar()
        self.v['vocal_level'] = tk.DoubleVar()
        card = self._card(box, 2)
        ttk.Checkbutton(card, text='Stimme', variable=self.v['vocal_keep'], style='Card.TCheckbutton',
                        command=self._changed).grid(row=0, column=0, sticky='w')
        ttk.Label(card, text='Stöhnen, Atmen, Schreien', style='CardMuted.TLabel').grid(
            row=1, column=0, sticky='w', padx=(22, 0))
        self._scale(card, 0, 'vocal_level', 5, 90, col=1, card=True)
        ttk.Label(card, text='niedriger = mehr zählt als Stimme', style='CardMuted.TLabel').grid(
            row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))

        self.v['speech_mode'] = tk.StringVar()
        self.v['speech_level'] = tk.DoubleVar()
        card = self._card(box, 3)
        ttk.Label(card, text='Dialog / Sprache', style='Card.TLabel').grid(row=0, column=0, sticky='w', padx=(2, 0))
        modes = ttk.Frame(card, style='Card.TFrame')
        modes.grid(row=1, column=0, sticky='w', padx=(20, 0))
        for val, text in (('remove', 'rausschneiden'), ('ignore', 'egal'), ('keep', 'behalten')):
            ttk.Radiobutton(modes, text=text, value=val, variable=self.v['speech_mode'], style='Card.TRadiobutton',
                            command=self._changed).pack(side='left', padx=(0, 6))
        self._scale(card, 0, 'speech_level', 10, 90, col=1, card=True)
        ttk.Label(card, text='niedriger = auch leises Reden zählt als Dialog', style='CardMuted.TLabel').grid(
            row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))

        self.v['music_keep'] = tk.BooleanVar()
        self.v['music_level'] = tk.DoubleVar()
        card = self._card(box, 4)
        ttk.Checkbutton(card, text='Musik', variable=self.v['music_keep'], style='Card.TCheckbutton',
                        command=self._changed).grid(row=0, column=0, sticky='w')
        self._scale(card, 0, 'music_level', 5, 90, col=1, card=True)
        ttk.Label(card, text='niedriger = mehr zählt als Musik', style='CardMuted.TLabel').grid(
            row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))

        for i, (key, text, lo, hi, unit, tip) in enumerate((
                ('min_len_s', 'Kürzeste Szene', 1, 30, 's', 'Kürzere Stücke fliegen raus.'),
                ('merge_gap_s', 'Lücken schließen', 0, 20, 's',
                 'Pausen bis zu dieser Länge zwischen zwei Szenen werden mitgenommen.'),
                ('pad_s', 'Rand an Schnitten', 0, 5, 's', 'Etwas Luft vor und nach jeder Szene.'),
                ('smooth_s', 'Glättung', 0.5, 8, 's', 'Höher = ruhigere Entscheidung, weniger Zerstückelung.'),
                ('block_s', 'Blockgröße', 5, 120, 's',
                 'Nur Action-Blöcke: höher = weniger, längere Blöcke, auch längere Pausen bleiben drin.'),
        )):
            self.v[key] = tk.DoubleVar()
            card = self._card(fine, i)
            ttk.Label(card, text=text, style='Card.TLabel').grid(row=0, column=0, sticky='w', padx=(2, 0))
            self._scale(card, 0, key, lo, hi, unit=unit, col=1, card=True)
            ttk.Label(card, text=tip, style='CardMuted.TLabel').grid(row=1, column=1, columnspan=2, sticky='w',
                                                                     padx=(12, 0))

        # ----- Bottom: progress -----
        bar = ttk.Frame(root, padding=(14, 0, 14, 10))
        bar.grid(row=1, column=0, columnspan=2, sticky='ew')
        bar.columnconfigure(0, weight=1)
        self.pbar = ttk.Progressbar(bar, mode='determinate', maximum=1.0)
        self.pbar.grid(row=0, column=0, sticky='ew', pady=(0, 3))
        self.btn_cancel = ttk.Button(bar, text='Abbrechen', command=self.cancel_current, state='disabled')
        self.btn_cancel.grid(row=0, column=1, rowspan=2, padx=(12, 0))
        self.status_var = tk.StringVar(value='Bereit')
        ttk.Label(bar, textvariable=self.status_var, foreground=C['muted'], font=FONT_S).grid(row=1, column=0, sticky='w')
        self._update_buttons()

    def _scale(self, parent, row, key, lo, hi, unit='', col=2, card=False):
        var = self.v[key]
        sc = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient='horizontal',
                       style='Card.Horizontal.TScale' if card else 'Horizontal.TScale',
                       command=lambda _v: self._changed())
        sc.grid(row=row, column=col, sticky='ew', padx=(12, 8))
        lbl = ttk.Label(parent, style='CardValue.TLabel' if card else 'Value.TLabel', anchor='e')
        lbl.grid(row=row, column=col + 1, sticky='e')

        def show(*_):
            val = var.get()
            lbl.configure(text=(f'{val:.1f} {unit}' if hi <= 10 else f'{val:.0f} {unit}').strip())
        var.trace_add('write', show)
        show()
        return sc

    def _card(self, parent, row):
        card = ttk.Frame(parent, style='Card.TFrame', padding=(10, 6))
        card.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        card.columnconfigure(0, minsize=300)  # wide enough for the decision radios: sliders line up
        card.columnconfigure(1, weight=1)
        return card

    def apply_preset(self, name):
        vals, _tip = PRESETS[name]
        mode = {k: getattr(self.s, k) for k in MODE_KEYS}
        self.s = fs.Settings.from_dict({**fs.Settings().to_dict(), **vals, **mode})
        self._loading = True
        self._apply_settings_to_widgets()
        self._loading = False
        self._recompute()
        self.status_var.set(f'Voreinstellung „{name}“ übernommen')

    def _update_presets(self):
        cur = self.s.to_dict()
        for name, (vals, _tip) in PRESETS.items():
            want = {k: v for k, v in {**fs.Settings().to_dict(), **vals}.items() if k not in MODE_KEYS}
            same = all(abs(cur[k] - v) < 0.05 if isinstance(v, float) else cur[k] == v for k, v in want.items())
            self.preset_btns[name].configure(style='PresetOn.TButton' if same else 'Preset.TButton')

    # ---------- Settings <-> widgets ----------

    def _apply_settings_to_widgets(self):
        s = self.s
        for k in ('motion_keep', 'vocal_keep', 'music_keep', 'speech_mode', 'vocal_level', 'speech_level',
                  'music_level', 'min_len_s', 'merge_gap_s', 'pad_s', 'smooth_s', 'decision', 'strictness',
                  'block_s'):
            self.v[k].set(getattr(s, k))
        self.v['motion_auto'].set(s.motion_level < 0)
        self.v['motion_level'].set(s.motion_level if s.motion_level >= 0 else 50)
        self._sync_motion_widgets()

    def _read_widgets(self):
        s = self.s
        for k in ('motion_keep', 'vocal_keep', 'music_keep'):
            setattr(s, k, bool(self.v[k].get()))
        s.speech_mode = self.v['speech_mode'].get()
        s.decision = self.v['decision'].get()
        for k in ('vocal_level', 'speech_level', 'music_level', 'min_len_s', 'merge_gap_s', 'pad_s', 'smooth_s',
                  'strictness', 'block_s'):
            setattr(s, k, round(float(self.v[k].get()), 1))
        s.motion_level = -1.0 if self.v['motion_auto'].get() else round(float(self.v['motion_level'].get()), 1)

    def _sync_motion_widgets(self):
        auto = self.v['motion_auto'].get()
        self.motion_scale.state(['disabled'] if auto else ['!disabled'])
        score = self.v['decision'].get() != 'classic'
        self.strict_scale.state(['!disabled'] if score else ['disabled'])
        self.strict_hint.configure(text='höher = strenger, nur die klarsten Stellen bleiben' if score else
                                   'Strenge gilt nur für „kombiniert“ und „Action-Blöcke“')

    def _changed(self):
        if self._loading:
            return
        self._read_widgets()
        self._sync_motion_widgets()
        if self._redraw_after:
            self.root.after_cancel(self._redraw_after)
        self._redraw_after = self.root.after(60, self._recompute)

    def _recompute(self):
        self._redraw_after = None
        it = self.current
        if it is not None and it.an is not None:
            it.res = fs.decide(it.an, self.s, it.overrides)
            if self.s.decision != 'classic':
                # What changes against the classic decision: the places worth a look
                classic = fs.Settings.from_dict({**self.s.to_dict(), 'decision': 'classic'})
                old = fs.decide(it.an, classic, it.overrides)['segments']
                it.res['vs_classic'] = (fs.spans_minus(it.res['segments'], old, 0.5),
                                        fs.spans_minus(old, it.res['segments'], 0.5))
            if self.s.motion_level < 0:
                self._loading = True
                self.v['motion_level'].set(round(it.res['auto_motion_level']))
                self._loading = False
        self.redraw()
        self._update_summary()
        self._update_presets()
        self._save_cfg()

    # ---------- Video list ----------

    def _on_drop(self, event):
        self.add_paths(self.root.tk.splitlist(event.data))

    def ask_files(self):
        paths = filedialog.askopenfilenames(
            title='Videos wählen',
            filetypes=[('Videos', ' '.join('*' + e for e in sorted(VIDEO_EXT))), ('Alle Dateien', '*.*')])
        self.add_paths(paths)

    def add_paths(self, paths):
        files = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                for dp, _dn, fn in os.walk(p):
                    # own exports (…_scenecut.mp4) only when dropped on purpose, not with a folder
                    files += [os.path.join(dp, f) for f in sorted(fn)
                              if os.path.splitext(f)[1].lower() in VIDEO_EXT
                              and not os.path.splitext(f)[0].endswith('_scenecut')]
            elif os.path.splitext(p)[1].lower() in VIDEO_EXT:
                files.append(p)
        known = {it.path for it in self.items}
        added = []
        for f in files:
            if f in known:
                continue
            it = Item(f)
            self.items.append(it)
            self.tree.insert('', 'end', iid=str(id(it)), text=it.name, values=(it.status,))
            self.jobs.put(('analyze', it))
            added.append(it)
            known.add(f)
        if not files and paths:
            self.status_var.set('Keine Videodateien erkannt.')
        if added and self.current is None:
            self.tree.selection_set(str(id(added[0])))
        if added:
            self._save_cfg()
        self._update_buttons()

    def remove_selected(self):
        it = self._selected()
        if it is None or it.status.startswith('analysiert') or self.busy_export:
            return
        it.status = 'entfernt'
        self.items.remove(it)
        self.tree.delete(str(id(it)))
        if self.current is it:
            self.current = None
            self._show_item(None)
        self._save_cfg()
        self._update_buttons()

    def retry_selected(self):
        it = self._selected()
        if it is not None and it.an is None and it.status in ('abgebrochen', 'Fehler'):
            it.status, it.error, it.progress = 'wartet', '', 0.0
            self._set_row(it)
            self.jobs.put(('analyze', it))

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return next((it for it in self.items if str(id(it)) == sel[0]), None)

    def _on_select(self, _e=None):
        self._show_item(self._selected())
        self._save_cfg()

    def _show_item(self, it):
        self.current = it
        if it is None:
            self.title_var.set('Kein Video ausgewählt')
            self.info_var.set('Zieh ein Video ins Fenster, um zu starten.')
        else:
            self.title_var.set(it.name)
            self._update_info(it)
            if it.an is not None:
                self._recompute()
                return
        self.redraw()
        self._update_summary()
        self._update_buttons()

    def _update_info(self, it):
        if it.an is not None:
            m = it.an['meta']
            sound = 'Ton ✓' if m.get('has_audio') else 'kein Ton analysiert'
            self.info_var.set(f"{fmt_time(m['duration'])} · {sound} · {os.path.dirname(it.path)}")
        elif it.error:
            self.info_var.set('Fehler: ' + it.error + ' · Doppelklick = nochmal versuchen')
        elif it.status == 'abgebrochen':
            self.info_var.set('Abgebrochen · Doppelklick in der Liste = neu analysieren')
        else:
            self.info_var.set(f'{it.status} · {os.path.dirname(it.path)}')

    def _set_row(self, it):
        if self.tree.exists(str(id(it))):
            self.tree.item(str(id(it)), values=(it.status,))

    # ---------- Timeline ----------

    # Lanes: (key, label, top, bottom). Motion is an area chart, the sound lanes are strips that
    # light up where the category is detected (full colour) and stay faint where it is only weak.
    LABEL_W = 78

    def _lanes(self, h):
        axis = 22
        keep = (6, 30)
        sound_h, gap = 14, 4
        music_b = h - axis - 2
        music = (music_b - sound_h, music_b)
        speech = (music[0] - gap - sound_h, music[0] - gap)
        vocal = (speech[0] - gap - sound_h, speech[0] - gap)
        motion = (keep[1] + 8, vocal[0] - 8)
        return {'keep': keep, 'motion': motion, 'vocal': vocal, 'speech': speech, 'music': music}, h - axis

    @staticmethod
    def _mix(color, amount, base):
        a = min(max(amount, 0.0), 1.0)
        c1 = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        c0 = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
        return '#%02x%02x%02x' % tuple(int(c0[k] + (c1[k] - c0[k]) * a) for k in range(3))

    def redraw(self):
        cv = self.canvas
        cv.delete('all')
        w, h = max(cv.winfo_width(), 50), max(cv.winfo_height(), 50)
        it = self.current
        self._tl = None
        if it is None or it.an is None or it.res is None:
            msg = 'Hier erscheint die Zeitleiste' if it is None else (
                it.error and 'Analyse fehlgeschlagen' or f'Analyse läuft … {it.progress * 100:.0f} %')
            cv.create_text(w / 2, h / 2, text=msg, fill=C['muted'], font=FONT)
            return
        res, s = it.res, self.s
        has_audio = it.an['meta'].get('has_audio')
        dur = res['stats']['duration'] or 1.0
        x0, x1 = self.LABEL_W, w - 10
        self._tl = (x0, x1, dur)
        lanes, axis_y = self._lanes(h)
        cols = max(2, int(x1 - x0))
        n = len(res['curves']['motion'])
        edges = np.linspace(0, n, cols + 1).astype(int)
        edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)
        edges = np.minimum(edges, n)

        def per_px(arr, how):
            arr = np.asarray(arr, np.float64)
            red = np.maximum.reduceat if how == 'max' else np.add.reduceat
            starts = np.minimum(edges[:-1], n - 1)
            out = red(arr, starts)
            if how == 'mean':
                out = out / np.maximum(np.diff(np.r_[starts, n]), 1)
            return out

        def strip(y0, y1, colors):
            """One rectangle per run of equal colour."""
            start = 0
            for i in range(1, cols + 1):
                if i == cols or colors[i] != colors[start]:
                    if colors[start] is not None:
                        cv.create_rectangle(x0 + start, y0, x0 + i, y1, fill=colors[start], outline='')
                    start = i

        def label(key, text, active):
            y0, y1 = lanes[key]
            cv.create_text(10, (y0 + y1) / 2, text=text, anchor='w', font=FONT_S,
                           fill=C['text'] if active else C['muted'])

        # Which pixel columns end up in the cut (final segments incl. snapping / padding)
        kept = np.zeros(cols, bool)
        spans = []
        for a, b in res['segments']:
            ia, ib = int(a / dur * cols), max(int(a / dur * cols) + 1, int(np.ceil(b / dur * cols)))
            kept[ia:min(ib, cols)] = True
            spans.append((x0 + a / dur * (x1 - x0), x0 + b / dur * (x1 - x0)))

        def lane_bg(y0, y1):
            cv.create_rectangle(x0, y0, x1, y1, fill=C['bg'], outline='')
            for xa, xb in spans:
                cv.create_rectangle(xa, y0, max(xb, xa + 1), y1, fill=C['keep_bg'], outline='')

        # Bleibt
        y0, y1 = lanes['keep']
        cv.create_rectangle(x0, y0, x1, y1, fill=C['bg'], outline='')
        for xa, xb in spans:
            cv.create_rectangle(xa, y0, max(xb, xa + 1), y1, fill=C['keep'], outline='')
        for a, b, mode in res.get('overrides', []):
            xa, xb = x0 + a / dur * (x1 - x0), x0 + b / dur * (x1 - x0)
            if mode == 'keep':
                cv.create_rectangle(xa, y0, max(xb, xa + 2), y1, outline='#ffffff', width=2)
            else:
                cv.create_rectangle(xa, y0, max(xb, xa + 2), y1, fill=C['drop'], outline=C['error'], width=1)
        # Score mode: bar under the lane where it differs from the classic decision
        for spans_, color in zip(res.get('vs_classic', ()), (C['new'], C['gone'])):
            for a, b in spans_:
                xa, xb = x0 + a / dur * (x1 - x0), x0 + b / dur * (x1 - x0)
                cv.create_rectangle(xa, y1 + 1, max(xb, xa + 2), y1 + 5, fill=color, outline='')
        label('keep', 'Bleibt', True)

        # Bewegung (score mode: the combined score): bright where the video stays, dimmed where cut out
        y0, y1 = lanes['motion']
        lane_bg(y0, y1)
        score_mode = res.get('score_line') is not None
        mot = per_px(res['curves']['score' if score_mode else 'motion'], 'mean')
        on = s.motion_keep or score_mode
        for i in range(cols):
            yv = y1 - min(max(mot[i], 0), 100) / 100 * (y1 - y0)
            col = (C['motion'] if kept[i] else C['motion_off']) if on else (C['muted'] if kept[i] else C['border'])
            cv.create_line(x0 + i, y1, x0 + i, yv, fill=col)
        if on:
            ly = y1 - (res['score_line'] if score_mode else res['motion_line']) / 100 * (y1 - y0)
            cv.create_line(x0, ly, x1, ly, fill=C['text'], dash=(4, 3))
        label('motion', 'Score' if score_mode else 'Bewegung', on)

        # Sound lanes
        modes = {'vocal': ('Stimme', s.vocal_keep), 'music': ('Musik', s.music_keep),
                 'speech': ({'remove': 'Sprache ✂', 'keep': 'Sprache', 'ignore': 'Sprache'}[s.speech_mode],
                            s.speech_mode != 'ignore')}
        for key in ('vocal', 'speech', 'music'):
            text, on = modes[key]
            y0, y1 = lanes[key]
            lane_bg(y0, y1)
            label(key, text, on and has_audio)
            if not has_audio:
                continue
            vals = per_px(res['curves'][key], 'max') / 100.0
            hit = per_px(res['masks'][key].astype(np.float64), 'mean') > 0.3
            base = C[key] if on else C['muted']
            colors = []
            for i in range(cols):
                if hit[i]:
                    colors.append(base)
                else:
                    q = round(min(vals[i], 1.0) * 4) / 4 * 0.35  # faint, 5 steps → few rectangles
                    colors.append(self._mix(base, q, C['keep_bg'] if kept[i] else C['bg']) if q > 0 else None)
            strip(y0, y1, colors)
        if not has_audio:
            y0 = lanes['vocal'][0]
            cv.create_text((x0 + x1) / 2, (y0 + lanes['music'][1]) / 2, text='kein Ton analysiert',
                           fill=C['muted'], font=FONT_S)

        # Cuts and time axis
        for c in res['cuts']:
            xc = x0 + c / fs.SAMPLE_FPS / dur * (x1 - x0)
            cv.create_line(xc, axis_y - 2, xc, axis_y + 3, fill=C['cut'])
        step = next(st for st in (10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200) if dur / st <= 11)
        t = 0
        while t <= dur:
            xt = x0 + t / dur * (x1 - x0)
            cv.create_line(xt, axis_y, xt, axis_y + 5, fill=C['muted'])
            if xt < x1 - 30:
                cv.create_text(xt + 2, axis_y + 12, text=fmt_time(t), fill=C['muted'], font=FONT_S, anchor='w')
            t += step

    def _x_to_time(self, x):
        if not getattr(self, '_tl', None):
            return None
        x0, x1, dur = self._tl
        if x < x0 - 2:  # lane labels
            return None
        return min(max((x - x0) / max(x1 - x0, 1), 0), 1) * dur

    def _on_hover(self, e):
        it = self.current
        self.canvas.delete('hover')
        if it is None or it.res is None:
            return
        t = self._x_to_time(e.x)
        if t is None:
            return
        i = min(int(t * fs.SAMPLE_FPS), len(it.res['keep']) - 1)
        c = it.res['curves']
        stays = any(a <= t < b for a, b in it.res['segments'])
        txt = f"{fmt_time(t)}  {'bleibt' if stays else 'fliegt raus'}"
        new, gone = it.res.get('vs_classic', ((), ()))
        if any(a <= t < b for a, b in new):
            txt += ' (neu ggü. klassisch)'
        elif any(a <= t < b for a, b in gone):
            txt += ' (klassisch: bleibt)'
        if 'score' in c:
            txt += f"   Score {c['score'][i]:.0f}"
        txt += f"   Bewegung {c['motion'][i]:.0f}"
        if it.an['meta'].get('has_audio'):
            txt += f"  Stimme {c['vocal'][i]:.0f}  Sprache {c['speech'][i]:.0f}"
        h = self.canvas.winfo_height()
        self.canvas.create_line(e.x, 0, e.x, h, fill=C['text'], tags='hover')
        w = self.canvas.winfo_width()
        anchor = 'ne' if e.x > w - 260 else 'nw'
        xo = -6 if anchor == 'ne' else 6
        self.canvas.create_text(e.x + xo, 36, text=txt, fill=C['text'], font=FONT_S, anchor=anchor, tags='hover')

    def _on_press(self, e):
        self._drag = (e.x, False)

    def _on_drag(self, e):
        if self._drag is None or self._x_to_time(self._drag[0]) is None:
            return
        x_start, moved = self._drag
        if not moved and abs(e.x - x_start) < 5:
            return
        self._drag = (x_start, True)
        self.canvas.delete('sel')
        h = self.canvas.winfo_height()
        self.canvas.create_rectangle(min(x_start, e.x), 2, max(x_start, e.x), h - 22, outline='#ffffff',
                                     dash=(3, 2), width=1, tags='sel')

    def _on_release(self, e):
        drag, self._drag = self._drag, None
        if drag is None:
            return
        x_start, moved = drag
        if not moved:
            self._on_click(e)
            return
        a, b = self._x_to_time(min(x_start, e.x)), self._x_to_time(max(x_start, e.x))
        if a is None:
            a = 0.0
        if b is None or b - a < 0.5:
            self.canvas.delete('sel')
            return
        self._range_menu(e, a, b, f'Bereich {fmt_time(a)}–{fmt_time(b)}')

    def _on_right_click(self, e):
        it = self.current
        t = self._x_to_time(e.x)
        if it is None or it.res is None or t is None:
            return
        # The stretch under the mouse: the kept segment there, or the gap between two kept ones
        segs = it.res['segments']
        dur = it.res['stats']['duration']
        hit = next(((a, b) for a, b in segs if a <= t < b), None)
        if hit is None:
            a = max([b for _a, b in segs if b <= t], default=0.0)
            b = min([a2 for a2, _b in segs if a2 > t], default=dur)
            hit = (a, b)
        self._range_menu(e, hit[0], hit[1], f'Abschnitt {fmt_time(hit[0])}–{fmt_time(hit[1])}', play_at=t)

    def _range_menu(self, e, a, b, title, play_at=None):
        it = self.current
        m = tk.Menu(self.root, tearoff=0, bg=C['panel2'], fg=C['text'], activebackground=C['accent_dim'],
                    activeforeground='#ffffff', disabledforeground=C['muted'], font=FONT)
        m.add_command(label=title, state='disabled')
        m.add_separator()
        m.add_command(label='Immer behalten', command=lambda: self._set_override(a, b, 'keep'))
        m.add_command(label='Immer rausschneiden', command=lambda: self._set_override(a, b, 'drop'))
        if any(o_a < b and o_b > a for o_a, o_b, _m in it.overrides):
            m.add_command(label='Eigene Festlegung hier entfernen', command=lambda: self._set_override(a, b, None))
        if play_at is not None:
            m.add_separator()
            m.add_command(label=f'Ab {fmt_time(play_at)} abspielen', command=lambda: self._play(play_at))
        if it.overrides:
            m.add_separator()
            m.add_command(label=f'Alle eigenen Festlegungen entfernen ({len(it.overrides)})',
                          command=self._clear_overrides)
        try:
            m.tk_popup(e.x_root, e.y_root)
        finally:
            m.grab_release()
            self.root.after(150, lambda: self.canvas.delete('sel'))

    def _set_override(self, a, b, mode):
        """mode 'keep' / 'drop' adds a correction (newest wins), None removes corrections in [a, b)."""
        it = self.current
        if it is None:
            return
        it.overrides = fs.set_override(it.overrides, a, b, mode)
        fs.save_overrides(it.path, it.overrides)
        self._recompute()
        what = {'keep': 'bleibt immer drin', 'drop': 'fliegt immer raus', None: 'wieder automatisch'}[mode]
        self.status_var.set(f'{fmt_time(a)}–{fmt_time(b)} {what}')

    def _clear_overrides(self):
        it = self.current
        if it is None:
            return
        it.overrides = []
        fs.save_overrides(it.path, [])
        self._recompute()
        self.status_var.set('Alle eigenen Festlegungen entfernt – wieder alles automatisch')

    def _on_click(self, e):
        it = self.current
        if it is None or it.res is None:
            return
        t = self._x_to_time(e.x)
        if t is None:
            return
        self._play(t)

    def _play(self, t):
        it = self.current
        if it is None:
            return
        try:
            mute = ['-an'] if self.preview_mute.get() else []
            subprocess.Popen(['ffplay', '-hide_banner', '-loglevel', 'error', *mute, '-ss', f'{t:.1f}',
                              '-x', '960', '-y', '540', '-window_title', f'{it.name} @ {fmt_time(t)}', it.path],
                             creationflags=fs.CREATE_NO_WINDOW)
            self.status_var.set(f'Vorschau ab {fmt_time(t)} (Fenster schließen mit Esc)')
        except FileNotFoundError:
            self.status_var.set('ffplay nicht gefunden – liegt normalerweise neben ffmpeg.')

    def _update_summary(self):
        it = self.current
        if it is None or it.res is None:
            self.summary_var.set('')
            self.motion_hint.configure(text='')
            return
        st = it.res['stats']
        own = f" · {len(it.overrides)} selbst festgelegt" if it.overrides else ''
        cmp = ''
        if 'vs_classic' in it.res:
            new, gone = (sum(b - a for a, b in sp) for sp in it.res['vs_classic'])
            cmp = f' · ggü. klassisch +{fmt_time(new)} / −{fmt_time(gone)}'
        self.summary_var.set(f"Behalten: {fmt_time(st['kept_sec'])} von {fmt_time(st['duration'])} "
                             f"({st['share'] * 100:.0f} %) · {st['segments']} {'Szene' if st['segments'] == 1 else 'Szenen'}"
                             f"{own}{cmp}")
        for sq, lbl in self.cmp_legend:
            if 'vs_classic' in it.res:
                sq.pack(side='left')
                lbl.pack(side='left')
            else:
                sq.pack_forget()
                lbl.pack_forget()
        lvl = it.res['motion_level']
        auto = ' (automatisch)' if self.s.motion_level < 0 else ''
        self.motion_hint.configure(text=f'aktivste {100 - lvl:.0f} % zählen als Action{auto}')

    # ---------- Worker ----------

    def _worker(self):
        while True:
            job = self.jobs.get()
            kind = job[0]
            try:
                if kind == 'analyze':
                    self._run_analyze(job[1])
                elif kind == 'export':
                    self._run_export(*job[1:])
                elif kind == 'import':
                    self._run_import(job[1])
            except Exception as e:  # never kill the worker
                self.events.put(('error', job, str(e)))

    def _run_analyze(self, it):
        if it.status == 'entfernt':
            return
        self.cancel.clear()
        it.status = 'analysiert 0 %'
        self.events.put(('item', it))

        def prog(p, _text):
            it.progress = p
            it.status = f'analysiert {p * 100:.0f} %'
            self.events.put(('item', it))
        try:
            it.an = fs.analyze(it.path, progress=prog, cancel=self.cancel)
            it.overrides = fs.load_overrides(it.path)
            it.status = 'fertig'
        except fs.Cancelled:
            it.status = 'abgebrochen'
        except Exception as e:
            it.status = 'Fehler'
            it.error = str(e)
        self.events.put(('item', it))

    def _run_export(self, items, kind, out_dir, full_length=False, render_kbps=None):
        self.cancel.clear()
        written = []
        loaded = []
        for k, it in enumerate(items, 1):
            res = fs.decide(it.an, self.s, it.overrides)
            if not res['segments']:
                self.events.put(('status', f'{it.name}: nichts zu behalten – übersprungen'))
                continue
            dest = out_dir or os.path.dirname(it.path)
            prefix = f'[{k}/{len(items)}] {it.name}: ' if len(items) > 1 else ''
            self.events.put(('progress', 0.0, prefix + 'Export läuft …'))
            try:
                if kind == 'davinci':
                    from davinci_timeline import load_into_timeline  # Resolve API only when needed
                    _proj, tl, _n = load_into_timeline(
                        it.path, res['segments'], log=lambda m: self.events.put(('status', prefix + m)),
                        full_length=full_length, duration=it.an['meta']['duration'],
                        deliver=render_kbps != 0, render_kbps=render_kbps, render_dir=out_dir or None)
                    loaded.append(tl)
                    fs.save_export_record(it.path, res['segments'], full_length, target=f'DaVinci: {tl}')
                    self.events.put(('progress', k / len(items), prefix + f'In DaVinci geladen: {tl}'))
                    continue
                if kind == 'mp4':
                    path = fs.export_mp4(it.path, res['segments'], dest, cancel=self.cancel,
                                         progress=lambda p, t: self.events.put(('progress', p, prefix + t)))
                else:
                    path = fs.export_davinci(it.path, res['segments'], it.an['meta']['fps'], dest, kind,
                                             full_length=full_length, duration=it.an['meta']['duration'])
                    # Reference for reading the edit back once the file is imported into Resolve
                    fs.save_export_record(it.path, res['segments'], full_length, target=path)
            except fs.Cancelled:
                self.events.put(('export_done', written, 'Export abgebrochen'))
                return
            written.append(path)
        if loaded:
            msg = (f'In DaVinci geladen: Timeline „{loaded[0]}“' if len(loaded) == 1 else
                   f'In DaVinci geladen: {len(loaded)} Timelines')
            self.events.put(('export_done', written, msg))
            return
        self.events.put(('export_done', written, None))

    def _run_import(self, it):
        from davinci_timeline import read_timeline_cut  # Resolve API only when needed
        project, timeline, spans, clips = read_timeline_cut(it.path, log=lambda m: self.events.put(('status', m)))
        self.events.put(('import_done', it, project, timeline, spans, clips))

    def start_import(self):
        it = self.current
        if it is None or it.an is None or self.busy_export:
            return
        self.busy_export = True
        self._update_buttons()
        self.jobs.put(('import', it))

    def _apply_import(self, it, project, timeline, spans, clips):
        """Turn the cut read back from DaVinci into corrections (and labels)."""
        dur = it.an['meta']['duration']
        final = [(max(0.0, a), min(dur, b)) for a, b in spans if min(dur, b) > max(0.0, a)]
        rec = fs.load_export_record(it.path)
        if rec is not None:
            exported, ref = rec['segments'], f"dem Export vom {rec.get('time', '?')}"
        else:
            res = it.res or fs.decide(it.an, self.s, it.overrides)
            exported, ref = res['segments'], 'dem aktuellen Vorschlag (kein Export gespeichert)'
        changes = fs.edit_to_overrides(exported, final)
        drops = [c for c in changes if c[2] == 'drop']
        keeps = [c for c in changes if c[2] == 'keep']
        kept = sum(b - a for a, b in final)
        added = sum(b - a for a, b, _m in keeps)
        # Every clip still in the timeline counts, orange ones too: a lot "added" usually means
        # orange parts that were meant to go are still in there
        warn = ''
        if added > 120 and added > 0.5 * sum(b - a for a, b in exported):
            warn = ('Achtung: auffällig viel dazugenommen. Liegen noch orange Clips in der Timeline, '
                    'die eigentlich raus sollen? Dann „Abbrechen“, in DaVinci rausschneiden und nochmal.\n\n')
        msg = (f'Timeline „{timeline}“ ({clips} Clips) gelesen.\n'
               f'Im Schnitt: {fmt_time(kept)} (alles, was noch in der Timeline liegt). Gegenüber {ref}:\n'
               f'  • {len(drops)} Stellen entfernt ({fmt_time(sum(b - a for a, b, _m in drops))})\n'
               f'  • {len(keeps)} Stellen dazugenommen ({fmt_time(added)})\n\n'
               + warn +
               'Hast du das ganze Video in DaVinci durchgesehen?\n\n'
               'Ja – alles in der Timeline gilt als gewollt, alles andere als ungewollt '
               '(auch was du unverändert gelassen hast).\n'
               'Nein – nur deine Änderungen werden übernommen.')
        answer = messagebox.askyesnocancel(APP_NAME, msg)
        if answer is None:
            self.status_var.set('Übernehmen abgebrochen – nichts geändert.')
            return
        for a, b, mode in changes:
            it.overrides = fs.set_override(it.overrides, a, b, mode)
        fs.save_overrides(it.path, it.overrides)
        fs.save_labels(it.path, final, reviewed=answer, source='davinci', note=f'{project} / {timeline}')
        if it is self.current:
            self._recompute()
        self.status_var.set(f'Aus DaVinci übernommen: {len(drops)} entfernt, {len(keeps)} dazugenommen'
                            + (' · Video als komplett geprüft gespeichert' if answer else ''))

    def _poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == 'item':
                    it = ev[1]
                    self._set_row(it)
                    if it is self.current:
                        self._update_info(it)
                        if it.an is not None and it.res is None:
                            self._show_item(it)
                        else:
                            self.pbar['value'] = it.progress
                            self.redraw()
                    if it.status.startswith('analysiert'):
                        self.pbar['value'] = it.progress
                        self.status_var.set(f'{it.name}: {it.status}')
                    elif it.status == 'fertig':
                        self.pbar['value'] = 1.0
                        m = it.an['meta']
                        how = ' (aus dem Cache)' if m.get('from_cache') else f" ({m.get('analysis_sec', 0):.0f} s)"
                        self.status_var.set(f'{it.name}: Analyse fertig{how}')
                    elif it.status == 'Fehler':
                        self.status_var.set(f'{it.name}: {it.error}')
                    self._update_buttons()
                elif kind == 'progress':
                    self.pbar['value'] = ev[1]
                    self.status_var.set(ev[2])
                elif kind == 'status':
                    self.status_var.set(ev[1])
                elif kind == 'export_done':
                    self.busy_export = False
                    written, msg = ev[1], ev[2]
                    if written:
                        self.last_export = written[-1]
                    if written or msg:
                        self.pbar['value'] = 1.0
                    self.status_var.set(msg or (f'Exportiert: {written[-1]}' if len(written) == 1 else
                                                f'{len(written)} Dateien exportiert' if written else 'Nichts exportiert'))
                    self._update_buttons()
                elif kind == 'import_done':
                    self.busy_export = False
                    self._update_buttons()
                    self._apply_import(*ev[1:])
                elif kind == 'error':
                    self.busy_export = False
                    self.status_var.set('Fehler: ' + ev[2])
                    messagebox.showerror(APP_NAME, ev[2])
                    self._update_buttons()
        except queue.Empty:
            pass
        self._poll_id = self.root.after(100, self._poll)

    # ---------- Export ----------

    def _export_kind_changed(self, save=True):
        davinci = self.export_kind.get() == 'davinci'
        self.kbps_box.state(['!disabled'] if davinci else ['disabled'])
        # A full-length MP4 would just be the original
        for rb in self.mode_radios:
            rb.state(['disabled'] if self.export_kind.get() == 'mp4' else ['!disabled'])
        self.btn_export.configure(text='In DaVinci laden' if davinci else 'Dieses Video exportieren')
        self.btn_export_all.configure(text='Alle in DaVinci laden' if davinci else 'Alle exportieren')
        self._set_out(self.out_dir.get(), save=False)
        if save:
            self._save_cfg()

    def _set_out(self, path, save=True):
        self.out_dir.set(path)
        text = path or 'neben dem jeweiligen Video'
        if self.export_kind.get() == 'davinci':
            text = ('Timeline im geöffneten DaVinci-Projekt (Resolve wird bei Bedarf gestartet), '
                    'Render-Ziel: ' + text)
        self.out_label.configure(text=text)
        if save:
            self._save_cfg()

    def ask_out_dir(self):
        d = filedialog.askdirectory(title='Zielordner wählen', initialdir=self.out_dir.get() or None)
        if d:
            self._set_out(os.path.abspath(d))

    def start_export(self, all_items):
        items = [it for it in self.items if it.an is not None] if all_items else \
            ([self.current] if self.current is not None and self.current.an is not None else [])
        if not items:
            return
        # Bitrate for DaVinci's Deliver page: None = from the source, 0 = leave Deliver alone
        val = self.render_kbps.get().strip().lower()
        if val in ('', 'auto'):
            kbps = None
        elif val == 'aus':
            kbps = 0
        elif val.isdigit() and int(val) >= 100:
            kbps = int(val)
        else:
            messagebox.showerror(APP_NAME, 'Max. Bitrate: Auto, Aus oder eine Zahl in kbit/s (z. B. 1000).')
            return
        self._save_cfg()
        self.busy_export = True
        self._update_buttons()
        self.jobs.put(('export', items, self.export_kind.get(), self.out_dir.get(), self.timeline_mode.get() == 'full',
                       kbps))

    def cancel_current(self):
        self.cancel.set()
        self.status_var.set('Wird abgebrochen …')

    def _open_last(self):
        if self.last_export:
            open_in_explorer(self.last_export)

    def _update_buttons(self):
        cur_ok = self.current is not None and self.current.an is not None
        any_ok = any(it.an is not None for it in self.items)
        running = self.busy_export or any(it.status.startswith('analysiert') for it in self.items)
        self.btn_export.state(['!disabled'] if cur_ok and not self.busy_export else ['disabled'])
        self.btn_import.state(['!disabled'] if cur_ok and not self.busy_export else ['disabled'])
        self.btn_export_all.state(['!disabled'] if any_ok and not self.busy_export else ['disabled'])
        self.btn_cancel.state(['!disabled'] if running else ['disabled'])
        self.btn_open.state(['!disabled'] if self.last_export else ['disabled'])

    def _on_close(self):
        self.cancel.set()
        for job in (self._poll_id, self._redraw_after):
            if job:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
        self._save_cfg()
        self.root.destroy()


def main():
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
