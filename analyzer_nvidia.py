import os
import csv
import subprocess
import tempfile
import configparser
import cv2
import numpy as np
import torch
import onnxruntime as ort
from faster_whisper import WhisperModel

CREATE_NO_WINDOW = 0x08000000
CFG_PATH = 'config_nvidia.ini'
WHISPER_MODEL_CACHE = {}
YAMNET_SESSION = None
YAMNET_CLASSNAMES = None
YAMNET_DEFAULT_WINDOW = 15600
YAMNET_DEFAULT_HOP = 7680
_HW_PROFILE_PRINTED = False


def _module_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _safe_unlink(path):
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(CFG_PATH, encoding='utf-8')
    return cfg


def clamp_int(v, low=0, high=100):
    return max(low, min(high, int(v)))


def _ctranslate2_cuda_devices():
    """faster-whisper uses ctranslate2 for GPU — this is the authoritative CUDA check."""
    try:
        import ctranslate2 as ct2

        return max(0, int(ct2.get_cuda_device_count()))
    except Exception:
        return 0


def _resolve_whisper_device(cfg):
    """
    INI whisper_device: auto | cuda | cpu
    GPU only if ctranslate2 sees CUDA devices (matches faster-whisper backend).
    """
    pref = cfg.get('Settings', 'whisper_device', fallback='auto').strip().lower()
    n_ct = _ctranslate2_cuda_devices()
    if pref in ('cpu',):
        return 'cpu'
    if pref in ('cuda', 'gpu'):
        if n_ct > 0:
            return 'cuda'
        print(
            'WARNUNG: whisper_device=cuda, aber ctranslate2 findet keine CUDA-GPU (CUDA/cuBLAS für faster-whisper prüfen). Nutze CPU.',
            flush=True,
        )
        return 'cpu'
    return 'cuda' if n_ct > 0 else 'cpu'


def _yamnet_onnx_providers(cfg):
    """Prefer GPU EP only if onnxruntime was built with CUDA (pip: onnxruntime-gpu)."""
    pref = cfg.get('Settings', 'yamnet_device', fallback='auto').strip().lower()
    avail = set(ort.get_available_providers())
    cuda_ok = 'CUDAExecutionProvider' in avail
    if pref in ('cpu',):
        return ['CPUExecutionProvider']
    if pref in ('cuda', 'gpu'):
        if cuda_ok:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(
            'WARNUNG: yamnet_device=cuda, aber ONNX Runtime hat keinen CUDAExecutionProvider. '
            'Installiere onnxruntime-gpu (passend zur CUDA-Version). Nutze CPU.',
            flush=True,
        )
        return ['CPUExecutionProvider']
    if cuda_ok:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def get_runtime_settings():
    cfg = load_config()
    device = _resolve_whisper_device(cfg)
    compute = cfg.get(
        'Settings',
        'whisper_compute_type',
        fallback='float16' if device == 'cuda' else 'int8',
    )
    if device == 'cpu' and compute in ('float16', 'bfloat16'):
        compute = 'int8'
    return {
        'whisper_model': cfg.get('Settings', 'whisper_model', fallback='base'),
        'beam_size': cfg.getint('Settings', 'whisper_beam_size', fallback=5),
        'motion_width': cfg.getint('Settings', 'motion_width', fallback=640),
        'motion_height': cfg.getint('Settings', 'motion_height', fallback=360),
        'yamnet_enabled': cfg.getboolean('Settings', 'yamnet_enabled', fallback=True),
        'device': device,
        'compute': compute,
        'story_word_target': cfg.getint('Settings', 'story_word_target', fallback=15),
        'calibration_segment_seconds': max(1, cfg.getint('Settings', 'calibration_segment_seconds', fallback=20)),
        'action_motion_target': cfg.getfloat('Settings', 'action_motion_target', fallback=5.0),
        'yamnet_peak_weight': cfg.getfloat('Settings', 'yamnet_peak_weight', fallback=0.72),
        'yamnet_mean_weight': cfg.getfloat('Settings', 'yamnet_mean_weight', fallback=0.28),
        'ctranslate2_cuda_devices': _ctranslate2_cuda_devices(),
        'torch_cuda': torch.cuda.is_available(),
        'onnx_providers_available': ort.get_available_providers(),
    }


def get_whisper_model():
    s = get_runtime_settings()
    key = (s['whisper_model'], s['device'], s['compute'])
    if key not in WHISPER_MODEL_CACHE:
        WHISPER_MODEL_CACHE[key] = WhisperModel(s['whisper_model'], device=s['device'], compute_type=s['compute'])
    return WHISPER_MODEL_CACHE[key], s


def get_yamnet_session():
    global YAMNET_SESSION
    if YAMNET_SESSION is not None:
        return YAMNET_SESSION
    model_path = os.path.join(_module_dir(), 'yamnet.onnx')
    if not os.path.exists(model_path):
        return None
    cfg = load_config()
    providers = _yamnet_onnx_providers(cfg)
    try:
        YAMNET_SESSION = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if providers[0] == 'CUDAExecutionProvider':
            print(f'WARNUNG: YAMNet CUDA fehlgeschlagen ({e}), fallback CPU.', flush=True)
            YAMNET_SESSION = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            raise
    return YAMNET_SESSION


def load_yamnet_classnames():
    global YAMNET_CLASSNAMES
    if YAMNET_CLASSNAMES is not None:
        return YAMNET_CLASSNAMES
    classmap_path = os.path.join(_module_dir(), 'yamnet_class_map.csv')
    if not os.path.exists(classmap_path):
        YAMNET_CLASSNAMES = []
        return YAMNET_CLASSNAMES
    names = []
    with open(classmap_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append((row.get('display_name') or '').strip().lower())
    YAMNET_CLASSNAMES = names
    return YAMNET_CLASSNAMES


def extract_audio(video_path, start_time, duration):
    fd, temp_audio = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration), '-i', video_path,
        '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', temp_audio,
    ]
    try:
        r = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
        )
        if r.returncode != 0 or not os.path.isfile(temp_audio) or os.path.getsize(temp_audio) == 0:
            _safe_unlink(temp_audio)
            return None
        return temp_audio
    except Exception:
        _safe_unlink(temp_audio)
        return None


def get_word_count_from_audio(audio_path):
    if not audio_path:
        return 0
    model, settings = get_whisper_model()
    try:
        segments, _ = model.transcribe(audio_path, beam_size=settings['beam_size'])
        return sum(len(seg.text.split()) for seg in segments)
    except Exception:
        return 0


def read_wav_mono_16k(audio_path):
    import wave
    with wave.open(audio_path, 'rb') as wf:
        if wf.getcomptype() != 'NONE':
            raise ValueError('unsupported WAV compression')
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        if width != 2:
            raise ValueError('expected 16-bit PCM WAV')
        if ch < 1:
            raise ValueError('invalid channel count')
        frames = wf.readframes(wf.getnframes())
    if not frames:
        return np.array([], dtype=np.float32)
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        target_len = int(len(audio) * float(16000) / sr)
        audio = np.interp(
            np.linspace(0, 1, num=target_len, endpoint=False),
            np.linspace(0, 1, num=len(audio), endpoint=False),
            audio,
        ).astype(np.float32)
    return audio


def _yamnet_time_samples(sess):
    shape = sess.get_inputs()[0].shape
    if not shape:
        return YAMNET_DEFAULT_WINDOW
    ints = [d for d in shape if isinstance(d, int) and d > 1]
    return max(ints) if ints else YAMNET_DEFAULT_WINDOW


def _yamnet_build_feed(sess, chunk_1d):
    inp = sess.get_inputs()[0]
    shape = inp.shape
    x = np.ascontiguousarray(chunk_1d, dtype=np.float32)
    if not shape or len(shape) <= 1:
        return x
    target_rank = len(shape)
    while x.ndim < target_rank:
        x = np.expand_dims(x, 0)
    return x


def _run_yamnet_windows(sess, waveform_1d):
    input_name = sess.get_inputs()[0].name
    window = _yamnet_time_samples(sess)
    hop = YAMNET_DEFAULT_HOP if window == YAMNET_DEFAULT_WINDOW else max(1, window // 2)
    w = np.ascontiguousarray(waveform_1d, dtype=np.float32)
    n = int(w.size)
    if n == 0:
        return None
    chunk_list = []
    if n < window:
        pad = np.zeros(window, dtype=np.float32)
        pad[:n] = w
        chunk_list.append(pad)
    else:
        starts = []
        s = 0
        while s + window <= n:
            starts.append(s)
            s += hop
        if not starts:
            starts = [0]
        elif starts[-1] + window < n:
            starts.append(n - window)
        for st in sorted(set(starts)):
            chunk = np.zeros(window, dtype=np.float32)
            take = min(window, n - st)
            chunk[:take] = w[st : st + take]
            chunk_list.append(chunk)
    rows = []
    for chunk in chunk_list:
        out = sess.run(None, {input_name: _yamnet_build_feed(sess, chunk)})[0]
        out = np.asarray(out)
        if out.ndim > 2:
            out = out.reshape(-1, out.shape[-1])
        elif out.ndim == 1:
            out = out.reshape(1, -1)
        rows.append(out)
    return np.vstack(rows)


def weighted_group_score(classnames, peak_scores, mean_scores, terms, exact_only=False):
    settings = get_runtime_settings()
    peak_vals, mean_vals = [], []
    terms = [t.strip().lower() for t in terms if t.strip()]
    for i, cname in enumerate(classnames):
        matched = False
        for t in terms:
            if exact_only:
                if cname == t:
                    matched = True
                    break
            else:
                if t in cname:
                    matched = True
                    break
        if matched:
            peak_vals.append(float(peak_scores[i]))
            mean_vals.append(float(mean_scores[i]))
    if not peak_vals:
        return 0
    peak_val = max(peak_vals)
    mean_val = max(mean_vals) if mean_vals else 0.0
    score = peak_val * settings['yamnet_peak_weight'] + mean_val * settings['yamnet_mean_weight']
    return int(min(100, max(0, score * 100.0)))


def top_group_matches(classnames, peak_scores, mean_scores, terms, topn=5):
    rows = []
    terms = [t.strip().lower() for t in terms if t.strip()]
    for i, cname in enumerate(classnames):
        if any(t in cname for t in terms):
            combo = float(peak_scores[i]) * 0.72 + float(mean_scores[i]) * 0.28
            rows.append((cname, combo, float(peak_scores[i]), float(mean_scores[i])))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:topn]


def get_yamnet_group_scores_from_audio(audio_path):
    settings = get_runtime_settings()
    empty = {
        'speech_like': 0,
        'music_like': 0,
        'silence_like': 0,
        'moan_like': 0,
        'breath_like': 0,
        'scream_like': 0,
        'human_vocal_like': 0,
        'top_audio_classes': [],
        'top_vocal_matches': []
    }
    if not settings['yamnet_enabled'] or not audio_path:
        return empty
    sess = get_yamnet_session()
    classnames = load_yamnet_classnames()
    if not sess or not classnames:
        return empty
    try:
        waveform = read_wav_mono_16k(audio_path)
        if waveform.size == 0:
            return empty
        scores = _run_yamnet_windows(sess, waveform)
        if scores is None or scores.size == 0:
            return empty
        mean_scores = np.mean(scores, axis=0)
        peak_scores = np.max(scores, axis=0)

        speech_terms_strict = ['male speech', 'female speech', 'child speech']
        speech_terms_broad = ['speech', 'conversation', 'narration', 'monologue']
        music_terms = ['music', 'song', 'musical']
        silence_terms = ['silence']
        moan_terms = ['moan', 'moaning', 'groan', 'grunt', 'whimper']
        breath_terms = ['breath', 'breathing', 'gasp', 'wheeze', 'sigh', 'pant', 'respiratory sounds']
        scream_terms = ['scream', 'screaming', 'shout', 'yell', 'shriek', 'wail']
        human_vocal_terms = ['groan', 'grunt', 'moan', 'pant', 'breath', 'breathing', 'gasp', 'wheeze', 'sigh', 'scream', 'shout', 'yell', 'wail', 'whimper', 'sob']

        speech_strict = weighted_group_score(classnames, peak_scores, mean_scores, speech_terms_strict)
        speech_broad = weighted_group_score(classnames, peak_scores, mean_scores, speech_terms_broad)
        music_like = weighted_group_score(classnames, peak_scores, mean_scores, music_terms)
        silence_like = weighted_group_score(classnames, peak_scores, mean_scores, silence_terms)
        moan_like = weighted_group_score(classnames, peak_scores, mean_scores, moan_terms)
        breath_like = weighted_group_score(classnames, peak_scores, mean_scores, breath_terms)
        scream_like = weighted_group_score(classnames, peak_scores, mean_scores, scream_terms)
        human_vocal_like = weighted_group_score(classnames, peak_scores, mean_scores, human_vocal_terms)

        speech_like = clamp_int((speech_strict * 0.75 + speech_broad * 0.25) - (moan_like * 0.25 + breath_like * 0.20 + scream_like * 0.15))
        human_vocal_like = clamp_int(human_vocal_like - speech_strict * 0.20)

        top_indices = np.argsort(peak_scores)[-10:][::-1]
        top_audio_classes = [
            {'class': classnames[i], 'peak': round(float(peak_scores[i]) * 100, 2), 'mean': round(float(mean_scores[i]) * 100, 2)}
            for i in top_indices
        ]
        vocal_terms_all = list(set(moan_terms + breath_terms + scream_terms + human_vocal_terms))
        top_vocal_matches = [
            {'class': cname, 'combo': round(combo * 100, 2), 'peak': round(pk * 100, 2), 'mean': round(mn * 100, 2)}
            for cname, combo, pk, mn in top_group_matches(classnames, peak_scores, mean_scores, vocal_terms_all)
        ]

        return {
            'speech_like': speech_like,
            'music_like': music_like,
            'silence_like': silence_like,
            'moan_like': moan_like,
            'breath_like': breath_like,
            'scream_like': scream_like,
            'human_vocal_like': human_vocal_like,
            'top_audio_classes': top_audio_classes,
            'top_vocal_matches': top_vocal_matches,
        }
    except Exception:
        return empty


def get_motion_score(video_path, start_time, duration):
    settings = get_runtime_settings()
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            return 0.0
        target_frame = int(start_time * fps)
        frame_count = int(fps * duration)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame1 = cap.read()
        if not ret:
            return 0.0
        prev = cv2.resize(
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
            (settings['motion_width'], settings['motion_height']),
        )
        total_motion, frames_processed = 0.0, 0
        skip_rate = 2
        for _ in range(0, frame_count - 1, skip_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame + frames_processed + skip_rate)
            ret, frame2 = cap.read()
            if not ret:
                break
            nxt = cv2.resize(
                cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                (settings['motion_width'], settings['motion_height']),
            )
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_motion += float(np.mean(mag))
            frames_processed += 1
            prev = nxt
        return total_motion / frames_processed if frames_processed > 0 else 0.0
    finally:
        cap.release()


def analyze_segment(video_path, start_time, duration=20):
    audio_path = extract_audio(video_path, start_time, duration)
    try:
        words = get_word_count_from_audio(audio_path)
        motion = get_motion_score(video_path, start_time, duration)
        yam = get_yamnet_group_scores_from_audio(audio_path)
        settings = get_runtime_settings()

        ref = max(1e-6, float(settings['calibration_segment_seconds']))
        dur_s = max(float(duration), 1e-6)
        target_wpm = settings['story_word_target'] * 60.0 / ref
        wpm = words * 60.0 / dur_s
        word_score = min(100.0, (wpm / target_wpm) * 100.0) if target_wpm > 0 else 0.0
        motion_target = max(0.1, settings['action_motion_target'])

        motion_score_mapped = min(100.0, motion / float(motion_target) * 100.0)

        speech_like = yam.get('speech_like', 0)
        music_like = yam.get('music_like', 0)
        silence_like = yam.get('silence_like', 0)
        moan_like = yam.get('moan_like', 0)
        breath_like = yam.get('breath_like', 0)
        scream_like = yam.get('scream_like', 0)
        human_vocal_like = yam.get('human_vocal_like', 0)

        # story_score: word arm = WPM vs target WPM; rest = YAMNet %-scores (already normalized)
        story_score = clamp_int(word_score * 0.55 + speech_like * 0.45 - moan_like * 0.20 - breath_like * 0.15)
        action_score = clamp_int(motion_score_mapped * 0.82 + music_like * 0.08 + scream_like * 0.10)
        low_word_bonus = max(0.0, 100.0 - min(100.0, word_score))
        sexual_vocal_score = clamp_int(moan_like * 0.45 + scream_like * 0.18 + breath_like * 0.27 + low_word_bonus * 0.10)
        if speech_like > 65 and wpm >= target_wpm:
            sexual_vocal_score = clamp_int(sexual_vocal_score * 0.72)

        _print_hw_profile(settings)

        return {
            'word_count': words,
            'wpm': round(wpm, 2),
            'target_wpm': round(target_wpm, 2),
            'word_score': round(word_score, 2),
            'motion_score': round(motion, 4),
            'motion_score_mapped': round(motion_score_mapped, 2),
            'speech_percent': speech_like,
            'music_percent': music_like,
            'silence_percent': silence_like,
            'moan_percent': moan_like,
            'breath_percent': breath_like,
            'scream_percent': scream_like,
            'human_vocal_percent': human_vocal_like,
            'story_score': story_score,
            'action_score': action_score,
            'sexual_vocal_score': sexual_vocal_score,
            'top_audio_classes': yam.get('top_audio_classes', []),
            'top_vocal_matches': yam.get('top_vocal_matches', []),
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def _print_hw_profile(settings):
    """Einmal pro Prozess: tatsächliche Beschleuniger für Segmentierung."""
    global _HW_PROFILE_PRINTED
    if _HW_PROFILE_PRINTED:
        return
    _HW_PROFILE_PRINTED = True
    if not settings.get('yamnet_enabled', True):
        y_ep = 'disabled (ini)'
    else:
        y_sess = get_yamnet_session()
        if y_sess:
            y_ep = y_sess.get_providers()[0]
        else:
            y_ep = 'no yamnet.onnx'
    nct = settings.get('ctranslate2_cuda_devices', 0)
    print(
        f'SCENECUT_HW: faster-whisper device={settings["device"]} compute={settings["compute"]} '
        f'(ctranslate2 CUDA GPUs={nct}; torch.cuda={settings.get("torch_cuda")}) | '
        f'YAMNet onnx active_EP={y_ep} | motion=CPU(OpenCV Farneback, pip wheels ohne CUDA)',
        flush=True,
    )


analyzesegment = analyze_segment
