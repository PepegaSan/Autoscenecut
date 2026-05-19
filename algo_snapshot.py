"""
Local tuning only: threshold INI values + analysis-related settings + logic pointers.

Artifacts under output/ (repo .gitignore) — not documented for end users / GitHub README.
Written to output/algo_runs/*.json and embedded in last_autocut_checkpoint.json as "algo_snapshot".
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

SNAPSHOT_FORMAT_VERSION = 1

# Settings keys that influence segment scoring or category rules (not export-only).
ANALYSIS_SETTING_KEYS = (
    'interval_seconds',
    'calibration_segment_seconds',
    'story_word_target',
    'action_motion_target',
    'method_a_scene_profile',
    'motion_width',
    'motion_height',
    'yamnet_enabled',
    'yamnet_peak_weight',
    'yamnet_mean_weight',
    'whisper_model',
    'whisper_device',
    'whisper_compute_type',
    'whisper_beam_size',
    'save_segment_analysis_csv',
)

# Human-oriented reference; keep formulas in sync when you change analyzer/autocut/gui.
LOGIC_REFERENCE: dict[str, Any] = {
    'code_locations': [
        {
            'file': 'analyzer_nvidia.py',
            'symbol': '_analyze_segment_impl',
            'role': 'Raw story / action / vocal scores per segment (Whisper, motion, YAMNet).',
        },
        {
            'file': 'autocut_nvidia.py',
            'symbol': 'decide_category',
            'role': 'Maps metrics + INI thresholds to final_category (dialogue / vocal / action / music / silence).',
        },
        {
            'file': 'gui_nvidia.py',
            'symbol': 'calc_method_a_thresholds',
            'role': 'Optional auto-thresholds from sampled segments (percentiles + clamps).',
        },
    ],
    'segment_scoring_formulas': {
        'story_score': 'clamp_int(word_score*0.55 + speech_like*0.45 - moan_like*0.20 - breath_like*0.15)',
        'action_score': 'clamp_int(motion_score_mapped*0.82 + music_like*0.08 + scream_like*0.10)',
        'sexual_vocal_score': 'clamp_int(moan*0.45 + scream*0.18 + breath*0.27 + low_word_bonus*0.10); damp if speech_like>65 and wpm>=target_wpm',
        'motion_score_mapped': 'min(100, motion / action_motion_target * 100)',
        'word_score': 'min(100, (wpm / target_wpm) * 100) with target_wpm from story_word_target and calibration_segment_seconds',
        'yamnet_group_score': 'peak* yamnet_peak_weight + mean* yamnet_mean_weight (see weighted_group_score)',
    },
    'decide_category_formulas': {
        'vocal_penalty': 'story * vocal_story_penalty_factor + speech * vocal_speech_penalty_factor',
        'action_penalty': 'story * action_story_penalty_factor + speech * action_speech_penalty_factor',
        'vocal_effective': 'clamp_int(vocal_raw - vocal_penalty)',
        'action_effective': 'clamp_int(action_raw - action_penalty)',
        'vocal_signal': 'max(vocal_effective, clamp_int(human_vocal * 0.50))',
        'wpm_floor': 'wpm_threshold_from_calibrated_words(n_words, calibration_segment_seconds)',
    },
    'method_a_thresholds': {
        'samples': '10 segments; start positions spread over duration with jitter (gui_nvidia._auto_thresholds_method_a_thread)',
        'new_story': 'clamp_int(max(12, min(95, story_p60 * 0.95))) with story_p60 = percentile(story_vals, 0.60)',
        'new_action': 'clamp_int(max(8, min(95, action_p70 * 0.90))) with action_p70 = percentile(action_vals, 0.70)',
        'new_vocal': 'clamp_int(max(6, min(95, vocal_p70 * 0.90))) with vocal_p70 = percentile(vocal_vals, 0.70)',
        'effective_scores_in_sampling': 'Same penalty factors as INI: vocal_eff, action_eff vs story/speech (Method A thread)',
    },
}


def _section_dict(cfg, section: str) -> dict[str, str]:
    if not cfg.has_section(section):
        return {}
    return {k: cfg.get(section, k) for k in cfg.options(section)}


def _analysis_settings(cfg) -> dict[str, str]:
    out: dict[str, str] = {}
    if not cfg.has_section('Settings'):
        return out
    for k in ANALYSIS_SETTING_KEYS:
        if cfg.has_option('Settings', k):
            out[k] = cfg.get('Settings', k)
    return out


def build_algo_run_snapshot(
    cfg,
    video_path: Optional[str] = None,
    cfg_path: str = 'config_nvidia.ini',
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a JSON-serializable dict (INI-derived + static logic reference)."""
    snap: dict[str, Any] = {
        'snapshot_format_version': SNAPSHOT_FORMAT_VERSION,
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'video_path': os.path.abspath(video_path) if video_path else None,
        'config_ini': os.path.abspath(cfg_path) if cfg_path else None,
        'thresholds': _section_dict(cfg, 'Thresholds'),
        'categories': _section_dict(cfg, 'Categories'),
        'analysis_settings': _analysis_settings(cfg),
        'logic_reference': LOGIC_REFERENCE,
    }
    if cfg_path:
        try:
            snap['config_ini_mtime_utc'] = datetime.fromtimestamp(
                os.path.getmtime(cfg_path), tz=timezone.utc
            ).isoformat()
        except OSError:
            snap['config_ini_mtime_utc'] = None
    if extra:
        snap['extra'] = dict(extra)
    return snap


def _safe_video_stem(video_path: Optional[str]) -> str:
    base = os.path.splitext(os.path.basename(video_path or 'run'))[0]
    s = re.sub(r'[^\w.\-]+', '_', base, flags=re.ASCII)
    return (s[:120] if s else 'run') or 'run'


def write_algo_run_artifact(
    snapshot: Mapping[str, Any],
    video_path: Optional[str],
    *,
    prefix: str = 'algo',
    output_root: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Write one timestamped JSON under <output_root or output/algo_runs/>.
    Returns absolute path written, or None if nothing written.
    """
    root = Path(output_root) if output_root is not None else Path('output') / 'algo_runs'
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_video_stem(video_path)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f'{prefix}_{stem}_{ts}.json'
    path = root / name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(snapshot), f, indent=2, ensure_ascii=False)
    ap = str(path.resolve())
    print(f'ALGO_SNAPSHOT:{ap}', flush=True)
    return ap
