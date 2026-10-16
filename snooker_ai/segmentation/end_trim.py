"""Reversible editing of saved endings, independent of detection evidence."""
from math import isfinite

from snooker_ai.config import Config
from snooker_ai.types import ShotRecord


def trim_saved_end(shot: ShotRecord, seconds: float, config: Config, *,
                   duration: float, next_start: float | None = None,
                   reset: bool = False) -> tuple[bool, bool]:
    """Set an absolute offset from a stable ending; return changed, limited.

    Confirmed automatic ends can restore their existing 2/3-second offset.
    Unconfirmed and manually edited ends never acquire extra unseen footage.
    Physical stop measurements and review/classification flags stay untouched.
    """
    if not isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError('End trim must be between 0 and 3600 seconds')
    previous = shot.evidence.get('end_trim')
    old_end = shot.clip_end
    if reset:
        if not previous:
            return False, False
        shot.clip_end = float(previous['original_clip_end'])
        shot.evidence['end_trim'] = {**previous, 'active': False, 'seconds': None,
            'applied_seconds': float(previous['reference_end'])-shot.clip_end}
    else:
        if previous:
            original = float(previous['original_clip_end'])
            reference = float(previous['reference_end'])
        else:
            original = reference = shot.clip_end
            if not shot.user_modified and shot.evidence.get('stop_confirmed'):
                usable = min(duration, float(shot.evidence.get('usable_source_end_timestamp', duration)))
                if next_start is not None and next_start > shot.cue_strike:
                    usable = min(usable, next_start)
                reference = max(original, min(shot.physical_stop_timestamp, usable))
        mode = config.mode_settings('strict')
        padding = float(shot.evidence.get('minimum_clip_transition_padding_seconds',
            2*float(config.get('export.transition_seconds', .24))
            if config.get('export.transition', 'cut') == 'mix' else 0.))
        minimum = max(shot.clip_start + float(shot.evidence.get('minimum_clip_seconds',
                          mode.get('minimum_clip_seconds', 4.))) + padding,
                      shot.cue_strike + float(shot.evidence.get('minimum_strike_visibility_seconds',
                          mode.get('minimum_strike_visibility_seconds', 2.))))
        # A source cut or a previously short manual clip wins over extending
        # footage just to satisfy a playback minimum.
        floor = min(reference, minimum)
        shot.clip_end = max(floor, reference-seconds)
        shot.evidence['end_trim'] = {
            'original_clip_end': original, 'reference_end': reference,
            'seconds': seconds, 'applied_seconds': reference-shot.clip_end, 'active': True,
        }
    shot.clip_end_timestamp = shot.clip_end
    shot.user_modified = True
    changed = abs(old_end-shot.clip_end) > 1e-6
    limited = not reset and reference-shot.clip_end < seconds-1e-6
    return changed, limited
