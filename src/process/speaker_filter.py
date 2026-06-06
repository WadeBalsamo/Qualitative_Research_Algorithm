"""
process/speaker_filter.py
-------------------------
Segment filtering by speaker role, extracted from orchestrator.py.
"""
from typing import List

from classification_tools.data_structures import Segment


def apply_speaker_filter(segments: List[Segment], filter_cfg) -> List[Segment]:
    """Return segments selected for classification based on SpeakerFilterConfig.

    'none'    — return all segments unchanged
    'exclude' — drop therapist segments (by role) from classification.
                Therapist sentences are already filtered before segmentation,
                but this catches any remaining therapist segments.
    """
    mode = getattr(filter_cfg, 'mode', 'none')
    speakers = set(getattr(filter_cfg, 'speakers', []))

    if mode == 'none' or not speakers:
        return segments

    if mode == 'exclude':
        return [s for s in segments if s.speaker != 'therapist']

    return segments
