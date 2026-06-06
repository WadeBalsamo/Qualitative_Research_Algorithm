"""
process/speaker_walkthrough.py
------------------------------
Interactive CLI walkthrough for adding new speakers to the persistent
anonymization key during incremental data addition.

Invoked by qra add-data (and qra run when new speakers are detected).
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from . import output_paths as _paths
from ._freeze import write_frozen
from .speaker_anonymization import load_speaker_map
from .transcript_ingestion import peek_speaker_labels


_PARTICIPANT_ID_RE = re.compile(r'(?:Participant_MM|participant_|unknownparticipant_)(\d+)$', re.IGNORECASE)
_THERAPIST_ID_RE = re.compile(r'therapist_(\d+)$', re.IGNORECASE)
_STAFF_ID_RE = re.compile(r'staff_(\d+)$', re.IGNORECASE)


def _next_id(existing_map: Dict[str, Tuple[str, str]], role: str) -> str:
    """Compute the next un-used anonymized_id for the given role.

    Participants use the Participant_MM### format (matching cohort-style IDs).
    Therapists use therapist_N. Staff use staff_N.
    """
    max_n = 0
    if role == 'participant':
        for _r, anon_id in existing_map.values():
            m = _PARTICIPANT_ID_RE.search(anon_id or '')
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f'Participant_MM{max_n + 1:03d}'
    if role == 'therapist':
        for _r, anon_id in existing_map.values():
            m = _THERAPIST_ID_RE.search(anon_id or '')
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f'therapist_{max_n + 1}'
    if role == 'staff':
        for _r, anon_id in existing_map.values():
            m = _STAFF_ID_RE.search(anon_id or '')
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f'staff_{max_n + 1}'
    raise ValueError(f"unknown role: {role!r}")


def discover_new_speakers(
    new_session_files: List[str],
    existing_map: Dict[str, Tuple[str, str]],
    max_samples_per_speaker: int = 5,
) -> Dict[str, List[str]]:
    """Return {raw_label: [sample_utterances]} for speakers not in existing_map.

    Samples are collected across all new_session_files. If the same label
    appears in multiple files, samples from the first file are used.
    """
    new_speakers: Dict[str, List[str]] = {}
    for path in new_session_files:
        labels = peek_speaker_labels(path, max_samples_per_speaker=max_samples_per_speaker)
        for label, samples in labels.items():
            if label in existing_map:
                continue
            if label in new_speakers:
                continue
            new_speakers[label] = samples
    return new_speakers


def _prompt(prompt_text: str, *, default: Optional[str] = None) -> str:
    """Wrapper around input() that supports a default value on empty input."""
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt_text}{suffix}: ").strip()
    return val if val else (default or "")


def _prompt_role(label: str, samples: List[str]) -> str:
    """Ask the user whether the new speaker is participant / therapist / staff."""
    print()
    print(f"  New speaker: {label!r}  ({len(samples)} sample utterance{'s' if len(samples) != 1 else ''})")
    for s in samples:
        print(f"    - {s}")
    while True:
        choice = _prompt("  Role? [p]articipant / [t]herapist / [s]taff (excluded from both)", default='p').lower()
        if choice in ('p', 'participant'):
            return 'participant'
        if choice in ('t', 'therapist'):
            return 'therapist'
        if choice in ('s', 'staff'):
            return 'staff'
        print("  Please enter 'p', 't', or 's'.")


def _prompt_anonymized_id(
    role: str,
    existing_map: Dict[str, Tuple[str, str]],
    suggested: str,
) -> str:
    """Ask the user to accept the suggested ID, override, or map to an existing one.

    Returns the chosen anonymized_id. May return an ID already present in
    existing_map when the user chose 'map' — caller must accept the duplicate
    mapping (two raw labels -> same anonymized_id).
    """
    while True:
        ans = _prompt(
            f"  Suggested anonymized_id: {suggested}\n"
            "    [Enter to accept, type an override, or 'map' to link to an existing ID]",
            default=suggested,
        )
        if ans.lower() == 'map':
            same_role = sorted({anon for r, anon in existing_map.values() if r == role})
            if not same_role:
                print(f"  No existing {role} IDs to map to. Pick a different option.")
                continue
            print(f"  Existing {role} IDs:")
            for i, anon in enumerate(same_role, 1):
                print(f"    [{i}] {anon}")
            pick = _prompt("  Pick a number")
            try:
                idx = int(pick) - 1
            except ValueError:
                print("  Invalid number; try again.")
                continue
            if 0 <= idx < len(same_role):
                return same_role[idx]
            print("  Out of range; try again.")
            continue
        if ans:
            return ans
        return suggested


def run_speaker_walkthrough(
    config,
    new_session_files: List[str],
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple[str, str]]:
    """Interactively extend the speaker anonymization key for new transcripts.

    1. Loads the existing key from <output_dir>/02_meta/speaker_anonymization_key.json
       (or from config.speaker_anonymization_key_path if no project key exists yet).
    2. For each new speaker label across new_session_files, prompts the user
       for role + anonymized_id and records the mapping.
    3. Atomically writes the updated key back to the project meta dir.

    Returns the full updated speaker map dict (raw_label -> (role, anonymized_id)).
    No-op (returns existing map unchanged, no file write) if no new speakers
    are detected.
    """
    run_dir = output_dir if output_dir is not None else getattr(config, 'output_dir', None)
    if run_dir is None:
        raise ValueError("run_speaker_walkthrough requires output_dir or config.output_dir")

    meta_dir = _paths.meta_dir(run_dir)
    os.makedirs(meta_dir, exist_ok=True)

    existing_map, _use_unknown_prefix = load_speaker_map(meta_dir, config)
    updated_map: Dict[str, Tuple[str, str]] = dict(existing_map)

    new_speakers = discover_new_speakers(new_session_files, existing_map)
    if not new_speakers:
        print(f"  Speaker walkthrough: 0 new speakers detected across {len(new_session_files)} files.")
        return updated_map

    print()
    print(f"  Speaker walkthrough: {len(new_speakers)} new speaker label(s) found across {len(new_session_files)} new file(s).")

    # Per-speaker prompting
    for label, samples in new_speakers.items():
        role = _prompt_role(label, samples)
        suggested = _next_id(updated_map, role)
        anon_id = _prompt_anonymized_id(role, updated_map, suggested)
        updated_map[label] = (role, anon_id)
        print(f"    -> {label!r} mapped to {anon_id} ({role})")

    # Atomic write
    key_path = os.path.join(meta_dir, 'speaker_anonymization_key.json')
    payload = {
        raw: {'role': role, 'anonymized_id': anon_id}
        for raw, (role, anon_id) in updated_map.items()
    }
    def _write(fh):
        json.dump(payload, fh, indent=2)
    write_frozen(key_path, _write, force=True)
    print(f"  Wrote {len(payload)} entries to {key_path}")

    return updated_map
