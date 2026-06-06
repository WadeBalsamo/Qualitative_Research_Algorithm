"""
process/speaker_anonymization.py
---------------------------------
Speaker anonymization key loading, extracted from orchestrator.py.
"""
import json
import os


def load_speaker_map(meta_dir: str, config) -> tuple:
    """Load speaker anonymization map from meta/ dir or imported config path.

    Priority: existing key in meta/ (stability on re-runs) > key imported via
    config.speaker_anonymization_key_path (first run with an imported key).

    Returns (existing_speaker_map, use_unknown_prefix):
      existing_speaker_map : {raw_name: (role, anonymized_id)}
      use_unknown_prefix   : True when the key was seeded from an imported file
    """
    speaker_key_path = os.path.join(meta_dir, 'speaker_anonymization_key.json')
    existing_speaker_map: dict = {}
    use_unknown_prefix: bool = False

    if os.path.exists(speaker_key_path):
        try:
            with open(speaker_key_path) as _f:
                _raw_key = json.load(_f)
            existing_speaker_map = {
                name: (entry['role'], entry['anonymized_id'])
                for name, entry in _raw_key.items()
            }
        except Exception:
            existing_speaker_map = {}
        use_unknown_prefix = bool(getattr(config, 'speaker_anonymization_key_path', None))

    elif getattr(config, 'speaker_anonymization_key_path', None):
        try:
            with open(config.speaker_anonymization_key_path) as _f:
                _raw_key = json.load(_f)
            existing_speaker_map = {
                name: (entry['role'], entry['anonymized_id'])
                for name, entry in _raw_key.items()
            }
            use_unknown_prefix = True
        except Exception as e:
            print(f"  Warning: Could not load speaker_anonymization_key_path: {e}")
            existing_speaker_map = {}

    return existing_speaker_map, use_unknown_prefix
