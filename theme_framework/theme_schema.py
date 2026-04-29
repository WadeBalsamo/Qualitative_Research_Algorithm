"""
theme_schema.py
---------------
Generalizable data structures for theme/stage classification frameworks.

A ThemeFramework is a collection of ThemeDefinition objects, each
representing a theoretical stage or theme with operational definitions,
prototypical features, and exemplar utterances for LLM prompting.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ThemeDefinition:
    """A single theme/stage in a theoretical framework."""
    theme_id: int
    key: str
    name: str
    short_name: str
    prompt_name: str
    definition: str
    prototypical_features: List[str]
    distinguishing_criteria: str
    exemplar_utterances: List[str]
    subtle_utterances: List[str] = field(default_factory=list)
    adversarial_utterances: List[str] = field(default_factory=list)
    word_prototypes: List[str] = field(default_factory=list)
    color: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class ThemeFramework:
    """A complete theme/stage classification framework."""
    name: str
    version: str
    description: str
    themes: List[ThemeDefinition] = field(default_factory=list)
    categories: Optional[Dict[str, List[str]]] = None

    @property
    def num_themes(self) -> int:
        return len(self.themes)

    def get_theme_by_id(self, theme_id: int) -> Optional[ThemeDefinition]:
        """Look up a theme by its integer ID."""
        for t in self.themes:
            if t.theme_id == theme_id:
                return t
        return None

    def build_name_to_id_map(self) -> Dict[str, int]:
        """
        Build a comprehensive lookup table for parsing LLM outputs.

        Maps full names, short names, prompt names, keys, and any
        registered aliases to integer theme IDs (all lowercased).
        """
        mapping: Dict[str, int] = {}
        for t in self.themes:
            mapping[t.name.lower()] = t.theme_id
            mapping[t.short_name.lower()] = t.theme_id
            mapping[t.prompt_name.lower()] = t.theme_id
            mapping[t.key.lower()] = t.theme_id
            for alias in t.aliases:
                mapping[alias.lower()] = t.theme_id
        return mapping

    def build_id_to_short_map(self) -> Dict[int, str]:
        """Map theme_id -> short_name."""
        return {t.theme_id: t.short_name for t in self.themes}

    def to_json(self) -> Dict:
        """Export as JSON structure for stage/theme definitions output."""
        return {
            'framework': self.name,
            'version': self.version,
            'themes': [
                {
                    'theme_id': t.theme_id,
                    'key': t.key,
                    'name': t.name,
                    'short_name': t.short_name,
                    'definition': t.definition,
                    'prototypical_features': t.prototypical_features,
                    'distinguishing_criteria': t.distinguishing_criteria,
                    'exemplar_utterances': t.exemplar_utterances,
                    'subtle_utterances': t.subtle_utterances,
                    'adversarial_utterances': t.adversarial_utterances,
                }
                for t in sorted(self.themes, key=lambda t: t.theme_id)
            ],
        }

    def to_prompt_string(
        self,
        randomize: bool = False,
        zero_shot: bool = False,
        n_exemplars: Optional[int] = None,
        include_subtle: bool = True,
        n_subtle: Optional[int] = None,
        include_adversarial: bool = True,
        n_adversarial: Optional[int] = None,
    ) -> str:
        """
        Format themes for LLM prompting.

        When zero_shot=True: definition, features, and key distinction only — no examples.
        When zero_shot=False (default): all exemplar types included. None counts = all available.
        """
        themes = list(self.themes)
        if randomize:
            random.shuffle(themes)

        parts = []
        for t in themes:
            features = '; '.join(t.prototypical_features)
            block = (
                f"{t.prompt_name.capitalize()}: {t.definition} "
                f"Prototypical features: {features}. "
                f"Key distinction: {t.distinguishing_criteria}."
            ).replace('\n', ' ').replace('  ', ' ')

            if not zero_shot:
                ex = (t.exemplar_utterances if n_exemplars is None
                      else t.exemplar_utterances[:n_exemplars])
                if ex:
                    block += f" Examples: {' | '.join(ex)}"
                if include_subtle and t.subtle_utterances:
                    sub = (t.subtle_utterances if n_subtle is None
                           else t.subtle_utterances[:n_subtle])
                    if sub:
                        block += f" Edge cases: {' | '.join(sub)}"
                if include_adversarial and t.adversarial_utterances:
                    adv = (t.adversarial_utterances if n_adversarial is None
                           else t.adversarial_utterances[:n_adversarial])
                    if adv:
                        block += f" Watch-outs (boundary cases): {' | '.join(adv)}"

            parts.append(block)

        return '\n\n'.join(parts)
