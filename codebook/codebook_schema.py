"""
codebook_schema.py
------------------
Generalizable data structures for qualitative research codebooks.

A Codebook is a collection of CodeDefinition objects, each representing
a single code with category name, description, inclusion/exclusion
criteria, and domain membership.
"""

import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeDefinition:
    """A single code in a qualitative research codebook."""
    code_id: str
    category: str
    domain: str
    description: str
    subcodes: List[str]
    inclusive_criteria: str
    exclusive_criteria: str
    exemplar_utterances: List[str] = field(default_factory=list)


@dataclass
class CodeAssignment:
    """Result of applying a single code to a segment."""
    code_id: str
    category: str
    confidence: float
    justification: str = ""
    method: str = ""  # 'embedding', 'llm', or 'ensemble'


@dataclass
class Codebook:
    """A complete qualitative research codebook."""
    name: str
    version: str
    description: str
    codes: List[CodeDefinition] = field(default_factory=list)
    domains: Optional[Dict[str, List[str]]] = None

    def get_codes_by_domain(self, domain: str) -> List[CodeDefinition]:
        """Filter codes by domain."""
        return [c for c in self.codes if c.domain == domain]

    def build_name_to_id_map(self) -> Dict[str, str]:
        """
        Build a lookup table mapping category names and code IDs to
        canonical code_id strings (all lowercased keys).
        """
        mapping: Dict[str, str] = {}
        for code in self.codes:
            mapping[code.category.lower()] = code.code_id
            mapping[code.code_id.lower()] = code.code_id
        return mapping

    @property
    def domain_names(self) -> List[str]:
        """Return sorted unique domain names."""
        return sorted(set(c.domain for c in self.codes))

    def to_prompt_string(self, randomize: bool = False) -> str:
        """
        Format all codes for LLM prompting.

        Each code is rendered as:
            Category: description
            Subcodes: subcode1, subcode2, ...
            Include when: inclusive_criteria
            Exclude when: exclusive_criteria
        """
        codes = list(self.codes)
        if randomize:
            codes = list(codes)
            random.shuffle(codes)

        parts = []
        for code in codes:
            subcodes_str = ', '.join(code.subcodes) if code.subcodes else 'N/A'
            block = (
                f"{code.category}: {code.description} "
                f"Subcodes: {subcodes_str}. "
                f"Include when: {code.inclusive_criteria} "
                f"Exclude when: {code.exclusive_criteria}"
            ).replace('\n', ' ').replace('  ', ' ')
            parts.append(block)

        return '\n\n'.join(parts)

    def to_embedding_targets(self) -> List[Dict[str, str]]:
        """
        Format codes for embedding-based comparison.

        Returns a list of dicts with keys:
          - 'code_id':    canonical code identifier
          - 'category':   human-readable category name
          - 'definition': code name + subcodes + description
          - 'criteria':   inclusive criteria text
          - 'exemplars':  space-joined exemplar utterances (may be empty)
        """
        targets = []
        for code in self.codes:
            targets.append({
                'code_id': code.code_id,
                'category': code.category,
                'definition': (
                    code.category + ', '
                    + ', '.join(code.subcodes) + ' '
                    + code.description
                ),
                'criteria': code.inclusive_criteria,
                'exemplars': ' '.join(code.exemplar_utterances) if code.exemplar_utterances else '',
            })
        return targets


def slugify(name: str) -> str:
    """Convert a category name to a machine-readable code_id slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug
