"""
markdown_loader.py
------------------
Parse VAAMR.md / PURER.md (and any framework markdown following the same
PARSER CONTRACT) into ThemeFramework / ThemeDefinition dataclasses.

Entry point:
    load_framework_md(path: Path) -> ThemeFramework

Supports both heading styles:
  VAAMR  →  ## Stage N — ShortName
  PURER  →  ## Move N — KEY — ShortName

Parser contract is specified in the HTML-comment block at the top of each
framework markdown file.  HTML comments are stripped before data extraction.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .theme_schema import ThemeDefinition, ThemeFramework


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _strip_html_comments(text: str) -> str:
    """Remove <!-- ... --> blocks (may span multiple lines)."""
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    """
    Split YAML frontmatter from the markdown body.

    Returns (frontmatter_dict, body_text).
    Raises ValueError if the file does not open with '---'.
    """
    if not text.startswith('---'):
        raise ValueError("Markdown file does not start with YAML frontmatter '---'")
    end = text.index('\n---\n', 3)
    fm_text = text[4:end]
    body = text[end + 5:]
    return yaml.safe_load(fm_text), body


def _normalize_prose(text: str) -> str:
    """Collapse whitespace and strip leading/trailing space from prose."""
    text = re.sub(r'(?m)^\s*---\s*$', '', text)  # strip horizontal rule separators
    return re.sub(r'\s+', ' ', text).strip()


def _parse_bullets(text: str) -> List[str]:
    """Parse a '- item' bullet list block into a plain list."""
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('- '):
            items.append(stripped[2:])
    return items


def _parse_blockquotes(text: str) -> List[str]:
    """
    Parse blockquote paragraphs into a list of utterance strings.

    Each contiguous run of '> ' lines is one utterance (continuation
    lines joined with a space).  Blank lines and non-blockquote lines
    act as paragraph separators.
    """
    items: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        rline = line.rstrip()
        if rline.startswith('> '):
            current.append(rline[2:])
        elif rline == '>':
            current.append('')
        else:
            if current:
                items.append(' '.join(current).strip())
                current = []
    if current:
        items.append(' '.join(current).strip())
    return [u for u in items if u]


def _parse_word_prototypes(text: str) -> List[str]:
    """
    Parse the Word Prototypes section.

    Expects a single comma-separated line.  Skips blank lines and
    horizontal rules (---) that may appear at the end of a theme block.
    """
    for line in text.splitlines():
        line = line.strip()
        if line and line != '---' and not line.startswith('#'):
            return [w.strip() for w in line.split(', ') if w.strip()]
    return []


# ---------------------------------------------------------------------------
# Theme block parser
# ---------------------------------------------------------------------------

def _extract_yaml_block(text: str) -> Tuple[dict, str]:
    """
    Extract the first ```yaml ... ``` fenced block.

    Returns (parsed_dict, remaining_text_after_block).
    """
    m = re.search(r'^```yaml\n(.*?)^```', text, re.MULTILINE | re.DOTALL)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)), text[m.end():]


def _split_h3_sections(text: str) -> Dict[str, str]:
    """
    Split text at every '### Heading' boundary.

    Returns a dict mapping heading name → section body (stripped).
    Content before the first ### heading is discarded.
    """
    pattern = re.compile(r'^### (.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]
    return sections


def _parse_theme_block(block_text: str) -> ThemeDefinition:
    """
    Parse a single theme / move block (text after the ## heading line)
    into a ThemeDefinition.
    """
    yaml_data, remaining = _extract_yaml_block(block_text)
    sections = _split_h3_sections(remaining)

    definition = _normalize_prose(sections.get('Definition', ''))
    features = _parse_bullets(sections.get('Prototypical Features', ''))
    criteria = _normalize_prose(sections.get('Distinguishing Criteria', ''))
    exemplars = _parse_blockquotes(sections.get('Exemplar Utterances', ''))
    subtle = _parse_blockquotes(sections.get('Subtle Utterances', ''))
    adversarial = _parse_blockquotes(sections.get('Adversarial Utterances', ''))
    word_protos = _parse_word_prototypes(sections.get('Word Prototypes', ''))

    return ThemeDefinition(
        theme_id=int(yaml_data['theme_id']),
        key=str(yaml_data['key']),
        name=str(yaml_data['name']),
        short_name=str(yaml_data['short_name']),
        prompt_name=str(yaml_data['prompt_name']),
        definition=definition,
        prototypical_features=features,
        distinguishing_criteria=criteria,
        exemplar_utterances=exemplars,
        subtle_utterances=subtle,
        adversarial_utterances=adversarial,
        word_prototypes=word_protos,
        aliases=list(yaml_data.get('aliases') or []),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VAAMR_HEADING = re.compile(r'^## Stage \d+ — .+$', re.MULTILINE)
_PURER_HEADING = re.compile(r'^## Move \d+ — [A-Z0-9]+ — .+$', re.MULTILINE)


def load_framework_md(path: Path) -> ThemeFramework:
    """
    Parse a framework markdown file (VAAMR.md or PURER.md style) into a
    ThemeFramework.

    The framework field in the YAML frontmatter selects the heading
    pattern:
        VAAMR  →  ## Stage N — ShortName
        PURER  →  ## Move N — KEY — ShortName

    HTML comments are stripped before parsing so inline annotations in
    exemplar / adversarial sections are invisible to the output.
    """
    text = path.read_text(encoding='utf-8')
    fm, body = _parse_frontmatter(text)

    # Strip HTML comments from body (annotations, PARSER CONTRACT block, etc.)
    body = _strip_html_comments(body)

    fw_name: str = fm.get('framework', '')
    heading_pattern = _VAAMR_HEADING if fw_name == 'VAAMR' else _PURER_HEADING

    matches = list(heading_pattern.finditer(body))
    themes: List[ThemeDefinition] = []
    for i, m in enumerate(matches):
        block_start = m.end() + 1  # skip the heading line itself
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block_text = body[block_start:block_end]
        themes.append(_parse_theme_block(block_text))

    categories = fm.get('categories') or None

    return ThemeFramework(
        name=fw_name,
        version=str(fm['version']),
        description=fm.get('framework_description', ''),
        themes=themes,
        categories=categories,
    )
