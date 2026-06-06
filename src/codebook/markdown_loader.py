"""
codebook/markdown_loader.py
----------------------------
Parse CODEBOOK.md into Codebook / CodeDefinition dataclasses.

Entry point:
    load_codebook_md(path: Path) -> Codebook

Heading format: ## Code N — code_id — Category Name
Per-block YAML: code_id, category, domain
H3 sections: Description, Inclusive Criteria, Exclusive Criteria,
             Exemplar Utterances

The canonical PARSER CONTRACT is specified in the HTML-comment block at the top
of frameworks/PHENOMENOLOGY_CODEBOOK.md (HTML comments are stripped before
data extraction).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .codebook_schema import CodeDefinition, Codebook


# ---------------------------------------------------------------------------
# Shared text helpers (mirrors theme_framework/markdown_loader.py)
# ---------------------------------------------------------------------------

def _strip_html_comments(text: str) -> str:
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    if not text.startswith('---'):
        raise ValueError("Markdown file does not start with YAML frontmatter '---'")
    end = text.index('\n---\n', 3)
    fm_text = text[4:end]
    body = text[end + 5:]
    return yaml.safe_load(fm_text), body


def _normalize_prose(text: str) -> str:
    text = re.sub(r'(?m)^\s*---\s*$', '', text)  # strip horizontal rule separators
    return re.sub(r'\s+', ' ', text).strip()


def _parse_blockquotes(text: str) -> List[str]:
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


def _extract_yaml_block(text: str) -> Tuple[dict, str]:
    m = re.search(r'^```yaml\n(.*?)^```', text, re.MULTILINE | re.DOTALL)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)), text[m.end():]


def _split_h3_sections(text: str) -> Dict[str, str]:
    pattern = re.compile(r'^### (.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]
    return sections


# ---------------------------------------------------------------------------
# Code block parser
# ---------------------------------------------------------------------------

_CODE_HEADING = re.compile(r'^## Code \d+ — [^ ]+ — .+$', re.MULTILINE)


def _parse_code_block(block_text: str) -> CodeDefinition:
    yaml_data, remaining = _extract_yaml_block(block_text)
    sections = _split_h3_sections(remaining)

    description = _normalize_prose(sections.get('Description', ''))
    inclusive = _normalize_prose(sections.get('Inclusive Criteria', ''))
    exclusive = _normalize_prose(sections.get('Exclusive Criteria', ''))
    exemplars = _parse_blockquotes(sections.get('Exemplar Utterances', ''))

    return CodeDefinition(
        code_id=str(yaml_data['code_id']),
        category=str(yaml_data['category']),
        domain=str(yaml_data['domain']),
        description=description,
        inclusive_criteria=inclusive,
        exclusive_criteria=exclusive,
        exemplar_utterances=exemplars,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_codebook_md(path: Path) -> Codebook:
    """
    Parse a CODEBOOK.md file into a Codebook.

    Heading format: ## Code N — code_id — Category Name
    HTML comments are stripped before parsing.
    """
    text = path.read_text(encoding='utf-8')
    fm, body = _parse_frontmatter(text)
    body = _strip_html_comments(body)

    matches = list(_CODE_HEADING.finditer(body))
    codes: List[CodeDefinition] = []
    for i, m in enumerate(matches):
        block_start = m.end() + 1
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block_text = body[block_start:block_end]
        codes.append(_parse_code_block(block_text))

    domains: Dict[str, List[str]] = {}
    for code in codes:
        domains.setdefault(code.domain, []).append(code.code_id)

    return Codebook(
        name=str(fm['codebook']),
        version=str(fm['version']),
        description=fm.get('codebook_description', ''),
        codes=codes,
        domains=domains,
    )
