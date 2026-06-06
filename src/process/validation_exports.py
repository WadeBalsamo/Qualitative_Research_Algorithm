"""
process/validation_exports.py
------------------------------
Cross-validation result aggregation helpers, extracted from orchestrator.py.
"""


def collect_top_associations(associations_by_theme: dict, n: int = 50) -> list:
    """Collect top theme-code associations across all themes, sorted by lift."""
    entries = []
    for theme_key, summary in associations_by_theme.items():
        for assoc in summary.get('top_associations', []):
            entries.append({
                'theme_key': theme_key,
                'code': assoc['code'],
                'count': assoc['count'],
                'rate': assoc['rate'],
                'base_rate': assoc['base_rate'],
                'lift': assoc['lift'],
            })
    entries.sort(key=lambda x: -x['lift'])
    return entries[:n]
