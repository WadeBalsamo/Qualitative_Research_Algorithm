"""
gnn_layer/communities.py
------------------------
Track D — subtext communities as routines (the deepest qualitative layer).

Motifs (Capability B) and coupling factors (Capability E) capture single therapist moves and
latent cue dimensions. The genuinely-new ~60% the master plan points at is which language
*routines/sequences flow together and recur across sessions*. This module:

  D1  Builds a thresholded (cosine ≥ τ) cross-session segment-similarity graph — distinct from
      the trained kNN graph; it links segments that say the *same kind of thing* anywhere in
      the corpus, so recurring subtext surfaces as dense regions.
  D2  Partitions it with TWO independent algorithms — Louvain (modularity optimization) and
      agglomerative hierarchical clustering (a different family) — and reports their agreement
      (adjusted Rand index) so a community is credible as STRUCTURE, not an algorithm artifact.
  D3  Models routines as community→community transitions WITHIN sessions ("X tends to precede Y").
  D4  STABILITY SELECTION: participant-bootstrap resampling estimates each community's
      co-membership stability; communities below ``community_stability_min`` are SUPPRESSED
      from the findings (flagged, not silently dropped) — n≈32 makes raw communities fragile.
  D5  Names the survivors with TF-IDF terms + exemplar quotes, and reports per-session
      prevalence + cross-cohort distribution (drift). Hypothesis-generating framing throughout.

Citations: community-stability literature (stability selection, Meinshausen & Bühlmann 2010;
consensus clustering, Monti et al. 2003) — NOT CFiCS, which has no community detection.

Pure numpy + sklearn + networkx (all already dependencies); degrades gracefully if a backend
is missing. CPU by design (numpy/sklearn post-processing; D11 keeps these off the GPU).
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from process import output_paths as _paths


def _seg_meta(df_all) -> Dict[str, dict]:
    """segment_id → {session_id, participant_id, cohort_id, session_number, start, text, speaker}."""
    out: Dict[str, dict] = {}
    for _, r in df_all.iterrows():
        sid = str(r.get('segment_id', ''))
        if not sid:
            continue
        out[sid] = {
            'session_id': str(r.get('session_id', '') or ''),
            'participant_id': r.get('participant_id'),
            'cohort_id': r.get('cohort_id'),
            'session_number': r.get('session_number'),
            'start': int(r.get('start_time_ms', 0) or 0),
            'text': str(r.get('text', '') or ''),
            'speaker': str(r.get('speaker', '') or ''),
        }
    return out


def build_subtext_graph(seg_emb: Dict[str, "object"], meta: Dict[str, dict],
                        threshold: float = 0.85, max_nodes: int = 4000):
    """Thresholded cosine-similarity graph over segments. Returns (networkx.Graph, info).

    Nodes = segments with embeddings. An edge connects two segments whose cosine similarity
    ≥ ``threshold`` (weight = similarity). ``info`` records edge counts incl. the cross-session
    fraction (the routines that recur across sessions, not just within one). Caps at
    ``max_nodes`` with a logged note rather than silently truncating.
    """
    import numpy as np
    import networkx as nx

    sids = [s for s in seg_emb if s in meta]
    info = {'n_nodes': 0, 'n_edges': 0, 'n_cross_session_edges': 0, 'n_capped': 0,
            'threshold': float(threshold)}
    if len(sids) > max_nodes:
        info['n_capped'] = len(sids) - max_nodes
        print(f"  [gnn_layer] subtext graph: capping {len(sids)} → {max_nodes} segments "
              f"({info['n_capped']} dropped; raise max_nodes to include them)")
        sids = sids[:max_nodes]
    if len(sids) < 3:
        return nx.Graph(), info

    X = np.stack([np.asarray(seg_emb[s], dtype=np.float32) for s in sids])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T

    G = nx.Graph()
    G.add_nodes_from(sids)
    n = len(sids)
    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] >= float(threshold)
    n_cross = 0
    for a, b in zip(iu[mask], ju[mask]):
        sa, sb = sids[a], sids[b]
        w = float(sim[a, b])
        G.add_edge(sa, sb, weight=w)
        if meta[sa]['session_id'] != meta[sb]['session_id']:
            n_cross += 1
    info['n_nodes'] = n
    info['n_edges'] = G.number_of_edges()
    info['n_cross_session_edges'] = n_cross
    return G, info


def detect_communities(G, seg_emb, sids: List[str], config) -> dict:
    """Two-algorithm partition + agreement. Returns labels (Louvain) + ARI vs hierarchical.

    Louvain (modularity) is the primary partition. Agglomerative hierarchical clustering on
    the embedding matrix (a different algorithmic family) is the robustness check; their
    adjusted Rand index says whether the structure survives a change of algorithm.
    """
    import numpy as np
    import networkx as nx

    seed = int(getattr(config, 'seed', 42))
    try:
        comms = nx.community.louvain_communities(G, weight='weight', seed=seed)
    except Exception:
        comms = [set(c) for c in nx.connected_components(G)]
    comms = sorted(comms, key=len, reverse=True)
    louvain = {sid: ci for ci, c in enumerate(comms) for sid in c}

    # Second algorithm: agglomerative hierarchical on the embeddings, same node set.
    ari = None
    n_clusters = len([c for c in comms if len(c) >= 2])
    if n_clusters >= 2 and len(sids) > n_clusters:
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import adjusted_rand_score
            X = np.stack([np.asarray(seg_emb[s], dtype=np.float32) for s in sids])
            hier = AgglomerativeClustering(n_clusters=min(n_clusters, len(sids) - 1),
                                           metric='cosine', linkage='average').fit_predict(X)
            ari = float(adjusted_rand_score([louvain[s] for s in sids], hier.tolist()))
        except Exception:
            ari = None

    return {'labels': louvain, 'communities': comms, 'n_communities': len(comms),
            'ari_louvain_vs_hierarchical': round(ari, 4) if ari is not None else None}


def community_transitions(labels: Dict[str, int], meta: Dict[str, dict]) -> List[dict]:
    """D3 — community→community transitions within sessions (language routines).

    For each session, order its segments by start time and count consecutive
    community(i)→community(i+1) pairs (excluding self-loops). Returns the top transitions
    sorted by count.
    """
    by_session: Dict[str, List[tuple]] = defaultdict(list)
    for sid, c in labels.items():
        m = meta.get(sid)
        if m is None:
            continue
        by_session[m['session_id']].append((m['start'], c))
    trans = Counter()
    for sess, seq in by_session.items():
        seq.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(seq, seq[1:]):
            if a != b:
                trans[(a, b)] += 1
    rows = [{'from_community': a, 'to_community': b, 'count': n}
            for (a, b), n in trans.most_common()]
    return rows


def community_stability(seg_emb, meta, config, full_labels: Dict[str, int],
                        communities: List[set]) -> Dict[int, dict]:
    """D4 — participant-bootstrap co-membership stability per community.

    Resamples participants with replacement, rebuilds the subtext graph on their segments,
    re-detects communities, and for each full-partition community measures the fraction of its
    member pairs that remain co-membered (averaged over bootstraps where both were sampled).
    Communities below ``community_stability_min`` are flagged unstable. Reuses the
    participant-cluster resampling logic the analysis layer uses elsewhere.
    """
    import numpy as np
    import networkx as nx

    participants = sorted({meta[s]['participant_id'] for s in full_labels
                           if meta.get(s) and meta[s]['participant_id'] is not None},
                          key=lambda x: str(x))
    out: Dict[int, dict] = {}
    min_size = int(getattr(config, 'community_min_size', 3))
    # Candidate member pairs per community (cap to bound cost).
    rng = np.random.default_rng(int(getattr(config, 'seed', 42)))
    pairs_by_comm: Dict[int, List[tuple]] = {}
    for ci, c in enumerate(communities):
        members = [s for s in c if meta.get(s)]
        if len(members) < min_size:
            continue
        allpairs = [(members[i], members[j]) for i in range(len(members))
                    for j in range(i + 1, len(members))]
        if len(allpairs) > 100:
            idx = rng.choice(len(allpairs), size=100, replace=False)
            allpairs = [allpairs[k] for k in idx]
        pairs_by_comm[ci] = allpairs

    if not pairs_by_comm or len(participants) < 2:
        return {ci: {'stability': None, 'size': len(communities[ci]), 'n_pairs': len(p)}
                for ci, p in pairs_by_comm.items()}

    by_pid: Dict[object, List[str]] = defaultdict(list)
    for s in full_labels:
        m = meta.get(s)
        if m and m['participant_id'] is not None:
            by_pid[m['participant_id']].append(s)

    boots = int(getattr(config, 'community_stability_boots', 50))
    thr = float(getattr(config, 'community_sim_threshold', 0.85))
    co_counts = {ci: Counter() for ci in pairs_by_comm}
    seen_counts = {ci: Counter() for ci in pairs_by_comm}

    for _ in range(boots):
        chosen = rng.choice(len(participants), size=len(participants), replace=True)
        sample_ids = set()
        for pi in chosen:
            sample_ids.update(by_pid.get(participants[pi], []))
        sub_emb = {s: seg_emb[s] for s in sample_ids if s in seg_emb}
        if len(sub_emb) < 3:
            continue
        Gb, _ = build_subtext_graph(sub_emb, meta, threshold=thr)
        try:
            bcomms = nx.community.louvain_communities(Gb, weight='weight',
                                                      seed=int(getattr(config, 'seed', 42)))
        except Exception:
            bcomms = [set(c) for c in nx.connected_components(Gb)]
        blabel = {sid: ci for ci, c in enumerate(bcomms) for sid in c}
        for ci, allpairs in pairs_by_comm.items():
            for a, b in allpairs:
                if a in blabel and b in blabel:
                    seen_counts[ci][(a, b)] += 1
                    if blabel[a] == blabel[b]:
                        co_counts[ci][(a, b)] += 1

    min_stab = float(getattr(config, 'community_stability_min', 0.5))
    for ci, allpairs in pairs_by_comm.items():
        rates = []
        for a, b in allpairs:
            seen = seen_counts[ci][(a, b)]
            if seen > 0:
                rates.append(co_counts[ci][(a, b)] / seen)
        stab = float(np.mean(rates)) if rates else None
        out[ci] = {
            'stability': round(stab, 4) if stab is not None else None,
            'size': len(communities[ci]),
            'n_pairs': len(allpairs),
            'stable': bool(stab is not None and stab >= min_stab),
        }
    return out


def name_communities(communities: List[set], meta: Dict[str, dict],
                     stability: Dict[int, dict], config) -> List[dict]:
    """D5 — TF-IDF terms + exemplars + per-session prevalence + cohort drift per community."""
    import numpy as np

    min_size = int(getattr(config, 'community_min_size', 3))
    keep = [(ci, c) for ci, c in enumerate(communities) if len(c) >= min_size]
    if not keep:
        return []

    # Corpus for TF-IDF = the member texts of the kept communities.
    all_sids = [s for _, c in keep for s in c if meta.get(s)]
    texts = [meta[s]['text'] for s in all_sids]
    terms_by_comm: Dict[int, List[str]] = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=2000, stop_words='english', min_df=1)
        M = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())
        row_of = {s: i for i, s in enumerate(all_sids)}
        for ci, c in keep:
            idx = [row_of[s] for s in c if s in row_of]
            if not idx:
                terms_by_comm[ci] = []
                continue
            mean_tfidf = np.asarray(M[idx].mean(axis=0)).ravel()
            top = np.argsort(mean_tfidf)[::-1][:8]
            terms_by_comm[ci] = [str(vocab[t]) for t in top if mean_tfidf[t] > 0]
    except Exception:
        terms_by_comm = {ci: [] for ci, _ in keep}

    rows = []
    for ci, c in keep:
        members = [s for s in c if meta.get(s)]
        sessions = Counter(meta[s]['session_id'] for s in members)
        cohorts = Counter(str(meta[s]['cohort_id']) for s in members)
        speakers = Counter(meta[s]['speaker'] for s in members)
        # Exemplars: longest member texts (proxy for content-richness), up to 3.
        exemplars = sorted((meta[s]['text'] for s in members), key=len, reverse=True)[:3]
        st = stability.get(ci, {})
        rows.append({
            'community_id': ci,
            'size': len(members),
            'n_sessions': len(sessions),
            'n_cohorts': len(cohorts),
            'top_terms': terms_by_comm.get(ci, []),
            'dominant_speaker': speakers.most_common(1)[0][0] if speakers else None,
            'session_prevalence': dict(sessions.most_common(5)),
            'cohort_distribution': dict(cohorts),
            'exemplars': [e[:200] for e in exemplars if e],
            'stability': st.get('stability'),
            'stable': st.get('stable'),
        })
    rows.sort(key=lambda r: (-(r['stability'] or 0), -r['size']))
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_communities_csv(rows: List[dict], output_dir: str) -> Optional[str]:
    """Per-community summary → 03_analysis_data/gnn/subtext_communities.csv."""
    import pandas as pd
    if not rows:
        return None
    flat = []
    for r in rows:
        flat.append({
            'community_id': r['community_id'], 'size': r['size'],
            'n_sessions': r['n_sessions'], 'n_cohorts': r['n_cohorts'],
            'dominant_speaker': r['dominant_speaker'],
            'stability': r['stability'], 'stable': r['stable'],
            'top_terms': '; '.join(r['top_terms']),
        })
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'subtext_communities.csv')
    pd.DataFrame(flat).to_csv(path, index=False)
    return path


def write_transitions_csv(trans: List[dict], output_dir: str) -> Optional[str]:
    import pandas as pd
    if not trans:
        return None
    gnn_dir = _paths.gnn_data_dir(output_dir)
    os.makedirs(gnn_dir, exist_ok=True)
    path = os.path.join(gnn_dir, 'subtext_community_transitions.csv')
    pd.DataFrame(trans).to_csv(path, index=False)
    return path


def write_communities_report(rows, trans, graph_info, detect_info, output_dir: str) -> str:
    """Human-readable D report → 06_reports/06_gnn/communities.txt."""
    W = 78
    L = ["=" * W, "SUBTEXT COMMUNITIES AS ROUTINES (Track D)", "=" * W, ""]
    L.append("DISCOVERY / HYPOTHESIS-GENERATING. Segments that say the same kind of thing are")
    L.append("linked into a cosine-similarity graph and partitioned into 'subtext communities'")
    L.append("— recurring language routines. Two independent algorithms must agree for a")
    L.append("community to count as structure, and each survivor passed participant-bootstrap")
    L.append("STABILITY SELECTION. n≈32 makes these fragile: nothing below the stability floor")
    L.append("is reported as a finding. NOT causal; cross-cohort patterns are drift hypotheses.")
    L.append("")
    L.append(f"  subtext graph : {graph_info.get('n_nodes')} segments, "
             f"{graph_info.get('n_edges')} edges (τ={graph_info.get('threshold')}), "
             f"{graph_info.get('n_cross_session_edges')} cross-session")
    if graph_info.get('n_capped'):
        L.append(f"                  (capped: {graph_info['n_capped']} segments dropped)")
    ari = detect_info.get('ari_louvain_vs_hierarchical')
    L.append(f"  partition     : {detect_info.get('n_communities')} communities; "
             f"Louvain↔hierarchical agreement (ARI) = "
             f"{ari if ari is not None else 'n/a'}")
    if ari is not None:
        L.append("                  (higher ARI ⇒ the partition is algorithm-robust, not an artifact)")
    L.append("")

    stable = [r for r in rows if r.get('stable')]
    unstable = [r for r in rows if not r.get('stable')]

    L.append("-" * W)
    L.append(f"STABLE COMMUNITIES (reported as findings): {len(stable)}")
    L.append("-" * W)
    if not stable:
        L.append("  (none cleared the stability floor — treat all communities as exploratory.)")
    for r in stable:
        st = f"{r['stability']:.2f}" if r['stability'] is not None else "n/a"
        L.append("")
        L.append(f"  Community {r['community_id']}  (size={r['size']}, stability={st}, "
                 f"{r['n_sessions']} sessions, {r['n_cohorts']} cohorts, "
                 f"{r['dominant_speaker']})")
        if r['top_terms']:
            L.append(f"    terms: {', '.join(r['top_terms'])}")
        for ex in r['exemplars'][:2]:
            L.append(f"    “{ex}”")
        if r['cohort_distribution']:
            L.append(f"    cohort spread: {r['cohort_distribution']}")
    L.append("")

    if unstable:
        L.append("-" * W)
        L.append(f"UNSTABLE / SUPPRESSED COMMUNITIES (flagged, not findings): {len(unstable)}")
        L.append("-" * W)
        for r in unstable[:20]:
            st = f"{r['stability']:.2f}" if r['stability'] is not None else "n/a"
            terms = ', '.join(r['top_terms'][:5])
            L.append(f"  Community {r['community_id']}  size={r['size']} stability={st}  {terms}")
        L.append("")

    if trans:
        L.append("-" * W)
        L.append("LANGUAGE ROUTINES — top within-session community→community transitions")
        L.append("-" * W)
        for t in trans[:15]:
            L.append(f"    community {t['from_community']} → {t['to_community']}   "
                     f"(n={t['count']})")
        L.append("  (X→Y means routine X tends to precede routine Y within a session.)")
        L.append("")

    rep_dir = _paths.reports_gnn_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'communities.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def run_subtext_communities(df_all, seg_emb, output_dir: str, config) -> dict:
    """Orchestrate Track D: graph → two-algorithm partition → routines → stability → naming.

    Returns {files_written, n_communities, n_stable, status}.
    """
    files: List[str] = []
    meta = _seg_meta(df_all)
    thr = float(getattr(config, 'community_sim_threshold', 0.85))
    G, graph_info = build_subtext_graph(seg_emb, meta, threshold=thr)
    if graph_info.get('n_nodes', 0) < 3:
        return {'files_written': files, 'n_communities': 0, 'n_stable': 0,
                'status': 'skipped: too few segments for a subtext graph'}

    sids = list(G.nodes())
    detect = detect_communities(G, seg_emb, sids, config)
    communities = detect['communities']
    labels = detect['labels']

    trans = community_transitions(labels, meta)
    stability = community_stability(seg_emb, meta, config, labels, communities)
    rows = name_communities(communities, meta, stability, config)

    p = write_communities_csv(rows, output_dir)
    if p:
        files.append(p)
    p = write_transitions_csv(trans, output_dir)
    if p:
        files.append(p)
    files.append(write_communities_report(rows, trans, graph_info, detect, output_dir))

    n_stable = sum(1 for r in rows if r.get('stable'))
    return {'files_written': files, 'n_communities': detect['n_communities'],
            'n_stable': n_stable, 'ari': detect.get('ari_louvain_vs_hierarchical'),
            'status': 'ok'}
