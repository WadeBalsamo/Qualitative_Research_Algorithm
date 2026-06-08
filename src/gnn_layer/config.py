"""
gnn_layer/config.py
-------------------
Configuration dataclass for the GNN representation-and-discovery layer.

Placed here (mirroring ``codebook/config.py``) and imported into
``process/config.py`` as ``PipelineConfig.gnn_layer``. The layer is ON by default
(``enabled=True``); it runs at analyze-time and is fully guarded, so it degrades
to a logged warning if training cannot complete. Set ``enabled=False`` to skip it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GnnLayerConfig:
    """Settings for the Qwen3-embedding GNN analysis layer.

    The layer is ON by default (``enabled=True``) and reuses the same embedding
    model QRA already uses for segmentation and VCE coding, so no second model is
    loaded. Set ``enabled=False`` to skip it on resource-constrained runs.
    """
    # Master switch. When False, analysis/runner.py never invokes the layer.
    # ON by default: the layer trains at analyze-time and degrades gracefully
    # (the call site in analysis/runner.py is try/except-wrapped, and torch is a
    # hard dependency). Set False to skip it on resource-constrained runs.
    enabled: bool = True

    # ---- Embedding substrate (reuses codebook EmbeddingCodebookClassifier) ----
    embedding_model: str = 'Qwen/Qwen3-Embedding-8B'
    use_query_prefix: bool = True
    embedding_batch_size: int = 8
    cache_embeddings: bool = True          # cache to 02_meta/gnn/segment_embeddings.npz

    # ---- Embedding backend selector (local SentenceTransformer vs remote /v1/embeddings) ----
    # 'local'  (default) → in-process EmbeddingCodebookClassifier path. Byte-identical to
    #                      prior behaviour; embedding_model is the HuggingFace repo id.
    # 'openai'           → POST segment/anchor texts to an OpenAI-compatible /v1/embeddings
    #                      endpoint (e.g. LM Studio serving Qwen3-Embedding-8B) via
    #                      gnn_layer/embeddings_remote.py — no second 8 GB model is loaded.
    #                      embedding_model is then the SERVED model id
    #                      (e.g. 'text-embedding-qwen3-embedding-8b') sent in the POST body.
    embedding_backend: str = 'local'
    # Root URL of the OpenAI-compatible server, ending at '/v1' (the client appends
    # '/embeddings'). Required when embedding_backend='openai'. e.g. 'http://10.0.0.58:1234/v1'.
    embedding_base_url: Optional[str] = None
    # Bearer token for the embeddings endpoint. Leave None for LM Studio (needs no key);
    # only sent as an Authorization header when set.
    embedding_api_key: Optional[str] = None

    # ---- Graph construction ----
    knn_k: int = 8                         # kNN similarity edges per segment node
    # Typed therapist->participant "precipitates" edges (Track A1): for each cue block,
    # connect each therapist cue segment to the FOLLOWING participant segment so the
    # participant representation can use the preceding cue. Turning this ON also activates
    # LEARNABLE per-edge-type weights in SAGEConv (a per-family gate, neutral at init so the
    # default OFF path is byte-identical to fixed-weight aggregation). Default OFF: the
    # family must EARN its place by raising out-of-sample κ on BOTH the human and LLM axes
    # (measured with the ablation.anchor_contribution with/without-family pattern).
    precipitates_edges: bool = False
    # Run the with/without-precipitates ablation checkpoint (Track A1): builds the graph
    # twice on identical folds/seed and compares out-of-sample κ on both the human and LLM
    # axes. Unlike anchors, precipitates edges are empirical (temporal/structural), NOT
    # seeded from construct definitions, so the LLM axis is a legitimate (un-inflated)
    # signal here. Default OFF (the checkpoint doubles gate cost).
    run_precipitates_ablation: bool = False
    # VCE is OFF by default: the research mission is PURER->VAAMR. VCE re-enters the
    # graph only when a study wants the Capability-D ablation to answer "does VCE carry
    # independent signal worth further research?" (set include_vce_nodes=True + run_gnn_ablation).
    include_vce_nodes: bool = False        # anchor nodes for VCE codes (ablatable via D)
    include_purer_nodes: bool = True       # PURER-move construct anchors (when anchors are used)
    include_vaamr_nodes: bool = True       # VAAMR-stage construct anchors (when anchors are used)
    cross_framework_min_lift: float = 1.5  # threshold for vaamr<->vce anchor<->anchor lift edges

    # ---- Construct anchors (Path B / G2) ----
    # An anchor is a construct DEFINITION embedded in the segment space; it attaches to
    # segments by similarity ONLY (no labels). Default OFF: the homogeneous graph is the
    # most defensible substrate for triangulation, so anchors must EARN their place via the
    # ablation below — which scores Δκ on the GNN<->HUMAN axis, never GNN<->LLM (anchors
    # inflate the latter by construction; see gnn_layer/anchors.py).
    use_anchor_nodes: bool = False         # include anchor nodes in the MAIN trained graph
    run_anchor_ablation: bool = False      # run the with/without-anchors human-axis Δκ test
    anchor_knn_m: int = 8                  # segments each anchor connects to (by similarity)
    anchor_min_human: int = 10             # min human-coded rows for the ablation to be decisive

    # ---- Model ----
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.5

    # ---- Objectives / training ----
    # Subset of: soft_vaamr, progression, vce_multilabel, purer,
    #            contrastive, link_prediction
    objectives: List[str] = field(default_factory=lambda: [
        'soft_vaamr', 'progression', 'contrastive', 'link_prediction',
    ])
    # 'weak'  -> train soft heads on LLM multi-run ballots (default; directional)
    # 'human' -> train only on human-validated subset (for the independence claim)
    # 'self_supervised' -> link-prediction only, no LLM labels (independence control)
    label_mode: str = 'weak'

    # ---- VAAMR class-imbalance handling & "No code" class ----------------------------
    # These four flags target design_decisions.md §4 root causes #2 ("No class
    # rebalancing": Avoidance n=20 / Metacognition n=29 get 0% held-out recall under the
    # unweighted batchmean KL) and #4 ("No 'No code' class": 134/339 participants are
    # No-code and were trained to uniform noise). They power arms A4 / A4n (§6). EVERY
    # flag defaults to the no-op, so with none set the soft-VAAMR loss and head are
    # byte-identical to the original unweighted 5-class path.
    #
    # 6th "No code" class. 5 (default) → the soft-VAAMR head is size 5 and behaviour is
    # byte-identical. 6 → head size 6 with class 5 = "No code"; build_soft_targets(...,
    # n_stages=6) then gives participant rows with a null final_label a one-hot on class 5
    # (instead of the uniform-noise fallback) while labeled rows keep their 5-stage
    # mixture in dims 0..4 (dim 5 = 0). (The runner/validation wiring of this value into
    # build_soft_targets(n_stages=...) is a separate follow-up; the head/target/loss
    # building blocks honour it today.)
    vaamr_n_classes: int = 5
    # Class balancing. When True, each labeled node's soft-VAAMR KL is weighted by the
    # inverse frequency of its dominant (argmax) class within the training batch,
    # normalized to mean 1 — so rare stages are not drowned out. False → unweighted.
    vaamr_class_balance: bool = False
    # Focal modulation. When >0, multiply each labeled node's soft-VAAMR KL by
    # (1 - p_true)^gamma, where p_true is the model's softmax mass on that row's
    # argmax-target class — down-weighting already-confident rows so hard/rare rows
    # dominate the gradient. 0 → off (identical). Composes with vaamr_class_balance.
    vaamr_focal_gamma: float = 0.0
    # Auxiliary hard-label cross-entropy. When >0, ADD a class-weighted hard CE term
    # (this value is its loss weight) on the labeled VAAMR nodes — target = argmax of the
    # soft mixture, class weight = inverse class frequency — giving a hard-classification
    # signal alongside the soft KL. 0 → off (no extra term, total unchanged).
    vaamr_hard_ce_weight: float = 0.0
    # Logit adjustment / topology-aware margin (Menon et al. 2020). When True, add
    # log(class_prior) (batch class frequencies) to the soft-VAAMR logits AT TRAIN TIME
    # so rare classes get an effective margin; inference uses the raw logits. False → off.
    vaamr_tam: bool = False

    # ---- Independence pass (G1: the construct-validity axis that survives review) ----
    # The main run trains on LLM ballots, so its GNN-vs-LLM agreement is DISTILLATION
    # FIDELITY, not independence. When True, a SECOND model is trained with LLM labels
    # WITHHELD and its triangulation written to a separate report — so GNN-vs-LLM κ there
    # is genuine corroboration (the model never saw LLM labels) and GNN-vs-human is the
    # load-bearing validity evidence. This is the substrate the manuscript should cite for
    # any "independent measurement" claim.
    report_independence_pass: bool = True
    # 'human'          -> heads trained only on the human blind-coded subset (LLM-free supervision)
    # 'self_supervised'-> link-prediction/contrastive only (geometry-only NULL control; low κ expected)
    # 'auto'           -> 'human' when a usable human subset exists, else 'self_supervised'
    independence_label_mode: str = 'auto'
    independence_min_human: int = 10       # min human-coded rows for 'auto' to pick 'human'

    contrastive_temp: float = 0.1
    epochs: int = 300
    lr: float = 1e-3
    patience: int = 40
    seed: int = 42
    device: Optional[str] = None           # None -> auto (cuda if available else cpu)

    # ---- Which speakers get GNN positioning ----
    # RESERVED — not yet enforced by the builder: both participant and therapist
    # segments are always positioned (graph_builder builds both node types
    # unconditionally). These flags are serialized for forward-compatibility.
    run_on_participants: bool = True       # VAAMR mixture / progression coordinate
    run_on_therapists: bool = True         # cue-block embeddings / PURER positioning

    # ---- Capability B (motif discovery) ----
    n_motif_clusters: int = 12
    min_motif_influence: float = 1.2       # flag motifs whose forward-transition lift exceeds this
    motif_min_block_count: int = 3         # ignore motifs with fewer than this many cue blocks

    # ---- Capability E (coupling / latent factors) ----
    n_latent_factors: int = 5
    interpret_against_cf_ic: bool = True   # label latent factors against the inline CF/IC lexicon

    # ---- Capability D (ablation) ----
    run_gnn_ablation: bool = False         # retrain per-head to rank construct signal (doubles training cost)
    # Direct test of the VCE-on-VAAMR hypothesis (methodology §3.3 / §5.2): retrain the
    # held-out reliability cross-validation WITH and WITHOUT the granular VCE multi-label
    # head and compare out-of-sample VAAMR κ (overall + per stage). A positive Δκ is
    # evidence the fine-grained phenomenology codebook sharpens the coarse five-stage
    # classification; Δκ ≈ 0 / negative justifies keeping VCE out of the classifier of record.
    test_vce_layer: bool = False           # write report_gnn_vce_contribution.txt (doubles the gate cost)

    # ---- Consensus-distillation classifier (gnn_layer/classifier/; SEPARATE CONCERN, default OFF) ----
    # Master switch for the GraphSAGE consensus-distillation CLASSIFIER + reliability gate. DEFAULT
    # OFF: the Cohorts 1–2 pilot REFUTED its scaler role (H5) — under leakage-free participant-grouped
    # CV it reproduces the LLM consensus at κ≈0.05–0.14 (below the human↔human band), a linear probe
    # on the same Qwen features ties/beats it, and Correct-&-Smooth (pure graph smoothing) is the
    # worst arm — VAAMR is not homophilous in embedding space (graph_experiments.md; methodology
    # §8.5). The LLM consensus (already human-level, κ=0.537 vs human) and the Qwen probe remain the
    # labels of record. The discovery + mechanism work-streams (discriminant_validity / transition_
    # model / subtext_communities / confound_localization above) run independently and are default
    # ON. Set True to train the classifier + gate and re-adjudicate at Cohorts 3–4 scale.
    gnn_classifier_enabled: bool = False
    produce_consensus_labels: bool = True  # write per-segment graph labels to the gnn_labels overlay
                                           # (only takes effect when gnn_classifier_enabled=True)
    # DEPRECATED + NO LONGER CONSULTED (methodology §8.6). The GNN was repositioned to a
    # non-authoritative FILL tier: it can label only segments the LLM left unlabeled (gated by
    # its reliability gate) and can NEVER override an LLM/human label. The probe is the
    # recommended cheap scaler. Retained for back-compat with existing qra_config.json files;
    # assemble_master_dataset ignores it.
    gnn_authoritative: bool = False

    # ---- Abstention / deferral (A2: don't poison the training set) ----
    # When the graph is the label of record, a per-stage max-prob confidence floor lets it
    # DEFER ("ABSTAIN") on segments it is not confident about — master_dataset then keeps the
    # LLM label rather than promote a confident-wrong graph guess. Abstention is recorded
    # (gnn_vaamr_abstain / gnn_purer_abstain in the overlay) so every deferral is auditable.
    abstain_threshold: Optional[float] = None        # global max-prob floor; None disables abstention
    # Rare VAAMR stages (Metacognition=3, Reappraisal=4) get a higher floor by default — they
    # are the over-smoothing risk, so we demand more confidence before promoting them.
    abstain_rare_stage_threshold: Optional[float] = None
    # Explicit per-VAAMR-stage floors {stage:int -> floor:float}; overrides the two above when set
    # (also where calibration writes its derived floors).
    abstain_per_stage: Optional[Dict[int, float]] = None
    # When True, derive per-stage floors from the held-out CV predictions: pick, per stage, the
    # smallest floor whose KEPT (non-abstained) held-out precision meets abstain_target_precision.
    abstain_calibrate: bool = False
    abstain_target_precision: float = 0.80

    # ---- Confidence calibration for domain shift (A3) ----
    # Softmax confidence is fit in-distribution; on genuinely new transcripts it can be
    # miscalibrated (over-confident). When True, a single temperature T is fit on the held-out
    # CV logits (minimizing NLL vs the LLM consensus) and applied to the soft-VAAMR logits at
    # inference, so the abstention floors (A2) operate on calibrated probabilities.
    calibrate: bool = False
    calibration_temperature: Optional[float] = None  # fitted T (>=1 typically); where calibration writes
    # Out-of-distribution gate: a new segment's mean cosine distance to its k nearest TRAINING
    # segments. Above ood_threshold the graph is extrapolating, so scale-mode forces ABSTAIN
    # (defer to the LLM) regardless of softmax confidence. None disables the OOD gate.
    ood_threshold: Optional[float] = None
    ood_knn_k: int = 8

    # ---- Semi-supervised label propagation (A4, optional + measured) ----
    # Post-training, diffuse the model's per-node soft predictions over the temporal/kNN edges
    # (F <- alpha * neighbour-mean(F) + (1-alpha) * P_model). When True, an ablation measures
    # held-out κ with vs without the diffusion; it is RETAINED only if Δκ >= +0.02 (else the
    # raw model wins). Default OFF.
    label_propagation: bool = False
    propagation_alpha: float = 0.5    # diffusion mixing weight (0 = model only, 1 = neighbours only)
    propagation_iters: int = 20       # diffusion iterations

    # ---- Scale-mode simulation gate (A5) ----
    # The k-fold reliability gate trains on the SAME topology it scores, so it does not
    # simulate the actual scale-mode condition: attaching genuinely-new SESSIONS inductively.
    # When True, whole sessions are held out, the model trains on the rest, the held-out
    # sessions are attached via attach_new_segments, and held-out κ is compared to the
    # in-sample CV κ. A large gap (> scale_sim_max_gap) flags domain-shift risk.
    run_scale_sim: bool = False
    scale_sim_holdout_sessions: int = 1   # whole sessions held out per simulation fold
    scale_sim_max_gap: float = 0.10       # CV κ − inductive κ above this flags domain-shift risk

    # ---- Mechanism (Track B) — superseded by the dyadic transition model ----
    # The old model-counterfactual-on-classifier flags (counterfactual / counterfactual_max_blocks
    # / influence_bootstrap_n / counterfactual_subgroups) were removed when gnn_layer/influence.py
    # was retired: that path read a per-segment classifier's incidental sensitivity (mis-specified
    # for mechanism, methodology §4.7). The mechanism instrument is now the dyadic transition model
    # (transition_* above) + confound localization (confound_localization above).

    # ---- H6 discriminant validity (construct-validation; gate-independent) ----
    # Packages the H5-refutation as the positive H6 finding: on identical participant-grouped
    # folds and the same Qwen embeddings, a supervised probe recovers VAAMR above chance while a
    # content-similarity model (Correct-&-Smooth/kNN) ≈ chance — and characterises the geometry
    # (stage-variance-by-content-PCs, stage⟂topic principal angles, community×stage independence).
    # Runs at analyze-time via gnn_layer/discriminant.py; reuses the reliability harness. Loads the
    # cached Qwen embeddings itself and degrades gracefully if they / the harness are unavailable.
    # Hypothesis-generating; NO causal claims. Default ON (cheap; cache is warm on the pilot).
    discriminant_validity: bool = True

    # ---- Mechanism: dyadic FROM→CUE→TO transition model (the rebuild; gnn_layer/transition.py) ----
    # Replaces the mis-specified mechanism-on-classifier path. A small learned response function
    # TO_mixture ≈ f(FROM_mixture, FROM_stage, pooled raw-Qwen cue) over cue-block triples: directed
    # FROM→CUE→TO only (NO kNN — H6 shows similarity neighbourhoods are stage-mixed), FROM-stage
    # conditioning baked in (the partial control for the elicitation confound — moves compared
    # WITHIN a starting state). It is validated by an earns-its-place participant-grouped CV (does
    # the cue improve held-out TO prediction over FROM alone?) and its counterfactual reads a
    # learned response, not a classifier's incidental sensitivity. Triangulated against the observed
    # Δprogression (mechanism.py LEAD). Sensitivity analysis, NOT causation. Default ON (cheap).
    transition_model: bool = True
    transition_cue_dim: int = 32        # PCA dim for the pooled cue embedding (small n → small model)
    transition_hidden: int = 16         # 1-hidden-layer MLP width; small by design at n≈160 blocks
    transition_dropout: float = 0.3
    transition_epochs: int = 300
    transition_lr: float = 1e-2
    transition_weight_decay: float = 1e-3
    transition_folds: int = 5           # participant-grouped CV folds for the earns-its-place check
    transition_bootstrap_n: int = 1000  # participant-clustered bootstrap for counterfactual CIs

    # ---- WS3: confound localization (gnn_layer/confound.py) ----
    # Signed divergence between the transition counterfactual and the observed Δprogression per
    # (from_stage × move) — where the elicitation/responsiveness confound (§9.4) most distorts the
    # observed mechanism table. A caveat instrument, NOT a claim. Reuses the transition result.
    confound_localization: bool = True

    # ---- Track D: subtext communities as routines (D1-D5) ----
    # The "deepest qualitative" layer: which language ROUTINES/SEQUENCES flow together and
    # recur across sessions. A thresholded cross-session segment-similarity graph is partitioned
    # by TWO independent algorithms (Louvain + hierarchical) to separate real structure from
    # algorithm artifact (agreement = adjusted Rand index); community→community within-session
    # transitions model routines; participant-bootstrap STABILITY SELECTION suppresses/flags
    # communities too fragile at n≈32; TF-IDF terms + exemplars + per-session prevalence name
    # them. Discovery / hypothesis-generating only. Independent of the gate. Default OFF.
    subtext_communities: bool = True
    # cosine-similarity edge threshold for the subtext graph. Calibrated for instruction-tuned
    # embeddings (Qwen3), whose pairwise cosines run high: a threshold probe on the pilot corpus
    # showed τ=0.85 yields only noise (Louvain↔hierarchical ARI≈0.003, ~all singletons) while
    # τ≈0.6 gives algorithm-robust structure (ARI≈0.29, ~24 multi-member communities). Stability
    # selection (D4) still suppresses fragile communities below this. Raise toward 0.8+ only for
    # un-instruction-tuned encoders (e.g. MiniLM) where cosines are lower.
    community_sim_threshold: float = 0.6
    community_min_size: int = 3             # ignore communities smaller than this
    community_stability_min: float = 0.5    # suppress communities whose bootstrap co-membership < this
    community_stability_boots: int = 50     # participant-bootstrap resamples for stability selection

    # ---- Track C: MindfulBERT training-set builder (C1-C5) ----
    # The end-goal artifact: a versioned (cue language → observed Δprogression) dataset for
    # fine-tuning MindfulBERT. The PRIMARY labels are the OBSERVED Δprogression of cue blocks
    # (the label of record); GNN model-counterfactual "would-progress" targets are a SEPARATE,
    # provenance-tagged augmentation channel produced only from a gate-passing model and
    # RETAINED only if a held-out ablation (C4) shows it helps (gain > augmentation_min_gain).
    build_mindfulbert_dataset: bool = True   # master switch for Track C (on by default — feeds the fine-tuning workshop)
    augmentation_enabled: bool = False       # add the GNN-counterfactual channel (C3; gate-passing only)
    augmentation_min_gain: float = 0.0       # retain augmentation only if held-out gain exceeds this (C4)

    # ---- Reliability gate (out-of-sample, the over-smoothing safeguard) ----
    # The gate is measured by cross-validation: each fold's soft-label targets are masked
    # during training, then predicted, so the reported kappa is on segments the graph did
    # NOT learn from. Per-stage / per-move breakdown is what catches rare-class (Reappraisal)
    # erosion that an aggregate kappa would hide.
    validation_folds: int = 5              # k for k-fold held-out evaluation
    validation_holdout: float = 0.2        # single-holdout fraction when validation_folds <= 1
    irr_target: float = 0.70               # kappa vs LLM consensus at which LLM-free scaling is recommended
    # PARTICIPANT-GROUPED cross-validation (the honest default). Random k-fold over a transcript
    # graph LEAKS: a held-out segment's temporal/kNN neighbours include its own and same-participant
    # segments (whose labels are visible), so the model effectively sees the answer — this inflated
    # the historical gate κ (0.247 random → ≈0.05 grouped on this corpus; see graph_experiments.md
    # §4.5). When True (default) AND the caller passes a per-node `groups` map (the runner builds one
    # from participant_id / session_id), crossval_predictions holds out WHOLE participants
    # (StratifiedGroupKFold), removing the leak. Set False to restore the legacy random k-fold.
    participant_grouped_cv: bool = True
