# Vote-Policy Comparison — Results

Experiment: re-vote stored ballots under four policies and score against
human-consensus IRR codes. No LLM calls; pure re-voting of stored ballots.

Policies:
  legacy          — pre-M0 buggy baseline (denominator includes ERROR ballots)
  majority        — strict majority of valid ballots; sub-majority → unlabeled
  majority_coded  — like majority but sub-majority resolves by CODED-preference
                    + confidence tie-break (labeled, flagged for review)
  coded_plurality — among CODED ballots only; ABSTAIN only if no CODED ballot
                    (monotone: adding a rater never unlabels a segment)

---
## A. VAAMR κ comparison (human-consensus IRR)

### MMORE_Processed

  Policy                κ (95% CI)                           % agree   n_scored  n_unlabeled  coverage
  ----------------------------------------------------------------------------------------------------
  legacy                0.597 [0.441, 0.741]  0.685           54           12  0.818 (54/66)       
  majority              0.597 [0.441, 0.741]  0.685           54           12  0.818 (54/66)       
  majority_coded        0.448 [0.299, 0.585]  0.561           66            0  1.000 (66/66)       
  coded_plurality       0.378 [0.247, 0.512]  0.485           66            0  1.000 (66/66)       

  Per-class recall (human ground truth):
  Label            legacy              majority            majority_coded      coded_plurality   
  -----------------------------------------------------------------------------------------------
  No-code          0.667 (n=18)      0.667 (n=18)      0.500 (n=24)      0.250 (n=24)    
  Vigilance        0.300 (n=10)      0.300 (n=10)      0.250 (n=12)      0.333 (n=12)    
  Avoidance        1.000 (n= 2)      1.000 (n= 2)      1.000 (n= 2)      1.000 (n= 2)    
  AttentionReg     0.900 (n=10)      0.900 (n=10)      0.900 (n=10)      0.900 (n=10)    
  Metacognition    0.500 (n= 4)      0.500 (n= 4)      0.400 (n= 5)      0.400 (n= 5)    
  Reappraisal      0.900 (n=10)      0.900 (n=10)      0.692 (n=13)      0.692 (n=13)    

### MMORE_Processed_cohort2
  Note: no qra.db (pre-migration JSONL); human IRR codes not machine-readable

---
## B. PURER coverage comparison (no human codes; coverage-only)

### MMORE_Processed  (1 rater (nvidia/nemotron-3-nano-4b))

  Policy                n_labeled  n_abstain  n_unlabeled  coverage  n_with_secondary
  ------------------------------------------------------------------------------------------
  legacy                      221          0            0     1.000               N/A
  majority                    221          0            0     1.000                23
  majority_coded              221          0            0     1.000                23
  coded_plurality             221          0            0     1.000                23

  Labeled↔unlabeled flips vs legacy:
  Policy                →labeled    →unlabeled    unchanged 
  ------------------------------------------------------------
  majority                       0             0         221
  majority_coded                 0             0         221
  coded_plurality                0             0         221

### MMORE_Processed_cohort2  (3 raters (nemotron-4b, gemma-4-4b, qwen3-8b))

  Policy                n_labeled  n_abstain  n_unlabeled  coverage  n_with_secondary
  ------------------------------------------------------------------------------------------
  legacy                       39          0            0     1.000               N/A
  majority                     39          0            0     1.000                34
  majority_coded               39          0            0     1.000                34
  coded_plurality              39          0            0     1.000                34

  Labeled↔unlabeled flips vs legacy:
  Policy                →labeled    →unlabeled    unchanged 
  ------------------------------------------------------------
  majority                       0             0          39
  majority_coded                 0             0          39
  coded_plurality                0             0          39

---
## DECISION

**Winner: `majority`**

Rationale: highest mean κ=0.597 (coverage=0.818)

Default changes applied:
  - `src/constructs/config.py` `ThemeClassificationConfig.vote_mode` → `majority`
  - `src/process/config.py` `purer_classification` default: `vote_mode=majority`

---
## Limitations

1. κ is measured on the VAAMR IRR testset items only (MMORE_Processed, n=66
   usable consensus items); the testset is also the reporting sample, so there
   is selection overlap — κ values are descriptive, not held-out.
2. PURER vote policy is unmeasurable by κ (no human PURER codes); coverage
   is the only available proxy. The winner is inherited from VAAMR.
3. MMORE_Processed_cohort2 has no SQLite qra.db (pre-migration JSONL format);
   human IRR codes are in .txt worksheets only (not machine-readable). VAAMR
   κ for cohort2 is therefore not computed. PURER coverage uses the JSONL file.
4. At n≈20 participants, κ CIs are wide — the decision is a best-available
   signal, not a high-powered experiment.
