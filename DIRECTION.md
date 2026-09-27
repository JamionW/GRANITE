# GRANITE direction (2026-09-27)

Authoritative for scope, title, and priorities. Supersedes the direction in any document that predates it. The committed tree remains the source of truth for code and numbers. Schedule of record: the GRANITE Post-Proposal Plan tracker, maintained outside the repo.

## Status

Proposal passed with conditions on 2026-09-23.

## Title

Working title (pending committee): "When Do Tract Means Reveal Address-Level Vulnerability? Graph Neural Networks Under Aggregate Supervision."
Retired: "Boundary Conditions for Constrained Graph Neural Network Spatial Disaggregation," and "boundary conditions" and "constrained" as framing terms.

## Question

How much within-tract variation can tract-mean supervision recover, and what governs it. The framing is conditions for recovery, not performance claims. Null results are an honest contribution.

## Statistical spine

For a linear learner trained on tract means, within-tract recovery equals the address-supervised ceiling times the within-covariance cosine between the between-tract and within-tract coefficients, less a finite-sample penalty that grows as feature count approaches tract count. It is exact only for linear learners. Its novelty is unconfirmed pending review by Dr. Gao.

## Work in scope

- Paper 1: coherence without recovery. The graph supplies coherence, not recovery (5b topology sweep), and pooled BG r is pinned by the constraint (M0). Adds one synthetic randomization run with known truth.
- Paper 2: the synthetic testbed. Linear benchmark first, then GraphSAGE and a minimal hierarchical GNN (address, block-group, and tract nodes), with supervision depth (tract only, tract plus block group) as a factor.
- GCN-GAT stays in Paper 1 only.

## Cut or demoted

Soft-penalty versus projection comparison; weighted GCN; GATv2; the external-target program (the delinquency pilot stays as a preliminary null); the 85-tract run unless time frees up; IDW and kriging (graveyard); hard-constraint framing; block-group r as a benchmark.

## Scheduled milestones carried from proposal hardening

- Variance components: split the 81-draw grid's per-tract paired difference into tract and seed components; report the implied detectable effect at 85 tracts for 3, 5, 9, and 15 seeds.
- Generator changes: a ceiling fit on the feature matrix; per-draw within-tract variation reporting; rejection of unknown parameter keys; a continuous weight mixing feature signal and spatial component; a tract-level effect parameter that sets alignment.
Neither starts until its prompt arrives.

## Known phantom figures (never reuse without a committed artifact)

- 0.844: Dasymetric single-BG constrained_r from a different experiment; canonical aggregate is 0.802.
- 11.742: a measured spatial mean degree, not an external constant.
- r=0.671: a prediction-to-prediction cross-mode correlation, not a feature-to-target statistic.
- r=0.033: address-level accessibility correlation from a deleted experiment; retired in commit 7dc0617.

## Working rules

- Design decisions go to a person before implementation.
- One milestone per prompt; read-only recon before edits; frozen artifacts never rewritten; deprecated code to graveyard/; absolute imports per IMPORTS.md.
- Every number and path traces to a command run with output pasted.
