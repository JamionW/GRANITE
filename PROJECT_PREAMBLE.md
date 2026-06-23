GRANITE (project codename; original acronym retired)

What it is
Spatial disaggregation of the CDC Social Vulnerability Index from census tract resolution to individual addresses, using a graph neural network that treats tract SVI as a soft aggregate constraint (tract mean) rather than a prediction target. Study area: Hamilton County, Tennessee, FIPS 47065. 85 tracts carry SVI (87 total), approximately 102,761 addresses, 102,330 matched to assessor parcels. PhD dissertation, UTC; chair Dr. Mehdi Khaleghian, co-chair Dr. Yu Liang. Repo JamionW/GRANITE, branch main.

Research question
Under what conditions does soft aggregate constraint enforcement shape learned spatial disaggregation, and which feature classes survive constraint correction. The framing is boundary conditions, not performance claims. Null results are the honest contribution.

Technical frame
Input: 73 address-level features (parcel attributes, Microsoft Building Footprints, FEMA flood zones, NLCD 2021 land cover, multi-modal OSRM accessibility, socioeconomic controls). Constraint: tract-level SVI as a soft tract-mean penalty. Architectures compared: GraphSAGE and GCN-GAT. Output: address-level SVI estimates with within-tract spatial coherence under aggregate consistency.

Production graph (canonical, supersedes any doc calling it spatial k-NN or claiming cross-tract edges)
Built per tract by the road-network hybrid in granite/data/loaders.py. Two-pass: road-network shortest paths first (OSM snapping, inverse path-length weights), geographic k-NN (k=6, Gaussian decay) as orphan-prevention fallback. Each tract built independently; message-passing graphs carry zero cross-tract edges. Measured mean degree 12.162 (39535 nodes, 20 tracts). Moran's I scores against a separate fixed external k=8 coordinate weights matrix, independent of the message-passing graph.

Validation reality
No address-level SVI ground truth exists by construction; get_address_truth_values(target='svi') returns None. Honest comparator is Dasymetric (single-attribute NLCD impervious-surface ancillary, Nguyen et al. 2021). IDW and kriging are retired to graveyard/ for reproducibility only; the old "IDW beats GRANITE 0.558 vs 0.469" framing is dead. Canonical pooled BG r (69 BGs, m0): GRANITE 0.769, Dasymetric 0.802, Pycnophylactic 0.768, confidence intervals fully overlapping, methods not statistically separable. Per-tract median BG r: Dasymetric 0.787, GRANITE 0.390, also not separable (95% CI spans zero). Per-tract recovery r is the primary metric. Pooled BG r is misleading for constraint-preserving methods because the constraint pins aggregate recovery by construction.

Six locked findings
1. Architecture-dependent feature survival: GraphSAGE and GCN-GAT select different features under identical soft-constraint loss.
2. Ecological fallacy at address scale: features with high tract-level SVI coupling produce near-zero address-level signal.
3. Within-tract feature redundancy (M3.5): GBM recovers any held-out feature from the rest at r approaching 1.0, so external targets are required for valid recovery tests.
4. Constraint-as-tax: the soft constraint behaves as a tax the trainer partially escapes under strong feature-target coupling.
5. bg_r flat: the graph supplies within-tract spatial structure (Moran's I), not block-group generalization accuracy.
6. Pooled BG r is the wrong primary metric for constraint-preserving disaggregation; per-tract recovery r is load-bearing.

Executive summary (Jamion confirmed)
"We proved a sophisticated model disaggregates vulnerability scores no more accurately than simple proportional allocation, and its only distinctive contribution, spatial coherence, is wholly a function of graph structure rather than learning."

Operating contract
Two instances: a strategic instance for architecture, framing, and milestone prompt authoring; Claude Code in Codespaces (/workspaces/GRANITE, branch main) for implementation, file access, git. One milestone per prompt with explicit scope, files to read, files to create, files to modify in place, behavioral spec, acceptance criteria, when-done report. Read-only recon before edits. In-place function updates, no new names that orphan existing code. Deprecated code to graveyard/, absolute imports per IMPORTS.md. Frozen artifacts never rewritten; new artifacts land alongside. One logical purpose per commit; append SESSION_LOG and Research_Status each step. The committed tree is the source of truth; project-context docs lag and do not override it. Verify the artifact, not the claim: every number and path traces to a command run with output pasted.

Known phantom figures (trace to a committed artifact before reuse)
0.844 Dasymetric (single-BG constrained_r from a different experiment; canonical aggregate is ~0.802). 11.742 "calibration target" (a measured spatial mean degree, not an external constant). r=0.671 (a prediction-to-prediction cross-mode correlation, not a feature-to-target statistic).

Style
No em dashes or en dashes. Active voice. Direct functional statements, no contrastive constructions. Terse, signal-dense, prose-first, minimal headers. Pushback expected both directions; flagging drift toward optimism is the job.
