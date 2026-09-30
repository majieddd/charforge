# Research source and methods audit — 2026-09-26

## Scope and handling

Four parallel read-only bibliography lanes audited 46 legacy entries, and two read-only method lanes audited current protocols/code. Findings were checked against local source and primary pages before editing. Existing measurements, model outputs, and dirty worktree changes were preserved. No model, Blender, GPU, `paper/collect.py`, or experiment job was run in this pass. The paper builder remains the intended structural/citation check; it does not validate every hand-maintained result.

## Source corrections recorded

- Separated the exact 2,453 count from Adobe's product page: the count is independently reproducible in the pinned [jasongzy Mixamo mirror snapshot](https://huggingface.co/datasets/jasongzy/Mixamo/tree/b1c7f4975ea3261d3d0aa2379f6e24754ccde9d8). The dataset is third-party, and access requires accepting conditions that include sharing contact information. Adobe's FAQ separately states project use, including commercial games; that statement alone does not establish redistribution rights for this mirror/raw files.
- Corrected the hand-animation paper/glTF distinction: Magnenat-Thalmann et al. is about joint-dependent local deformations; glTF is the source for its linear-blend skinning path. DQS reduces, but does not eliminate, collapse in CharForge's measured case.
- Distinguished Qwen-Image/Qwen-Image-2.0 papers from the actual Qwen-Image 2.1 repository and its [Qwen Research License](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE), which limits use to non-commercial research/evaluation unless a separate commercial license is obtained.
- Added the [TRELLIS.2 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xiang_Native_and_Compact_Structured_Latents_for_3D_Generation_CVPR_2026_paper.html) and separate [official project-page notice](https://microsoft.github.io/TRELLIS.2/): the page calls it a research project and says its materials are solely for academic/research use, not commercial exploitation. This does not determine generated-output rights or replace review of code/weight licenses.
- Split MiniMax H3 into the actual community Ref2VA fine-tune, upstream H3 model, primary [upstream license text](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), and Comfy-Org encoder/VAE packaging. The fine-tune card declares Apache-2.0, while the upstream agreement defines H3 Works to include model derivatives and its use policy restricts Outputs outside the Applicable Territory. The relationship is unresolved; no legal conclusion is made.
- Narrowed the Cinevva/AI3DGen prose to features actually advertised by each vendor; removed the unsupported assertion that both share a particular backend or both add rigging/preset clips. Updated source dates/versions and clarified LBS, PINOC, OpenVDB, HMR and DINOv3 attribution details.

## Method/code reconciliation

E102 remains planned; no new measurements are claimed. Current source inspection shows `blender/aberration_audit.py` now:

- fails when evaluated loop-triangle indexing/order differs from rest before reusing masks;
- computes a centered local-shape chord residual and attributes it to the center sample;
- reports `n-2` eligible windows on the explicit surface cadence;
- has a default no-clobber path, with replacement requiring `--force`;
- labels BVH overlap/rest-centroid-filter pairs as broad-phase candidates, not intersections; and
- includes per-measure worst-sample render paths.

These are code-level observations only. Clean controls, injected defects, schedule sensitivity, renderer execution, and the Blender runtime path are not verified. The candidate normalization in code uses all possible triangle pairs (`nF choose 2`) while candidate eligibility filters for live triangles; the tracker now makes that denominator explicit and does not claim a live-pair denominator. Normal-ray exposure is not camera visibility; floor flags are not contact-conditioned; runtime morphs, texture drift, and narrow-phase distance/intersection checks remain separate gaps.

## Rights/provenance follow-up

The audit found committed H3-generated demonstration videos in `paper/media/` and `docs/paper/media/`, despite the earlier README sentence that nothing from H3 ships. The README now corrects that statement. Historical files were not removed or overwritten. E115 requires their source/terms and public copies to be inventoried before external distribution. It also requires exact checkpoint/hash/revision and location/use restrictions, and an explicit review gate for contradictory base/fine-tune licenses. Unknown rights stay unknown; this note is not legal advice.

Other source-specific gates now recorded in the paper/README/tracker: Qwen 2.1 non-commercial research/evaluation only; TRELLIS.2 project materials research-only; H3 territory/output restrictions and fine-tune license conflict; and Mixamo mirror access/redistribution provenance. These terms should be checked again before a release because pages and agreements can change.

## Queue and next verification

- Keep E102/E115 and earlier experiments ahead of new model trials. The user-requested classifier from the supplied video remains E127, the final tracker entry; the linked video/model has not been identified or investigated.
- When the GPU/Studio lock is free, run E102 synthetic clean controls and seeded one-frame/pop/intersection defects at step=1 / bvh-step=1, then compare cheaper cadences. Keep outputs in a new run folder and do not replace historical results.
- Before any further H3 run or external distribution of H3-derived media, reconcile the official agreement and fine-tune label with qualified rights review/authorization as applicable. Preserve the existing files while that question is assessed.
- Continue the staged packets in `research/GAPS.md`; only mark conclusions complete after their predeclared tests and artifacts exist.
