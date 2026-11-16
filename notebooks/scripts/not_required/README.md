# Not required to reproduce the paper's results

None of these are read by any script in the required pipeline (`../REPRODUCE.md`
at the repo root has the full ordered list). They're kept for provenance — showing
the due diligence behind decisions made in the required pipeline — not because
re-running them changes any reported number.

**Proof-of-concept scripts** that justified a design choice on a small sample
(n≈1,500) before the required pipeline committed to it at full scale:
- `nb8_attnpool_poc.py`, `nb8_pretrained_pool_poc.py` — established that V-JEPA2's
  own frozen pretrained SSv2 attentive pooler beats naive mean-pooling. This
  conclusion is why `05_extract_vjepa2_pretrainedpool_embeddings.py` exists.
- `nb8_framecount_poc.py` — tested feeding more frames than the checkpoint was
  trained on; inconclusive, not adopted.
- `nb8_poolerfinetune_poc.py`, `nb8_poolerfinetune_poc_v2.py` — tested letting the
  pretrained pooler fine-tune instead of staying frozen. Result: worse than frozen
  (-5.3pp, overfits on 1,200 examples), no motion-specific benefit. This ruled-out
  result is cited directly in the paper text; reproducing the citation doesn't
  require rerunning the script.

**Incomplete / abandoned extensions** — started, never finished at a scale that
would support a claim:
- `nb8_vjepa2_attnpool_train.py`, `nb8_vjepa2_temporal_extract.py` — a learned
  per-timestep attention-pooling head, blocked on a per-timestep extraction that
  was only ever completed for 203 of 13,003 segments.
- `nb8_video_adapter_full_eval.py` — a nonlinear video-side adapter; only its LR
  sweep ran (inconclusive, no improvement over the pretrained-pool headline).

**Superseded** — produced a result the paper no longer relies on:
- `nb8_singleframe_motion_test.py` — fed the original motion/static verb split,
  a single person's linguistic categorization. Superseded by
  `09_temporal_order_directional_vs_cyclic_verbs.py` in the required pipeline,
  which is grounded in measured order-sensitivity instead.

**Utility, not analysis:**
- `nb8_smoketest.py` — a smoke test.
- `dinov2_zeroshot_fullval_fixed.py`, `dinov3_zeroshot_fullval_fixed.py` — optional
  cache warmers for zero-shot results that `01_train_matched_projection_baselines.py`
  regenerates itself if these outputs aren't already present.
