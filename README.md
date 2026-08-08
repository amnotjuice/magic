# MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration

A training-free, test-time debiasing method for vision-language models. The
model's own decision margin drives a two-rung ladder that spends correction
compute only where the answer is contestable:

```
rung k fires iff  margin(state) <= theta_k          (allocation)
rung 1 (prior isolation): REPLACE the state with the clean2 lift
rung 2 (visual residual, ADD):     commit iff
        margin(state + residual) - margin(state) >= tau_V   (commitment)
```

The visual residual draws from a 3-channel corruption set (grayscale,
blur, centre-mask). On multiple-choice questions the method *routes* to
the most task-relevant channel; on binary/open-ended answers, where
routing gives no measured gain, it defaults to grayscale.

Baselines compared against in the paper include SCI (Tang et al.), VCD,
and M3ID.

## Quick start

```bash
pip install -r requirements.txt
python3 magic.py
```

`magic.py` is stdlib-only (no torch, no GPU) and reproduces every frozen
Test-set number reported in the paper bit-exactly from the two score
bundles shipped alongside it (`magic_mcq_bundle.json.gz`,
`magic_oth_bundle.json.gz`). A successful run prints `ALL PARITY OK`.

## Repository layout

```
magic.py                        # reference implementation (root, zero deps)
magic_mcq_bundle.json.gz        # frozen MCQ score bundle
magic_oth_bundle.json.gz        # frozen open-ended score bundle
experiments/                    # scripts that reproduce every paper table/figure
    README.md                   # script -> table/figure map + dependency tiers
    data/                       # the ~43M of frozen provenance these scripts read
pipeline/                       # code that produced the bundles: run the model,
    README.md                   # collect VCF/TCF views, assemble the bundles
    vlmeval/                    # trimmed VLMEvalKit v0.2 fork (2 patched wrappers)
    data/                       # small caches the bundle-builders read (~19M)
```

## Reproducing results

See [`experiments/README.md`](experiments/README.md) for the full
script-to-table/figure mapping and what each script needs to run.

## Scope

This repository contains the method (`magic.py`), the code that
reproduces the paper's reported numbers from already-collected data
(`experiments/`), and the code that produced that data in the first place
by running the base VLMs under VCF/TCF perturbations (`pipeline/`).

`pipeline/` needs a GPU, real model checkpoints, and the benchmark
datasets to actually run — it hasn't been re-executed in this environment,
and it has two disclosed, real gaps in the underlying raw data (some
dumps were never written in the original collection run; see
[`pipeline/README.md`](pipeline/README.md#known-gaps)). It is the same
logic that produced the shipped bundles, not a from-scratch reproof of
them.

## License

Apache-2.0, see [LICENSE](LICENSE). See [NOTICE](NOTICE) for attribution.

## Citation

```bibtex
@inproceedings{magic2026,
  title     = {MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration for VLM Test-time Robustness Scaling},
  author    = {TODO},
  booktitle = {CVPR},
  year      = {2026}
}
```
