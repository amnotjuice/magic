"""Add the two missing visual channels to every open-ended row.

Family C routes over three corrupted views -- grayscale, strong blur and
centre mask -- and picks one per sample by argmin-iic.  The older pair's
open-ended caches only ever stored grayscale, so the family cannot be screened
the way the newer pair screens it.  This computes the other two.

Each backbone and dataset keeps its own recipe, because they were built by
different probes.  The recipe is not assumed correct: every run recomputes the
real default view alongside the channels and reports how it compares against
the frozen bundle, so a wrong prompt or instrument shows up before the channels
are used.

LLaVA scores are recorded at both generation steps; step 0 is a newline on that
backbone and step 1 carries the answer.  Qwen has no leading token, so step 0
is already the answer and only that is recorded.

Usage:
  <myenv>/bin/python research/magic/oth_channels.py --combo llava_vilp_test
  <myenv>/bin/python research/magic/oth_channels.py --all
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import torch

csv.field_size_limit(10 ** 9)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LMU = Path.home() / "LMUData"
OUT = ROOT / "pipeline/data/oth_channels_cache"
POS = ROOT / "pipeline/data/llava_oth_position"
QREBUILD = ROOT / "pipeline/data/vilp_qwen_rebuild"
CHANNELS = {"strongblur": "vcf_strong_blur", "centermask": "vcf_center_mask"}
# Qwen's MME calibration cache only stored scores enveloped over the three
# wordings, which merges families A and B.  These three views separate them.
EXTRA = {"real_ans": ("answer_format", "default"),
         "blank_def": ("default", "vcf_color0"),
         "blank_ans": ("answer_format", "vcf_color0"),
         "vc_gray": ("default", "vcf_grayscale")}
COMBOS = ("llava_vilp_test", "llava_vilp_val", "llava_mme_test", "llava_mme_val",
          "qwen_vilp_test", "qwen_vilp_val", "qwen_mme_test", "qwen_mme_val")


# --------------------------------------------------------------------------
# LLaVA
# --------------------------------------------------------------------------

def llava_items(ds, split):
    """(pool ids source, image, prompt) per row, plus the frozen reference."""
    import llava_vilp_inference as SHORT
    import llava_mme_inference as BIN
    import oth_candidate_preflight as PRE

    if ds == "vilp":
        model = SHORT.build_llava(10)
        tok = model.processor.tokenizer
        tsv = LMU / f"ViLP_LLaVA-NeXT-8B_Biased_{split.capitalize()}.tsv"
        by_key = {}
        with tsv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                by_key[str(int(float(row["index"])))] = row
        tc = PRE.load_tc_logs("llava", split.capitalize())
        items = []
        for (d, key), rec in tc.items():
            if d != "ViLP" or str(key) not in by_key:
                continue
            ids = {c: SHORT.label_token_ids(tok, c) for c in rec.get("candidates", []) if c}
            ids = {c: t for c, t in ids.items() if t}
            if not ids:
                continue
            items.append((str(key), ids,
                          SHORT.materialize_image(by_key[str(key)], tsv.stem),
                          SHORT.GEN_PROMPTS["default"].format(q=str(rec["question"])),
                          None))
        return model, items

    model = BIN.build_llava()
    tok = model.processor.tokenizer
    ab = {"A": BIN.label_token_ids(tok, "A"), "B": BIN.label_token_ids(tok, "B")}
    stem = f"MME_LLaVA-NeXT-8B_Biased_{split.capitalize()}"
    n = 513 if split == "test" else 80
    rows = pd.read_csv(LMU / f"{stem}.tsv", sep="\t").to_dict("records")[:n]
    items = [(BIN.row_key(r), ab, BIN.materialize_image(r, stem),
              BIN.BINARY_OPTION_PROMPTS["default"].format(q=str(r["question"])),
              {"A": "Yes", "B": "No"}) for r in rows]
    return model, items


def llava_scores(model, msg, visual, ids, rename):
    old = model.kwargs
    model.kwargs = {**old, "max_new_tokens": 2, "return_dict_in_generate": True,
                    "output_logits": True}
    try:
        with torch.inference_mode():
            lg = model.generate_ids_or_logits(msg, visual_type=visual,
                                              textual_type="default",
                                              get_logits=True).detach().float().cpu()
    finally:
        model.kwargs = old
    out = {}
    for p in range(min(2, lg.shape[1])):
        z = torch.log_softmax(lg[0, p], -1)
        sc = {c: round(max(float(z[t]) for t in tid), 5) for c, tid in ids.items()}
        out[p] = {rename[k]: v for k, v in sc.items()} if rename else sc
    return out


def run_llava(ds, split):
    model, items = llava_items(ds, split)
    out = []
    for key, ids, img, prompt, rename in items:
        rec = {"key": key}
        for name, vis in [("real_def", "default")] + list(CHANNELS.items()):
            msg = [{"type": "image", "value": str(img)},
                   {"type": "text", "value": prompt}]
            for p, sc in llava_scores(model, msg, vis, ids, rename).items():
                rec[f"{name}@{p}"] = sc
        out.append(rec)
        if len(out) % 50 == 0:
            print(f"{len(out)} rows", flush=True)
    return out


# --------------------------------------------------------------------------
# Qwen
# --------------------------------------------------------------------------

def qwen_model():
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
    m = Qwen2VLChat(model_path="/data/shunshungu/models/Qwen2-VL-7B-Instruct",
                    min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28,
                    model_name=None, save_logits=False, dump_path=None,
                    dtype=torch.bfloat16, visual_type="default", textual_type="default")
    m.generate_kwargs["return_dict_in_generate"] = True
    m.generate_kwargs["output_logits"] = True
    return m


def qwen_items(ds, split, model):
    import qwen_mme_inference as LIFT
    tok = model.processor.tokenizer
    bos = getattr(tok, "bos_token_id", None)

    def fti(c):
        s = set()
        for t in (c, " " + c, c.lower(), " " + c.lower()):
            x = [i for i in tok(t).input_ids if i != bos]
            if x:
                s.add(x[0])
        return sorted(s)

    if ds == "vilp":
        from vlmeval.dataset import build_dataset
        dsname = f"ViLP_Qwen2-VL-7B_Biased_{split.capitalize()}"
        d = build_dataset(dsname)
        model.set_dump_image(d.dump_image)
        src = json.loads((QREBUILD / f"vilp_qwen_{split}.json").read_text())
        by_i = {r["row_i"]: r for r in src}
        items = []
        for i in range(len(d.data)):
            if i not in by_i:
                continue
            raw = d.data.iloc[i]
            struct = (model.build_prompt(raw, dataset=dsname)
                      if model.use_custom_prompt(dsname) else d.build_prompt(raw))
            msg = [{"role": "user", "content": model._prepare_content(struct, dataset=dsname)}]
            ids = {c: fti(c) for c in by_i[i]["pool"]}
            items.append((str(by_i[i]["idx"]), {c: t for c, t in ids.items() if t}, msg, None))
        return items

    stem = f"MME_Qwen2-VL-7B_Biased_{split.capitalize()}"
    n = 222 if split == "test" else 68
    rows = pd.read_csv(LMU / f"{stem}.tsv", sep="\t").to_dict("records")[:n]
    ab = {"A": [i for i in [tok("A").input_ids[0]]], "B": [i for i in [tok("B").input_ids[0]]]}
    ab = {k: sorted({tok(t).input_ids[0] for t in (k, " " + k)}) for k in ("A", "B")}
    items = []
    img_dir = OUT / "qwen_mme_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        path = img_dir / f"{stem}_{r['index']}.jpg"
        if not path.exists():
            LIFT.load_image(r).save(path, format="JPEG", quality=90)
        msgs = {}
        for pk in ("default", "answer_format"):
            struct = [{"type": "image", "value": str(path)},
                      {"type": "text", "value":
                       LIFT.BINARY_OPTION_PROMPTS[pk].format(q=str(r["question"]))}]
            msgs[pk] = [{"role": "user",
                         "content": model._prepare_content(struct, dataset=None)}]
        items.append((str(r["index"]), ab, msgs, {"A": "Yes", "B": "No"}))
    return items


def qwen_scores(model, msg, visual, ids, rename):
    lg = model.generate_ids_or_logits(msg, visual_type=visual, textual_type="default",
                                      get_logits=True).detach().float().cpu()
    v = lg[0, 0] if lg.dim() == 3 else lg.squeeze()
    if v.dim() > 1:
        v = v[-1]
    z = torch.log_softmax(v, -1)
    sc = {c: round(max(float(z[t]) for t in tid), 5) for c, tid in ids.items()}
    return {rename[k]: v for k, v in sc.items()} if rename else sc


def run_qwen(ds, split, extra=False):
    model = qwen_model()
    items = qwen_items(ds, split, model)
    out = []
    for key, ids, msg, rename in items:
        rec = {"key": key}
        spec = ([("real_def", ("default", "default"))] +
                ([(k, v) for k, v in EXTRA.items()] if extra else
                 [(k, ("default", v)) for k, v in CHANNELS.items()]))
        for name, (pk, vis) in spec:
            m = msg[pk] if isinstance(msg, dict) else msg
            rec[f"{name}@0"] = qwen_scores(model, m, vis, ids, rename)
        out.append(rec)
        if len(out) % 50 == 0:
            print(f"{len(out)} rows", flush=True)
    return out


# --------------------------------------------------------------------------

def report(rows, mk, ds, split):
    """How the recomputed real default view compares against what is on file."""
    step = 1 if mk == "llava" else 0
    ref = None
    if mk == "llava" and (POS / f"llava_{ds}_{split}_positions.json").exists():
        ref = {r["key"]: r.get(f"real_def@{step}")
               for r in json.loads((POS / f"llava_{ds}_{split}_positions.json").read_text())
               if "key" in r}
    elif mk == "qwen" and ds == "vilp":
        ref = {str(r["idx"]): r["real_def"]
               for r in json.loads((QREBUILD / f"vilp_qwen_{split}.json").read_text())}
    elif mk == "qwen" and ds == "mme":
        fz = [r for r in json.load(gzip.open(ROOT / "magic_oth_bundle.json.gz", "rt"))["qwen"]
              if r["binary"]]
        if split == "test" and len(fz) == len(rows):
            ref = {r["key"]: f["real_def"] for r, f in zip(rows, fz)}
    if not ref:
        print("  no reference on file for this combo -- channels written unchecked")
        return
    d = [max(abs(r[f"real_def@{step}"][c] - ref[r["key"]][c]) for c in ref[r["key"]])
         for r in rows if r["key"] in ref and
         set(r[f"real_def@{step}"]) == set(ref[r["key"]])]
    if not d:
        print("  reference rows did not line up -- channels written unchecked")
        return
    d.sort()
    print(f"  real_def vs file (n={len(d)}): exact {sum(x == 0 for x in d)}, "
          f"median |delta| {d[len(d) // 2]:.4f}, max {d[-1]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", choices=COMBOS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="the three views that separate families A and B")
    args = ap.parse_args()
    todo = COMBOS if args.all else (args.combo,)
    OUT.mkdir(parents=True, exist_ok=True)
    for combo in todo:
        mk, ds, split = combo.split("_")
        print(f"\n=== {combo} ===", flush=True)
        rows = (run_llava(ds, split) if mk == "llava"
                else run_qwen(ds, split, args.extra))
        report(rows, mk, ds, split)
        name = f"{combo}_extra.json" if args.extra else f"{combo}.json"
        (OUT / name).write_text(json.dumps(rows))
        print(f"  wrote {OUT / name}  {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
