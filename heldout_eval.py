"""Pre-publication checks on data no Argus model saw:
  1. never-seen scroll PHerc0841 w00: AUC, bAcc@0.5, best Dice inside the label supervision mask
  2. false ink: threshold each model at 80% ink recall on 0841 (same recall for everyone), then report
     - background of 0841 marked as ink
     - blank papyrus PHerc0139 w053 marked as ink (reader-v3's own release reports 5.5% on w053)
  3. reversed depth order on 0841 (hallucination control: ink shapes should collapse)
usage: heldout_eval.py [--tta 8]"""
import argparse, json, os, types
import numpy as np
import torch

import train as T
from ensemble import load
from eval_all import load as load_public

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"Argus V1 (soup)": ("ours", "runs/soup-all7-v2.pt"),
          "dense-all7": ("pub", "official/dense-all7-75k.pth"),
          "dense-native": ("pub", "official/nieuwlaar-dense-native.safetensors"),
          "reader-v3": ("pub", "official/reader-v3.pth"),
          "ink_9um s43-20k": ("pub", "official/ink9um-s43-02k.pth"),
          "Argus nieu-soup": ("ours", "runs/soup-nieu.pt"),
          "Argus zflip0-soup": ("ours", "runs/soup-z0.pt"),
          "Argus V1 release": ("pub", "release/argus-v1/argus-v1.pth")}


class Reversed(T.Segment):
    """same segment, depth order flipped"""
    def __init__(self, s):
        self.__dict__.update(s.__dict__)
        self.vol = np.asarray(s.vol)[::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--only", default="", help="comma-separated model names to run")
    ap.add_argument("--extra", default="", help="name=path of one more of our checkpoints (runs only it)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    args = types.SimpleNamespace(patch=a.patch, depth=17, batch=32 if a.patch == 128 else 8, amp="bf16", tta=a.tta)
    s41 = T.Segment("pherc0841-w00")
    s53 = T.Segment("pherc0139-w053")
    rev = Reversed(s41)
    lab, reg = s41.ink_eval, s41.val
    res = {}
    if a.extra:
        n, pth = a.extra.split("=", 1)
        MODELS[n] = ("ours", pth)
        a.only = n
    for name, (kind, path) in MODELS.items():
        p = os.path.join(HERE, path)
        if not os.path.exists(p) or (a.only and name not in a.only.split(",")):
            continue
        m = (load(p) if kind == "ours" else load_public(p)[0]).to(dev).eval()
        pr = T.predict(m, s41, args, dev, region=reg)
        met = T.metrics(pr, lab, reg, 0.5)
        ink_p = pr[reg & lab]
        t80 = float(np.quantile(ink_p, 0.20))                 # 80% of the labelled ink is above this
        bg = float((pr[reg & ~lab] >= t80).mean())
        p53 = T.predict(m, s53, args, dev, region=s53.val)
        fa53 = float((p53[s53.val] >= t80).mean())
        prr = T.predict(m, rev, args, dev, region=reg)
        mr = T.metrics(prr, lab, reg, 0.5)
        res[name] = dict(auc_0841=met["auc"], bacc_0841=met["bacc@0.5"], dice_0841=met["dice_best"],
                         bg_as_ink_at80=bg, blank_w053_as_ink_at80=fa53, auc_0841_reversed=mr["auc"])
        print(f"{name:18s} 0841 AUC {met['auc']:.4f} bAcc {met['bacc@0.5']:.4f} Dice {met['dice_best']:.4f} | "
              f"false ink @80% recall: 0841 background {bg*100:.1f}%  blank w053 {fa53*100:.1f}% | reversed AUC {mr['auc']:.4f}",
              flush=True)
        os.makedirs(os.path.join(HERE, "runs", "heldout"), exist_ok=True)
        np.save(os.path.join(HERE, "runs", "heldout", f"0841_{name.split()[0]}.npy"), pr.astype(np.float16))
        del m
        torch.cuda.empty_cache()
    out = os.path.join(HERE, "runs", "heldout_results.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    json.dump({**old, **res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
