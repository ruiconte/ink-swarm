"""Model soup: average the WEIGHTS of fine-tunes that share one init (Wortsman et al. 2022) -> a single checkpoint,
as easy to drop into HerculeX / ink-sdk as any other model. Scores the soup like cv_select (per segment, tta 8).
usage: soup.py --match all7 --out runs/soup-all7.pt"""
import argparse, glob, os, types
import numpy as np
import torch

import train as T
import official

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="all7")
    ap.add_argument("--exclude", default="scratch,aug1.5,mix")
    ap.add_argument("--kind", default="best")
    ap.add_argument("--out", default="runs/soup-all7.pt")
    ap.add_argument("--tta", type=int, default=8)
    a = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", f"{a.kind}.pt")))
             if a.match in p and not any(x in p for x in a.exclude.split(","))]
    print(len(paths), "ingredients:", [os.path.basename(os.path.dirname(p)) for p in paths], flush=True)
    acc = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=True)["model"]
        if acc is None:
            acc = {k: v.double().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
    soup = {k: (v / len(paths)).float() if v.is_floating_point() else v for k, v in acc.items()}
    torch.save({"model": soup, "args": {"arch": "official"}, "ingredients": [os.path.relpath(p, HERE) for p in paths]},
               os.path.join(HERE, a.out))
    dev = torch.device("cuda")
    m = official.OfficialInkNet("attn_max")
    m.load_state_dict(soup)
    m.to(dev).eval()
    args = types.SimpleNamespace(patch=128, depth=17, batch=32, amp="bf16", tta=a.tta)
    res = []
    for n in T.VAL_SEGMENTS:
        s = T.Segment(n)
        pr = T.predict(m, s, args, dev, region=s.val)
        r = T.metrics(pr, s.ink_eval, s.val, 0.5)
        res.append((r["bacc@0.5"], r["auc"], r["dice_best"]))
        print(n, {k: round(v, 4) for k, v in zip(("bacc", "auc", "dice"), res[-1])}, flush=True)
    r = np.array(res)
    print(f"SOUP mean bAcc {r[:,0].mean():.4f} AUC {r[:,1].mean():.4f} Dice {r[:,2].mean():.4f}")


if __name__ == "__main__":
    main()
