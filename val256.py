"""3 official validation masks, patch 256, tta 8, for public-format checkpoints. usage: val256.py path ..."""
import sys, types, numpy as np, torch
import train as T
from eval_all import load
dev = torch.device("cuda")
args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=8)
segs = [T.Segment(n) for n in T.VAL_SEGMENTS]
for p in sys.argv[1:]:
    m = load(p)[0].to(dev).eval()
    r = []
    for s in segs:
        met = T.metrics(T.predict(m, s, args, dev, region=s.val), s.ink_eval, s.val, 0.5)
        r.append((met["bacc@0.5"], met["auc"], met["dice_best"]))
    r = np.array(r)
    print(f"{p:45s} bAcc {r[:,0].mean():.4f} AUC {r[:,1].mean():.4f} Dice {r[:,2].mean():.4f}  per-seg bAcc {np.round(r[:,0],4).tolist()}", flush=True)
    del m; torch.cuda.empty_cache()
