"""Same exam for every model: AUC / best Dice / balanced accuracy on the three official online-validation masks
(pherc0139-w016, pherc0814-46527, pherc1667-w029), same normalization, same sliding windows, tta 1 (and 8 for the top).
Downloads public checkpoints (scrollprize/ink_9um x14, reader-v2, reader-v3, ink9um-dense x4, Nieuwlaar native x2)
into official/ and writes runs/baselines.json.
usage: eval_all.py [extra.pt ...]"""
import json, os, sys, time, types
import numpy as np
import torch

import official
import prep
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.path.join(HERE, "official")
HF = "https://huggingface.co"
CKPTS = {}
for seed in (42, 43):
    for step in ("010000", "020000", "030000", "040000", "050000", "060000", "075000"):
        CKPTS[f"ink9um-s{seed}-{step[:2]}k"] = f"{HF}/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed{seed}/step-{step}.pth"
CKPTS.update({
    "reader-v3": f"{HF}/domenicor046/reader-v3/resolve/main/reader-v3.pth",
    "reader-v2-40k": f"{HF}/domenicor046/reader-v2/resolve/main/reader-v2-step040000.pth",
    "dense-w016excl-75k": f"{HF}/domenicor046/ink9um-dense/resolve/main/dense9um-w016excluded-step075000.pth",
    "dense-all7-75k": f"{HF}/domenicor046/ink9um-dense/resolve/main/dense9um-all7-step075000.pth",
    "nieuwlaar-dense-native": f"{HF}/Nieuwlaar/ink9um-dense-native/resolve/main/weights/dense_native_016000.safetensors",
    "nieuwlaar-soup42": f"{HF}/Nieuwlaar/ink9um-dense-native/resolve/main/weights/soup42_last4.safetensors",
})


def fetch(name, url):
    ext = ".safetensors" if url.endswith(".safetensors") else ".pth"
    p = os.path.join(OFF, name + ext)
    if not os.path.exists(p):
        r = prep.S.get(url, headers=prep.HF_AUTH, timeout=900)
        r.raise_for_status()
        open(p, "wb").write(r.content)
    return p


def load(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck["model"] if "model" in ck else ck
        if "args" in ck and ck.get("args", {}).get("arch", "official") != "official":
            a = types.SimpleNamespace(**ck["args"])
            m = T.InkNet(a.depth, a.stem, a.width, a.levels, a.zproj, getattr(a, "stem_blocks", 0))
            m.load_state_dict(sd)
            return m, "ours"
    sd = {k.replace("module.", ""): v for k, v in sd.items() if ".all_modules." not in k}
    m = official.OfficialInkNet("attn_max")
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    note = "ok" if not missing and not unexpected else f"missing {len(missing)} unexpected {len(unexpected)}"
    return m, note


def main():
    os.makedirs(OFF, exist_ok=True)
    dev = torch.device("cuda")
    segs = [T.Segment(n) for n in T.VAL_SEGMENTS]
    out_path = os.path.join(HERE, "runs", "baselines.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    items = list(CKPTS.items()) + [(os.path.basename(p), None) for p in sys.argv[1:]]
    for name, url in items:
        if name in res:
            continue
        try:
            path = fetch(name, url) if url else name
            m, note = load(path if url else sys.argv[[os.path.basename(x) for x in sys.argv].index(name)])
        except Exception as e:
            print(f"{name}: cannot load ({str(e)[:120]})", flush=True)
            continue
        m = m.to(dev).eval()
        a = types.SimpleNamespace(patch=128, depth=17, batch=32, amp="bf16", smooth=0.5, tta=1)
        per = {}
        t = time.time()
        for s in segs:
            prob = T.predict(m, s, a, dev, region=s.val)
            per[s.name] = T.metrics(prob, s.ink, s.val, 0.5)
        mean = {k: float(np.mean([per[n][k] for n in per])) for k in ("auc", "dice_best", "bacc@0.5")}
        res[name] = {"mean": mean, "per": {n: {k: per[n][k] for k in ("auc", "dice_best", "bacc@0.5")} for n in per}, "load": note}
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"{name:26s} AUC {mean['auc']:.4f}  Dice {mean['dice_best']:.4f}  bAcc {mean['bacc@0.5']:.4f}  "
              + "  ".join(f"{n.split('-', 1)[1]} {per[n]['auc']:.3f}" for n in per) + f"  [{note}] {time.time() - t:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
