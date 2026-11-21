"""Read one deep surface with Priyank's Studio model (PHerc0211 winner, CC-BY-NC) next to our readers.
Calibrated on PHerc1447 w058 (expert labels): Studio takes the same layer order as Argus V3 ('as-is', AUC 0.872 vs
Argus V3 0.869; reversed order 0.47-0.53 = clean control), 16 planes around the sheet. Here: sheet tracked pixel by
pixel (same rule as final4), 16 planes, depths -2/0/+2 averaged, page = the side final4 chose for Argus V3, control =
the other side. Writes studio_page.png / studio_ctl.png, isoflat versions, and p_studio.npz (fwd/rev).
usage: studio_read.py DEEP_DIR"""
import os, sys, json, subprocess, numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, "ext/studio")
from model import DeepResidualUNet
from safetensors.torch import load_file
D_ = sys.argv[1]
sv = np.load(os.path.join(D_, "sv.npy"), mmap_mode="r"); L = sv.shape[0]; c0 = L // 2; valid = np.asarray(sv[c0]) > 0; H, W = valid.shape
sub = np.stack([ndimage.zoom(np.asarray(sv[j], np.float32), 1 / 8, order=1) for j in range(c0 - 5, c0 + 6)])
d = np.argmax(ndimage.gaussian_filter(sub, (1, 2, 2)), 0).astype(np.float32) - 5
d = ndimage.gaussian_filter(ndimage.median_filter(d, 9), 4); Dm = np.clip(ndimage.zoom(d, (H / d.shape[0], W / d.shape[1]), order=1)[:H, :W], -5, 5)
SVA = np.ascontiguousarray(sv)
def window(off, n=16):
    win = np.zeros((n, H, W), np.uint8)
    for j in range(n):
        z = c0 - n // 2 + j + off + Dm; z0 = np.clip(np.floor(z).astype(int), 0, L - 2); f = (z - z0).astype(np.float32)
        zmin, zmax = int(z0.min()), int(z0.max()) + 1; blk = SVA[zmin:zmax + 1]; k = (z0 - zmin)[None]
        lo = np.take_along_axis(blk, k, 0)[0].astype(np.float32); hi = np.take_along_axis(blk, k + 1, 0)[0].astype(np.float32)
        win[j] = np.clip((1 - f) * lo + f * hi, 0, 255)
    return win
dev = torch.device("cuda")
m = DeepResidualUNet(deeper=False); m.load_state_dict(load_file("ext/studio/studio_softloss_ema.safetensors", device="cpu"), strict=True); m = m.eval().to(dev)
K = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96); K = (K[:, None] * K[None]).astype(np.float32)
@torch.inference_mode()
def studio(vol):
    out = np.zeros((H, W), np.float32); wt = np.zeros((H, W), np.float32)
    pad = np.pad(vol, ((0, 0), (16, 112), (16, 112)), mode="reflect")
    pos = [(y, x) for y in range(0, H, 64) for x in range(0, W, 64) if valid[y:y + 96, x:x + 96].any()]
    for i in range(0, len(pos), 8):
        b = pos[i:i + 8]; t = torch.from_numpy(np.stack([pad[:, y:y + 128, x:x + 128] for y, x in b])[:, None]).to(dev).float().clamp_(0, 200).div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = m(t)[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
        for (y, x), q in zip(b, p):
            hh, ww = min(96, H - y), min(96, W - x); out[y:y + hh, x:x + ww] += q[:hh, :ww] * K[:hh, :ww]; wt[y:y + hh, x:x + ww] += K[:hh, :ww]
    return np.where(valid, out / np.maximum(wt, 1e-9), 0)
acc = {"fwd": 0, "rev": 0}
for off in (-2, 0, 2):
    w = window(off)
    acc["fwd"] = acc["fwd"] + studio(w) / 3; acc["rev"] = acc["rev"] + studio(np.ascontiguousarray(w[::-1])) / 3
    print("studio depth", off, "done", flush=True)
np.savez_compressed(os.path.join(D_, "p_studio.npz"), fwd=acc["fwd"].astype(np.float16), rev=acc["rev"].astype(np.float16))
S = json.load(open(os.path.join(D_, "summary.json"))); rd = S.get("argus-v3", {}).get("read", "rev")
pg, ct = acc[rd], acc["fwd" if rd == "rev" else "rev"]
for tag, p in (("page", pg), ("ctl", ct)):
    fn = os.path.join(D_, f"studio_{tag}.png"); Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).save(fn)
    subprocess.run([sys.executable, "-u", "isoflat.py", os.path.join(D_, "mesh"), "16", fn, fn.replace(".png", "_iso.png")], capture_output=True)
S["studio"] = dict(read=rd, ink_page=float((pg[valid] > 0.5).mean()), ink_ctl=float((ct[valid] > 0.5).mean()))
json.dump(S, open(os.path.join(D_, "summary.json"), "w"), indent=1)
print("studio ink page", round(S["studio"]["ink_page"], 4), "ctl", round(S["studio"]["ink_ctl"], 4), flush=True)
