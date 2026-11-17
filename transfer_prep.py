"""Turn our pooled 2.4 um training segments (21 layers, level 2) into native-looking 9.362 um / 113 keV segments with the
scan transfer fitted by fit_transfer.py (runs/scan_transfer_0814.npz): canvas rescaled by the fitted factor, each of
the 28 native layers predicted from the 21 pooled layers by its KxK kernels, plus Gaussian noise of the residual sd,
mapped back to native intensities. Labels are resampled by nearest neighbour. Output: data/t<segment>/ in the segment
format, scroll key 't<scroll>' (train.SCROLL_OF maps the 't' prefix like 'n'). Validation regions are dropped.
usage: transfer_prep.py [--segments a,b] [--noise 1.0]"""
import argparse, json, os
import numpy as np, torch, zarr
from numcodecs import Blosc
from scipy.ndimage import binary_dilation

ROOT = os.environ.get("INK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--transfer", default=os.path.join(HERE, "runs", "scan_transfer_0814.npz"))
    a = ap.parse_args()
    T = np.load(a.transfer)
    kern = torch.tensor(T["kernel"].astype(np.float32))   # (28, 21, K, K)
    bias, sd, stats = T["bias"], T["resid_sd"], T["nat_stats"]
    sy, sx = (float(x) for x in T["scale"])
    K = kern.shape[-1]; r = K // 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kern = kern.to(dev)
    names = a.segments.split(",") if a.segments else sorted(
        d for d in os.listdir(ROOT) if d.startswith(("pherc", "phercparis")) and
        os.path.exists(os.path.join(ROOT, d, "done_tiles.npy")))
    rng = np.random.default_rng(0)
    for name in names:
        src = os.path.join(ROOT, name)
        out = os.path.join(ROOT, "t" + name)
        if os.path.exists(os.path.join(out, "done_tiles.npy")):
            continue
        vol = zarr.open(os.path.join(src, "volume.zarr"), mode="r")
        L = np.load(os.path.join(src, "labels.npz"))
        dp = os.path.join(src, "dense.npz")
        ink, sup = (np.load(dp)["ink"], np.load(dp)["sup"]) if os.path.exists(dp) else (L["ink"] > 0, L["sup"] > 0)
        if L["val"].size:
            sup = sup & ~(L["val"] > 0)
        Z, h, w = vol.shape
        if Z != 21:
            continue
        H, W = int(round(h * sy)), int(round(w * sx))
        yi = np.minimum((np.arange(H) / sy).astype(int), h - 1)
        xi = np.minimum((np.arange(W) / sx).astype(int), w - 1)
        sup_n = sup[yi][:, xi]; ink_n = (ink[yi][:, xi] > 0) & sup_n
        ov = zarr.open(os.path.join(out, "volume.zarr"), mode="w", shape=(28, H, W), chunks=(28, 128, 128), dtype="u1",
                       compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE), fill_value=0)
        tiles = np.zeros((-(-H // 128), -(-W // 128)), bool)
        ys, xs = np.nonzero(sup_n[::8, ::8]); tiles[ys * 8 // 128, xs * 8 // 128] = True
        tiles = binary_dilation(tiles, np.ones((3, 3), bool))
        big = np.zeros((-(-H // TILE), -(-W // TILE)), bool)
        ty, tx = np.nonzero(tiles); big[ty * 128 // TILE, tx * 128 // TILE] = True
        for by, bx in zip(*np.nonzero(big)):
            Y0, X0 = by * TILE, bx * TILE
            Y1, X1 = min(Y0 + TILE, H), min(X0 + TILE, W)
            py = np.minimum(((np.arange(Y0 - r, Y1 + r)) / sy).clip(0).astype(int), h - 1)
            px = np.minimum(((np.arange(X0 - r, X1 + r)) / sx).clip(0).astype(int), w - 1)
            y0, y1, x0, x1 = py.min(), py.max() + 1, px.min(), px.max() + 1
            blk = np.asarray(vol[:, y0:y1, x0:x1]).astype(np.float32)[:, py - y0][:, :, px - x0]   # (21, h', w')
            pap = blk[10] > 0
            if pap.sum() < 100:
                continue
            m = blk[:, pap].mean(1, keepdims=True)[:, :, None]; s = blk[:, pap].std(1, keepdims=True)[:, :, None] + 1e-6
            zt = torch.tensor((blk - m) / s, device=dev)[None]
            with torch.no_grad():
                o = torch.nn.functional.conv2d(zt, kern)[0].cpu().numpy()        # (28, Y1-Y0, X1-X0)
            o += bias[:, None, None] + a.noise * sd[:, None, None] * rng.standard_normal(o.shape).astype(np.float32)
            o = o * stats[:, 1, None, None] + stats[:, 0, None, None]
            o[:, ~pap[r:-r, r:-r]] = 0
            ov[:, Y0:Y1, X0:X1] = np.clip(o, 0, 255).astype(np.uint8)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "labels.npz"), ink=ink_n.astype(np.uint8), sup=sup_n.astype(np.uint8),
                            val=np.zeros(0, np.uint8), attrs=json.dumps({"source": src, "transfer": os.path.basename(a.transfer)}))
        json.dump(dict(segment="t" + name, shape=[28, H, W], ink_px=int(ink_n.sum()), sup_px=int(sup_n.sum())),
                  open(os.path.join(out, "info.json"), "w"), indent=1)
        np.save(os.path.join(out, "done_tiles.npy"), tiles)
        print("t" + name, H, W, int(sup_n.sum()), flush=True)


if __name__ == "__main__":
    main()
