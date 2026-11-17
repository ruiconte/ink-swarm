"""Fit our own pooled(2.4 um level-2, 21 layers) -> native(9.362 um, 113 keV, 28 layers) scan transfer on the registered
pair PHerc0814 segment 46527 (same mesh rendered from both scans; never a test segment). Steps:
  1. resample the pooled render onto the native canvas (scale 2180/2130 x 3560/3455) and search the (dy, dx) shift
     that best correlates the middle layers;
  2. ridge-regress every native layer on all 21 pooled layers x a KxK neighbourhood (z-scored on papyrus), fitted on
     the left 70 % of the papyrus and scored (R^2) on the right 30 %;
  3. save kernels, bias, residual sd and the native intensity stats to runs/scan_transfer_0814.npz.
usage: fit_transfer.py [--k 5]"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, zarr
from scipy.ndimage import map_coordinates, shift as nd_shift
import prep

HERE = os.path.dirname(os.path.abspath(__file__))
NAT = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/"
       "surface-volumes/9.362um-1.2m-113keV-volume-20250804134230.zarr")


def native():
    za = prep.get(f"{NAT}/0/.zarray").json()
    Z, H, W = za["shape"]; cz, cy, cx = za["chunks"]; sep = za.get("dimension_separator", ".")
    v = np.zeros((Z, H, W), np.uint8)

    def one(k):
        iy, ix = k
        r = prep.get(f"{NAT}/0/0{sep}{iy}{sep}{ix}")
        if r is not None:
            a = np.frombuffer(r.content, np.uint8).reshape(cz, cy, cx)
            v[:, iy * cy:(iy + 1) * cy, ix * cx:(ix + 1) * cx] = a[:, :min(cy, H - iy * cy), :min(cx, W - ix * cx)]
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, [(y, x) for y in range(-(-H // cy)) for x in range(-(-W // cx))]))
    return v


def resample(p, H, W):
    """pooled (Z, h, w) -> (Z, H, W) by linear interpolation of the canvas scale"""
    Z, h, w = p.shape
    yy, xx = np.meshgrid(np.arange(H) * h / H, np.arange(W) * w / W, indexing="ij")
    return np.stack([map_coordinates(p[z].astype(np.float32), [yy, xx], order=1) for z in range(Z)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5); a = ap.parse_args()
    nat = native().astype(np.float32)
    pooled = np.asarray(zarr.open(os.path.join(prep.ROOT, "pherc0814-46527", "volume.zarr"), mode="r")).astype(np.float32)
    Z, H, W = nat.shape
    pr = resample(pooled, H, W)
    valid = (pr[pr.shape[0] // 2] > 0) & (nat[Z // 2] > 0)

    def z(x, m):
        return (x - x[m].mean()) / (x[m].std() + 1e-6)
    pm = z(pr[pr.shape[0] // 2], valid)
    best = None
    for dy in range(-14, 15, 2):
        for dx in range(-14, 15, 2):
            s = nd_shift(pm, (dy, dx), order=0)
            for j in range(Z):
                c = float(np.mean(s[valid] * z(nat[j], valid)[valid]))
                if best is None or c > best[0]:
                    best = (c, dy, dx, j)
    c, dy, dx, j = best
    print(f"best shift dy={dy} dx={dx}, pooled middle layer <-> native layer {j}, r={c:.3f}", flush=True)
    pr = np.stack([nd_shift(l, (dy, dx), order=1) for l in pr])
    valid &= np.stack([l > 0 for l in pr]).all(0)
    P = np.stack([z(l, valid) for l in pr])                      # (21, H, W)
    N = np.stack([z(l, valid) for l in nat])                     # (28, H, W)
    K = a.k; r = K // 2
    ys, xs = np.nonzero(valid[r:-r, r:-r]); ys += r; xs += r
    rng = np.random.default_rng(0); pick = rng.choice(len(ys), min(len(ys), 400000), replace=False)
    ys, xs = ys[pick], xs[pick]
    train = xs < np.quantile(xs, 0.7)
    feats = np.stack([P[:, ys + oy, xs + ox] for oy in range(-r, r + 1) for ox in range(-r, r + 1)], 2).transpose(1, 0, 2).reshape(len(ys), -1)
    X = np.concatenate([feats, np.ones((len(ys), 1), np.float32)], 1)
    lam = 1e-1 * len(ys)
    A = X[train].T @ X[train] + lam * np.eye(X.shape[1]); A[-1, -1] -= lam
    W_all, r2, sds = [], [], []
    for jn in range(Z):
        y = N[jn, ys, xs]
        w = np.linalg.solve(A, X[train].T @ y[train])
        res = y[~train] - X[~train] @ w
        r2.append(1 - res.var() / y[~train].var()); sds.append(float(res.std())); W_all.append(w)
    print("held-out R^2 per native layer:", np.round(r2, 3).tolist(), flush=True)
    kern = np.stack([w[:-1].reshape(-1, K, K) for w in W_all])   # (28, 21, K, K): pooled layer, dy, dx
    stats = [(float(nat[jn][valid].mean()), float(nat[jn][valid].std())) for jn in range(Z)]
    np.savez(os.path.join(HERE, "runs", "scan_transfer_0814.npz"), kernel=kern.astype(np.float32),
             bias=np.array([w[-1] for w in W_all], np.float32), resid_sd=np.array(sds, np.float32),
             nat_stats=np.array(stats, np.float32), scale=np.array([H / pooled.shape[1], W / pooled.shape[2]]),
             r2=np.array(r2, np.float32))
    json.dump(dict(shift=[dy, dx], layer_match=j, r=c, r2=[float(x) for x in r2]),
              open(os.path.join(HERE, "runs", "scan_transfer_0814.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
