"""Our own ink-detection trainer for the Vesuvius Challenge 9 um corpus (data from prep.py).
Model: a small 3D stem reads a 17-slice window of the surface volume, projects depth away, then a 2D U-Net predicts
per-pixel ink. Every training choice is a command-line option so many variants can run in parallel on rented GPUs.
Validation uses the three official online-validation segments (pherc0139-w016, pherc0814-46527, pherc1667-w029):
their _validation_mask pixels are never trained on.
usage: python train.py --out runs/base [--iters 20000 --lr 0.01 ...]"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

ROOT = os.environ.get("INK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
SCROLL_OF = lambda seg: ("0139" if seg.startswith(("pherc0139", "w0")) else "1667" if seg.startswith("pherc1667")
                         else "Paris4" if seg.startswith("phercparis4") else "0814" if seg.startswith("pherc0814") else seg)
VAL_SEGMENTS = ("pherc0139-w016", "pherc0814-46527", "pherc1667-w029")


def opts():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optim", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--wd", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--depth", type=int, default=17, help="Z window read by the model")
    p.add_argument("--z_jitter", type=int, default=2, help="max random shift of the Z window around the centre")
    p.add_argument("--width", type=int, default=32, help="first U-Net level channels")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--stem", type=int, default=16)
    p.add_argument("--zproj", choices=["meanmax", "conv", "attn"], default="meanmax")
    p.add_argument("--stem_blocks", type=int, default=0, help="extra 3D residual blocks in the stem (Kaggle 2023 winners: 3D ResNets)")
    p.add_argument("--mosaic", type=float, default=0.0, help="probability of a 2x2 mosaic of crops from different places")
    p.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="flips / rotations averaged at validation")
    p.add_argument("--arch", choices=["ours", "official"], default="ours", help="official = re-implemented ink_9um network")
    p.add_argument("--labels", choices=["manual", "dense"], default="manual",
                   help="dense = domenicor046/ink9um-dense-labels where available (7 segments), manual elsewhere")
    p.add_argument("--pseudo", type=float, default=0.0, help="share of patches drawn from teacher pseudo-labels (0 = off)")
    p.add_argument("--pseudo_w", type=float, default=0.5, help="loss weight of pseudo-labeled pixels")
    p.add_argument("--init", default="", help="start from these weights (official .pth from scrollprize/ink_9um, or our .pt)")
    p.add_argument("--smooth", type=float, default=0.5, help="BCE label smoothing (0.5 = reference recipe)")
    p.add_argument("--dice_w", type=float, default=1.0)
    p.add_argument("--aug", type=float, default=1.0, help="augmentation strength 0..2")
    p.add_argument("--zflip", type=float, default=-1, help="prob. of reversing depth order in training (-1: 0.3*aug, "
                   "the old default). 0 keeps the physical direction, so ink must sit on the written face")
    p.add_argument("--mix", default="0139:29,1667:22,Paris4:11,0814:2", help="patches per scroll per 64")
    p.add_argument("--exclude_scroll", default="", help="hold a whole scroll out of training (generalization test)")
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--amp", default="bf16")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------- data
class Segment:
    LABELS = "manual"

    def __init__(self, name):
        d = os.path.join(ROOT, name)
        L = np.load(os.path.join(d, "labels.npz"))
        self.name, self.scroll = name, SCROLL_OF(name)
        self.vol = zarr.open(os.path.join(d, "volume.zarr"), mode="r")
        self.ink = (L["ink"] > 0)
        self.sup = (L["sup"] > 0)
        self.val = (L["val"] > 0) if L["val"].size else np.zeros_like(self.sup)
        self.ink_eval = self.ink                                # the exam always uses the experts' labels
        dp = os.path.join(d, "dense.npz")
        if Segment.LABELS == "dense" and os.path.exists(dp):
            Dn = np.load(dp)
            self.ink, self.sup = Dn["ink"], Dn["sup"]          # validation mask stays the manual one
        self.train_mask = self.sup & ~self.val
        self.D = self.vol.shape[0]
        self.H, self.W = self.sup.shape
        ys, xs = np.nonzero(self.train_mask[::8, ::8])        # candidate centres on an 8 px grid
        self.centres = np.stack([ys * 8, xs * 8], 1) if len(ys) else np.zeros((0, 2), int)
        self.pseudo = None
        pp = os.path.join(d, "pseudo.npz")
        if os.path.exists(pp):
            P = np.load(pp)
            pm = P["mask"] & ~self.sup & ~self.val
            ys, xs = np.nonzero(pm[::8, ::8])
            if len(ys):
                self.pseudo = (P["prob"], pm)
                self.pcentres = np.stack([ys * 8, xs * 8], 1)

    def crop_pseudo(self, y, x, P, z0, depth, w):
        """crop with teacher soft labels on pseudo pixels (weight w) and true labels on supervised pixels"""
        img, lab, msk = self.crop(y, x, P, z0, depth)
        prob, pm = self.pseudo
        y0, x0 = y - P // 2, x - P // 2
        sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + P, self.H), min(x0 + P, self.W)
        if sy1 > sy0 and sx1 > sx0:
            pl = prob[sy0:sy1, sx0:sx1].astype(np.float32) / 255
            pmk = pm[sy0:sy1, sx0:sx1]
            sl = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0))
            lab[sl] = np.where(pmk, pl, lab[sl])
            msk[sl] = np.where(pmk, w, msk[sl])
        return img, lab, msk

    def crop(self, y, x, P, z0, depth):
        y0, x0 = y - P // 2, x - P // 2
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y0 + P, self.H), min(x0 + P, self.W)
        img = np.zeros((depth, P, P), np.uint8)
        lab = np.zeros((P, P), np.float32)
        msk = np.zeros((P, P), np.float32)
        if sy1 > sy0 and sx1 > sx0:
            img[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.vol[z0:z0 + depth, sy0:sy1, sx0:sx1]
            lab[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.ink[sy0:sy1, sx0:sx1]
            msk[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.train_mask[sy0:sy1, sx0:sx1]
        return img, lab, msk


def robust_norm(x):
    """per-patch robust normalisation (1-99 % clip, median / MAD), applied on the GPU-free worker side"""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-3
    return ((x - med) / mad).astype(np.float32)


class TrainSet(torch.utils.data.IterableDataset):
    def __init__(self, segs, a):
        self.segs, self.a = segs, a
        mix = dict(kv.split(":") for kv in a.mix.split(","))
        self.by_scroll = {}
        for s in segs:
            if len(s.centres):
                self.by_scroll.setdefault(s.scroll, []).append(s)
        tot = sum(float(mix.get(k, 0)) for k in self.by_scroll)
        self.prob = {k: float(mix.get(k, 0)) / tot for k in self.by_scroll}

    def __iter__(self):
        a = self.a
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(a.seed * 1000 + (wi.id if wi else 0) + int(time.time()) % 1000)
        keys = list(self.prob)
        p = np.array([self.prob[k] for k in keys])
        while True:
            sc = keys[rng.choice(len(keys), p=p)]
            segs = self.by_scroll[sc]
            s = segs[rng.integers(len(segs))]
            cy, cx = s.centres[rng.integers(len(s.centres))]
            cy += rng.integers(-4, 5); cx += rng.integers(-4, 5)
            centre = (s.D - a.depth) // 2
            z0 = int(np.clip(centre + rng.integers(-a.z_jitter, a.z_jitter + 1), 0, s.D - a.depth))
            if a.mosaic and rng.random() < a.mosaic:
                h = a.patch // 2
                img = np.zeros((a.depth, a.patch, a.patch), np.uint8)
                lab = np.zeros((a.patch, a.patch), np.float32); msk = np.zeros_like(lab)
                for qy in (0, 1):
                    for qx in (0, 1):
                        s2 = self.by_scroll[keys[rng.choice(len(keys), p=p)]]
                        s2 = s2[rng.integers(len(s2))]
                        c2y, c2x = s2.centres[rng.integers(len(s2.centres))]
                        zz = int(np.clip((s2.D - a.depth) // 2 + rng.integers(-a.z_jitter, a.z_jitter + 1), 0, s2.D - a.depth))
                        i2, l2, m2 = s2.crop(int(c2y), int(c2x), h, zz, a.depth)
                        img[:, qy * h:(qy + 1) * h, qx * h:(qx + 1) * h] = i2
                        lab[qy * h:(qy + 1) * h, qx * h:(qx + 1) * h] = l2
                        msk[qy * h:(qy + 1) * h, qx * h:(qx + 1) * h] = m2
            elif a.pseudo and rng.random() < a.pseudo:
                ps = [q for q in self.segs if q.pseudo is not None]
                if not ps:
                    continue
                q = ps[rng.integers(len(ps))]
                py, px = q.pcentres[rng.integers(len(q.pcentres))]
                zq = int(np.clip((q.D - a.depth) // 2 + rng.integers(-a.z_jitter, a.z_jitter + 1), 0, q.D - a.depth))
                img, lab, msk = q.crop_pseudo(int(py), int(px), a.patch, zq, a.depth, a.pseudo_w)
            else:
                img, lab, msk = s.crop(int(cy), int(cx), a.patch, z0, a.depth)
            if msk.sum() < 16:
                continue
            img = robust_norm(img)
            img, lab, msk = augment(img, lab, msk, rng, a.aug, a.zflip)
            yield torch.from_numpy(img[None].copy()), torch.from_numpy(lab[None].copy()), torch.from_numpy(msk[None].copy())


def augment(img, lab, msk, rng, s, zflip=-1):
    if s <= 0:
        return img, lab, msk
    k = rng.integers(4)
    img, lab, msk = np.rot90(img, k, (1, 2)), np.rot90(lab, k), np.rot90(msk, k)
    if rng.random() < 0.5:
        img, lab, msk = img[:, :, ::-1], lab[:, ::-1], msk[:, ::-1]
    if rng.random() < (0.3 * s if zflip < 0 else zflip):
        img = img[::-1]                                       # papyrus seen from the other side
    if rng.random() < 0.8:
        img = img * (1 + 0.15 * s * rng.standard_normal()) + 0.15 * s * rng.standard_normal()
    if rng.random() < 0.5:
        img = img + rng.standard_normal(img.shape).astype(np.float32) * 0.1 * s
    if rng.random() < 0.3 * s:                                # simulated z-blur / layer mixing
        w = rng.uniform(0.1, 0.4)
        img = (1 - w) * img + w * 0.5 * (np.roll(img, 1, 0) + np.roll(img, -1, 0))
    if rng.random() < 0.2 * s:                                # cutout
        P = img.shape[1]; c = rng.integers(8, P // 3)
        y, x = rng.integers(0, P - c, 2)
        img = img.copy(); img[:, y:y + c, x:x + c] = 0
    return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(lab), np.ascontiguousarray(msk)


def shm_free_gb():
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return 99.0          # Windows: no /dev/shm limit


def make_loader(ds, a):
    """worker processes when shared memory allows it, else threads (containers often get a 64 MB /dev/shm)"""
    if a.workers and shm_free_gb() >= 2:
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4)
        return iter(dl)
    import queue, threading
    q = queue.Queue(maxsize=8)
    n = max(a.workers, 1)

    class Info:
        def __init__(self, i): self.id = i

    def work(i):
        torch.utils.data.get_worker_info = lambda: Info(i)      # distinct random streams per thread
        g = iter(ds)
        while True:
            items = [next(g) for _ in range(a.batch)]
            q.put(tuple(torch.stack(t) for t in zip(*items)))
    for i in range(n):
        threading.Thread(target=work, args=(i,), daemon=True).start()
    print(f"data loader: {n} threads (shared memory {shm_free_gb():.2f} GB)", flush=True)

    def gen():
        while True:
            yield q.get()
    return gen()


# ---------------------------------------------------------------------------------------------------- model
def conv2(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.01, True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.01, True))


class Res3d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv3d(c, c, 3, padding=1, bias=False), nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, True),
                               nn.Conv3d(c, c, 3, padding=1, bias=False), nn.InstanceNorm3d(c, affine=True))
        self.act = nn.LeakyReLU(0.01, True)

    def forward(self, x):
        return self.act(x + self.b(x))


class InkNet(nn.Module):
    def __init__(self, depth=17, stem=16, width=32, levels=5, zproj="meanmax", stem_blocks=0):
        super().__init__()
        self.zproj = zproj
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem, 3, padding=1, bias=False), nn.InstanceNorm3d(stem, affine=True), nn.LeakyReLU(0.01, True),
            nn.Conv3d(stem, stem, 3, padding=1, bias=False), nn.InstanceNorm3d(stem, affine=True), nn.LeakyReLU(0.01, True),
            *[Res3d(stem) for _ in range(stem_blocks)])
        if zproj == "conv":
            self.proj = nn.Conv3d(stem, 2 * stem, (depth, 1, 1))
        elif zproj == "attn":
            self.attn = nn.Conv3d(stem, 1, 1)
        cin = 2 * stem
        ch = [min(width * 2 ** i, 320) for i in range(levels)]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = cin
        for i, co in enumerate(ch):
            self.enc.append(conv2(c, co))
            if i < levels - 1:
                self.down.append(nn.Conv2d(co, co, 2, stride=2))
            c = co
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 2, -1, -1):
            self.up.append(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2))
            self.dec.append(conv2(2 * ch[i], ch[i]))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):                          # x: B,1,D,H,W
        f = self.stem(x)
        if self.zproj == "conv":
            f = self.proj(f).squeeze(2)
        elif self.zproj == "attn":
            w = torch.softmax(self.attn(f), dim=2)
            f = torch.cat([(f * w).sum(2), f.amax(2)], 1)
        else:
            f = torch.cat([f.mean(2), f.amax(2)], 1)
        skips = []
        for i, e in enumerate(self.enc):
            f = e(f)
            if i < len(self.down):
                skips.append(f)
                f = self.down[i](f)
        for u, d in zip(self.up, self.dec):
            f = u(f)
            f = d(torch.cat([f, skips.pop()], 1))
        return self.head(f)


def loss_fn(logit, lab, msk, smooth, dice_w):
    tgt = lab * (1 - smooth) + 0.5 * smooth
    bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
    bce = (bce * msk).sum() / msk.sum().clamp(min=1)
    p = torch.sigmoid(logit) * msk
    inter = (p * lab).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (lab * msk).sum() + 1)
    return bce + dice_w * dice


# ---------------------------------------------------------------------------------------------------- validation
@torch.no_grad()
def predict(model, s, a, dev, region=None, stride=None):
    """sliding-window probability map over the bounding box of `region` (bool mask), Hann-blended"""
    P = a.patch
    stride = stride or P // 2
    m = region if region is not None else s.sup
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - P // 2, 0), min(ys.max() + P // 2, s.H)
    x0, x1 = max(xs.min() - P // 2, 0), min(xs.max() + P // 2, s.W)
    z0 = (s.D - a.depth) // 2
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    wsum = np.zeros_like(acc)
    win = np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1]).astype(np.float32)
    pos = [(y, x) for y in range(y0, max(y1 - P, y0) + 1, stride) for x in range(x0, max(x1 - P, x0) + 1, stride)]
    for i in range(0, len(pos), a.batch):
        batch = []
        for y, x in pos[i:i + a.batch]:
            img, _, _ = s.crop(y + P // 2, x + P // 2, P, z0, a.depth)
            batch.append(robust_norm(img)[None])
        xb = torch.from_numpy(np.stack(batch)).to(dev)
        tta = getattr(a, "tta", 1)
        acc_p = 0
        for t in range(tta):                         # t: bit0 = horizontal flip, bits 1-2 = number of 90 degree turns
            k, fl = t >> 1, t & 1
            xt = torch.rot90(xb, k, (3, 4))
            if fl:
                xt = torch.flip(xt, (4,))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                pt = torch.sigmoid(model(xt).float())
            if fl:
                pt = torch.flip(pt, (3,))
            acc_p = acc_p + torch.rot90(pt, -k, (2, 3))
        pr = (acc_p / tta).cpu().numpy()[:, 0]
        for (y, x), p in zip(pos[i:i + a.batch], pr):
            hh, ww = min(P, y1 - y), min(P, x1 - x)
            acc[y - y0:y - y0 + hh, x - x0:x - x0 + ww] += p[:hh, :ww] * win[:hh, :ww]
            wsum[y - y0:y - y0 + hh, x - x0:x - x0 + ww] += win[:hh, :ww]
    out = np.zeros((s.H, s.W), np.float32)
    out[y0:y1, x0:x1] = acc / np.maximum(wsum, 1e-6)
    return out


def metrics(prob, lab, region, smooth):
    p = prob[region]
    if smooth:                                    # smoothed targets live in [s/2, 1 - s/2]: map back to [0, 1]
        p = (p - smooth / 2) / max(1 - smooth, 1e-6)
    t = lab[region]
    out = {}
    for thr in (0.3, 0.4, 0.5, 0.6):
        b = p >= thr
        tp = (b & t).sum(); fp = (b & ~t).sum(); fn = (~b & t).sum(); tn = (~b & ~t).sum()
        out[f"dice@{thr}"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        out[f"bacc@{thr}"] = float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
    # threshold-free: rank AUC on a sample
    rng = np.random.default_rng(0)
    idx = rng.choice(len(p), min(len(p), 400000), replace=False)
    ps, ts = p[idx], t[idx]
    order = np.argsort(ps)
    r = np.empty(len(ps)); r[order] = np.arange(1, len(ps) + 1)
    npos, nneg = ts.sum(), (~ts).sum()
    out["auc"] = float((r[ts].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
    out["dice_best"] = max(v for k, v in out.items() if k.startswith("dice@"))
    return out


def validate(model, val_segs, a, dev):
    model.eval()
    res = {}
    for s in val_segs:
        prob = predict(model, s, a, dev, region=s.val)
        res[s.name] = metrics(prob, s.ink_eval, s.val, a.smooth)
    model.train()
    keys = ["auc", "dice_best", "bacc@0.5"]
    res["mean"] = {k: float(np.mean([res[n][k] for n in res])) for k in keys}
    return res


# ---------------------------------------------------------------------------------------------------- main
def main():
    a = opts()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    names = sorted(d for d in os.listdir(ROOT) if os.path.exists(os.path.join(ROOT, d, "done_tiles.npy")))
    Segment.LABELS = a.labels
    segs = [Segment(n) for n in names]
    excl = set(a.exclude_scroll.split(",")) - {""}
    train_segs = [s for s in segs if s.scroll not in excl]
    val_segs = [s for s in segs if s.name in VAL_SEGMENTS and s.val.any()]
    print(f"{len(train_segs)} training segments, validation on {[s.name for s in val_segs]}", flush=True)
    dev = torch.device("cuda")
    if a.arch == "official":
        import official
        model = official.OfficialInkNet("attn_max")
    else:
        model = InkNet(a.depth, a.stem, a.width, a.levels, a.zproj, a.stem_blocks)
    if a.init:
        if a.init.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = {k: v.float() for k, v in load_file(a.init).items()}
        else:
            sd = torch.load(a.init, map_location="cpu", weights_only=True)
            sd = sd["model"] if "model" in sd else sd
        sd = {k: v for k, v in sd.items() if ".all_modules." not in k}
        print("init from", a.init, model.load_state_dict(sd, strict=True), flush=True)
    model = model.to(dev)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    if a.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.99, nesterov=True, weight_decay=a.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(i / a.iters, 1))))
    it = make_loader(TrainSet(train_segs, a), a)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best = -1
    t0 = time.time()
    for step in range(1, a.iters + 1):
        x, y, m = (t.to(dev, non_blocking=True) for t in next(it))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
            logit = model(x)
        loss = loss_fn(logit.float(), y, m, a.smooth, a.dice_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0], "s_per_100": round(time.time() - t0, 1)}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); t0 = time.time()
        if step % a.val_every == 0 or step == a.iters:
            v = validate(model, val_segs, a, dev)
            rec = {"step": step, "val": v}
            print({"step": step, **v["mean"]}, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "args": vars(a), "step": step, "val": v}, os.path.join(a.out, "last.pt"))
            if v["mean"]["auc"] > best:
                best = v["mean"]["auc"]
                torch.save({"model": model.state_dict(), "args": vars(a), "step": step, "val": v}, os.path.join(a.out, "best.pt"))
    json.dump({"best_auc": best}, open(os.path.join(a.out, "done.json"), "w"))


if __name__ == "__main__":
    main()
