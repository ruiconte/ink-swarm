"""Interprete: letter recogniser for ink-map candidates, trained on ~215k real Greek papyrus letters (AL-PUB, HomerComp).
Clean letter masks are degraded on the GPU into what an ink model's probability map looks like (blurred strokes,
broken / thickened strokes, blob noise, partial erasure, rotation, scale, shift), plus a 25th class "not a letter"
(blobs, fibre lines, cracks, ring noise). Validation = papyri never seen in training.
usage: interprete.py train            -> runs/interprete.pt
       (library) Interprete().predict(crops_64x64_float) -> probs (N, 25)"""
import math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡϹΤΥΦΧΨΩ") + ["non-lettre"]
ALPUB = dict(Alpha="Α", Beta="Β", Gamma="Γ", Delta="Δ", Epsilon="Ε", Zeta="Ζ", Eta="Η", Theta="Θ", Iota="Ι", Kappa="Κ",
             Lambda="Λ", Mu="Μ", Nu="Ν", Xi="Ξ", Omicron="Ο", Pi="Π", Rho="Ρ", LunateSigma="Ϲ", Tau="Τ", Upsilon="Υ",
             Phi="Φ", Chi="Χ", Psi="Ψ", Omega="Ω")
NONE = len(CLASSES) - 1
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"


def load():
    xs, ys, ps = [], [], []
    z = np.load(os.path.join(HERE, "hunt/letters_alpub.npz"))
    lab = [CLASSES.index(ALPUB[l]) if l in ALPUB else -1 for l in z["labels"]]
    xs.append(z["shapes"]); ys.append(np.array(lab)); ps.append(np.array(["A" + s for s in z["src"]]))
    z = np.load(os.path.join(HERE, "hunt/letters_homercomp.npz"))
    lab = [CLASSES.index(l.upper()) if l.upper() in CLASSES else -1 for l in z["labels"]]
    xs.append(z["shapes"]); ys.append(np.array(lab)); ps.append(np.array(["H" + s for s in z["src"]]))
    x, y, p = np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)
    k = y >= 0
    return x[k], y[k], p[k]


def gblur(x, sigma):
    """per-sample gaussian blur, x (B,1,H,W), sigma (B,)"""
    r = 9; t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t[None] ** 2) / (2 * sigma[:, None] ** 2)); k = k / k.sum(1, keepdim=True)          # (B, 2r+1)
    B = x.shape[0]; xx = x.reshape(1, B, *x.shape[2:])
    xx = F.conv2d(F.pad(xx, (r, r, 0, 0), mode="reflect"), k[:, None, None, :], groups=B)
    xx = F.conv2d(F.pad(xx, (0, 0, r, r), mode="reflect"), k[:, None, :, None], groups=B)
    return xx.reshape_as(x)


def smooth_noise(B, S, scale, dev):
    n = torch.randn(B, 1, S // scale + 2, S // scale + 2, device=dev)
    n = F.interpolate(n, size=(S, S), mode="bicubic", align_corners=False)
    return n / (n.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


def warp(x, strength=1.0):
    B = x.shape[0]; dev = x.device
    ang = (torch.rand(B, device=dev) - 0.5) * math.radians(40) * strength
    sc = 1 / (0.7 + 0.45 * torch.rand(B, device=dev))
    sh = (torch.rand(B, device=dev) - 0.5) * 0.3 * strength
    tx, ty = ((torch.rand(B, 2, device=dev) - 0.5) * 0.3 * strength).unbind(1)
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    th = torch.stack([torch.stack([c, -s + sh, tx], 1), torch.stack([s, c, ty], 1)], 1)
    g = F.affine_grid(th, x.shape, align_corners=False)
    g = g + 0.04 * strength * torch.stack([smooth_noise(B, x.shape[-1], 16, dev)[:, 0], smooth_noise(B, x.shape[-1], 16, dev)[:, 0]], -1)
    return F.grid_sample(x, g, align_corners=False, padding_mode="zeros")


PHYS = os.environ.get("INTERP_PHYS", "1") == "1"


def squash(x):
    """crushing of the scroll: independent horizontal / vertical scale and a shear"""
    B, dev = x.shape[0], x.device
    sx, sy = 0.75 + 0.5 * torch.rand(B, device=dev), 0.75 + 0.5 * torch.rand(B, device=dev)
    sh = (torch.rand(B, device=dev) - 0.5) * 0.4
    z = torch.zeros(B, device=dev)
    th = torch.stack([torch.stack([1 / sx, sh, z], 1), torch.stack([z, 1 / sy, z], 1)], 1)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)


def fibre_gaps(x):
    """ink lost along papyrus fibres: horizontal bands (recto fibres) and weaker vertical ones, 2-6 px (0.1-0.3 mm)"""
    B, dev, S = x.shape[0], x.device, x.shape[-1]
    w = int(torch.randint(2, 6, (1,)))
    rows = F.interpolate(torch.rand(B, 1, S // w + 1, 1, device=dev), size=(S, 1), mode="nearest")
    cols = F.interpolate(torch.rand(B, 1, 1, S // w + 1, device=dev), size=(1, S), mode="nearest")
    p_row, p_col = 0.2 * torch.rand(B, 1, 1, 1, device=dev), 0.06 * torch.rand(B, 1, 1, 1, device=dev)
    return x * (rows > p_row).float() * (cols > p_col).float()


def crack(x):
    """a crack across the letter: a thin empty line, and one side shifted by 1-4 px along it"""
    B, dev, S = x.shape[0], x.device, x.shape[-1]
    ang = torch.rand(B, device=dev) * math.pi; n = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    off = (torch.rand(B, device=dev) - 0.5) * 0.8
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, S, device=dev), torch.linspace(-1, 1, S, device=dev), indexing="ij")
    d = xx[None] * n[:, 0, None, None] + yy[None] * n[:, 1, None, None] - off[:, None, None]
    side = (d > 0).float()[:, None]
    shift = (1 + 3 * torch.rand(B, device=dev)) * 2 / S
    th = torch.zeros(B, 2, 3, device=dev); th[:, 0, 0] = 1; th[:, 1, 1] = 1
    th[:, 0, 2] = -n[:, 1] * shift; th[:, 1, 2] = n[:, 0] * shift                         # move along the crack
    moved = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    gap = (d.abs() > (1 + torch.rand(B, 1, 1, device=dev) * 2) / S).float()[:, None]
    out = (x * (1 - side) + moved * side) * gap
    use = (torch.rand(B, 1, 1, 1, device=dev) < 0.35).float()
    return use * out + (1 - use) * x


def degrade(m):
    """m: (B,1,64,64) float masks in {0,1} -> ink-map-like probability images"""
    B, dev = m.shape[0], m.device
    if PHYS:
        m = crack(fibre_gaps(squash(m)))
    x = warp(m)
    # stroke width change: blur + rethreshold
    x = (gblur(x, 0.6 + 1.5 * torch.rand(B, device=dev)) > (0.25 + 0.5 * torch.rand(B, 1, 1, 1, device=dev))).float()
    # broken strokes: multiply by a low-frequency keep mask
    keep = (smooth_noise(B, 64, 8, dev) > (-1.2 + 1.6 * torch.rand(B, 1, 1, 1, device=dev) * (torch.rand(B, 1, 1, 1, device=dev) < (0.3 if PHYS else 0.6)))).float()
    x = x * keep
    # partial erasure of a band
    er = torch.rand(B, device=dev) < 0.25
    if er.any():
        yy = torch.arange(64, device=dev)[None, :, None].float(); a = torch.rand(B, 1, 1, device=dev) * 64
        band = ((yy - a).abs() < 8).float()[:, None]
        x = torch.where(er[:, None, None, None], x * (1 - band), x)
    # ink-model look: blurred, soft, with blob noise
    x = gblur(x, 1.0 + 3.0 * torch.rand(B, device=dev))
    x = x / (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-6) * (0.5 + 0.5 * torch.rand(B, 1, 1, 1, device=dev))
    blobs = gblur((torch.rand(B, 1, 64, 64, device=dev) < 0.004 * torch.rand(B, 1, 1, 1, device=dev)).float(), 2 + 2 * torch.rand(B, device=dev)) * 12
    x = x + blobs * (torch.rand(B, 1, 1, 1, device=dev) * 0.8) + 0.15 * torch.rand(B, 1, 1, 1, device=dev) * smooth_noise(B, 64, 4, dev).relu()
    return x.clamp(0, 1)


def negatives(B, dev):
    """things an ink map shows that are not letters"""
    kind = torch.randint(0, 4, (B,), device=dev)
    blob = (smooth_noise(B, 64, 6 + int(torch.randint(0, 10, (1,))), dev) > 1.0).float()                       # blobs
    yy, xx = torch.meshgrid(torch.arange(64, device=dev).float(), torch.arange(64, device=dev).float(), indexing="ij")
    ph = torch.rand(B, 1, 1, device=dev) * 6.28; fr = 0.15 + 0.3 * torch.rand(B, 1, 1, device=dev)
    lines = ((torch.sin(fr * (yy if True else xx)[None] + ph) > 0.85).float()[:, None]) * (smooth_noise(B, 64, 16, dev) > -0.3).float()  # fibres
    lines = torch.where((torch.rand(B, 1, 1, 1, device=dev) < 0.5), lines, lines.transpose(2, 3))
    crack = torch.zeros(B, 1, 64, 64, device=dev)
    t = torch.linspace(0, 1, 200, device=dev)
    p0 = torch.rand(B, 2, device=dev) * 64; d = torch.randn(B, 2, device=dev); d = d / d.norm(dim=1, keepdim=True)
    wig = torch.cumsum(torch.randn(B, 200, device=dev) * 0.4, 1)
    py = (p0[:, :1] + 70 * t * d[:, :1] + wig * d[:, 1:]).long().clamp(0, 63); px = (p0[:, 1:] + 70 * t * d[:, 1:] - wig * d[:, :1]).long().clamp(0, 63)
    crack.view(B, -1).scatter_(1, py * 64 + px, 1.0)                                                         # cracks
    noise = (smooth_noise(B, 64, 3, dev) > 1.3).float()                                                       # speckle
    x = torch.stack([blob, lines, crack, noise], 0)[kind, torch.arange(B, device=dev)]
    return degrade(x)


class Net(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(1, 32), nn.MaxPool2d(2), blk(32, 64), nn.MaxPool2d(2), blk(64, 128), nn.MaxPool2d(2), blk(128, 256),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, n))

    def forward(self, x):
        return self.f(x)


def train(epochs=8, bs=512):
    x, y, p = load()
    papyri = np.unique(p); rng = np.random.default_rng(0); val_p = set(rng.choice(papyri, len(papyri) // 20, replace=False))
    va = np.array([q in val_p for q in p])
    X = torch.from_numpy(x).to(DEV); Y = torch.from_numpy(y).long().to(DEV)
    tr_idx, va_idx = torch.from_numpy(np.nonzero(~va)[0]).to(DEV), torch.from_numpy(np.nonzero(va)[0][:8000]).to(DEV)
    print(len(tr_idx), "train letters,", len(va_idx), "val letters from", len(val_p), "unseen papyri", flush=True)
    cnt = torch.bincount(Y[tr_idx], minlength=len(CLASSES)).float(); w = (cnt.sum() / (cnt + 1)).clamp(max=50) ** 0.5; w[NONE] = 1.0
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * len(tr_idx) // bs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    g = torch.Generator(device=DEV); g.manual_seed(1)
    with torch.no_grad():
        Xv = degrade(X[va_idx][:, None].float()); Yv = Y[va_idx]
        Nv = negatives(2000, DEV)
    t0 = time.time()
    for s in range(steps):
        net.train()
        i = tr_idx[torch.randint(0, len(tr_idx), (bs,), device=DEV, generator=g)]
        with torch.no_grad():
            xb = degrade(X[i][:, None].float()); yb = Y[i]
            nb = bs // 8; xb = torch.cat([xb, negatives(nb, DEV)]); yb = torch.cat([yb, torch.full((nb,), NONE, device=DEV)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(xb), yb, weight=w, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if s % 400 == 0 or s == steps - 1:
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pv = net(Xv).float().argmax(1); pn = net(Nv).float().argmax(1)
            acc = (pv == Yv).float().mean().item(); top_letter = (pv != NONE).float().mean().item(); rej = (pn == NONE).float().mean().item()
            print(f"step {s}/{steps} loss {loss.item():.3f} val_acc(unseen papyri, degraded) {acc:.3f} letters_called_letters {top_letter:.3f} "
                  f"non-letters_rejected {rej:.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "classes": CLASSES}, os.path.join(HERE, "runs/interprete.pt"))
    print("saved runs/interprete.pt")


class Interprete:
    def __init__(self, path=os.path.join(HERE, "runs/interprete.pt"), dev=DEV):
        ck = torch.load(path, map_location=dev); self.net = Net().to(dev).eval(); self.net.load_state_dict(ck["model"]); self.dev = dev

    @torch.no_grad()
    def predict(self, crops):
        x = torch.as_tensor(np.asarray(crops, np.float32)).to(self.dev)[:, None]
        out = [F.softmax(self.net(x[k:k + 1024]).float(), 1) for k in range(0, len(x), 1024)]
        return torch.cat(out).cpu().numpy()


if __name__ == "__main__" and sys.argv[1:2] == ["train"]:
    train()
