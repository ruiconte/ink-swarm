"""Re-flatten a tifxyz surface so that distances on the sheet are kept (the papyrus does not stretch), like sail or
garment pattern flattening: 2D positions of the grid nodes are optimised so that every grid edge and diagonal keeps its
3D length (Adam on GPU), starting from the given grid. Then warps an image rendered on the old grid to the new one.
usage: isoflat.py MESH_DIR STRIDE IMG_IN IMG_OUT"""
import sys, numpy as np, tifffile, torch
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata
md, st, fin, fout = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
P = np.stack([tifffile.imread(f"{md}/{a}.tif").astype(np.float64) for a in "xyz"], -1); ok = P[..., 0] > 0
H, W = ok.shape; idx = -np.ones((H, W), int); idx[ok] = np.arange(ok.sum())
E = []
for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
    a = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
    b = idx[max(0, dy):H - max(0, -dy) if dy <= 0 else H, max(0, dx):W - max(0, -dx) if dx <= 0 else W]
    a = idx[0:H - dy, max(0, -dx):W - max(0, dx)]; b = idx[dy:H, max(0, dx):W - max(0, -dx) if dx < 0 else W][:, :a.shape[1]] if dx >= 0 else idx[dy:H, 0:W + dx]
    m = (a >= 0) & (b >= 0); E.append(np.stack([a[m], b[m]], 1))
E = np.concatenate(E); X = P[ok]
L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / st                     # edge length in grid units
yy, xx = np.nonzero(ok); u0 = np.stack([xx, yy], 1).astype(np.float64)
dev = torch.device("cuda"); u = torch.tensor(u0, device=dev, requires_grad=True)
Et = torch.tensor(E, device=dev); Lt = torch.tensor(L, device=dev)
opt = torch.optim.Adam([u], lr=0.05)
def err(uu): return (torch.linalg.norm(uu[Et[:, 0]] - uu[Et[:, 1]], dim=1) - Lt) / Lt
with torch.no_grad(): e0 = err(u).abs()
for i in range(3000):
    opt.zero_grad(); loss = (err(u) ** 2).mean(); loss.backward(); opt.step()
with torch.no_grad(): e1 = err(u).abs()
print(f"edges {len(E)}  length error >10%: before {(e0 > 0.1).float().mean():.1%} after {(e1 > 0.1).float().mean():.1%}  median {e0.median():.3f} -> {e1.median():.3f}")
un = u.detach().cpu().numpy(); un -= un.min(0)
# inverse map: for each output pixel (new coords), where was it on the old grid?
img = np.asarray(Image.open(fin), np.float32)
gh, gw = int(np.ceil(un[:, 1].max())) + 1, int(np.ceil(un[:, 0].max())) + 1
gy, gx = np.mgrid[0:gh, 0:gw]
oldx = griddata(un, u0[:, 0], (gx, gy), method="linear"); oldy = griddata(un, u0[:, 1], (gx, gy), method="linear")
s = img.shape[0] / H
fy = ndimage.zoom(np.nan_to_num(oldy, nan=-1), s, order=1) * s; fx = ndimage.zoom(np.nan_to_num(oldx, nan=-1), s, order=1) * s
inside = (fy >= 0) & (fx >= 0)
out = ndimage.map_coordinates(img, [fy, fx], order=1, cval=0); out[~inside] = 0
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(fout); print("saved", out.shape)
