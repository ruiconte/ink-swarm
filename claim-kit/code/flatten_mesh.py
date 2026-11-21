"""Write a FLATTENED tifxyz: same surface, new regular grid on which grid steps keep their 3D length (isometric).
Same optimisation as isoflat.py (every grid edge and diagonal keeps its 3D length, Adam on GPU), then the 3D points
are resampled on a regular grid of the flattened coordinates. Any image rendered on the output mesh is then
distance-true (1 render pixel = 1 voxel = 9.362 um in both directions).
usage: flatten_mesh.py MESH_IN STRIDE MESH_OUT"""
import sys, os, json, numpy as np, tifffile, torch
from scipy.interpolate import griddata
md, st, mo = sys.argv[1], int(sys.argv[2]), sys.argv[3]; os.makedirs(mo, exist_ok=True)
P = np.stack([tifffile.imread(f"{md}/{a}.tif").astype(np.float64) for a in "xyz"], -1); ok = P[..., 0] > 0
H, W = ok.shape; idx = -np.ones((H, W), int); idx[ok] = np.arange(ok.sum())
E = []
for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
    a = idx[0:H - dy, max(0, -dx):W - max(0, dx)]
    b = idx[dy:H, max(0, dx):W - max(0, -dx) if dx < 0 else W][:, :a.shape[1]] if dx >= 0 else idx[dy:H, 0:W + dx]
    m = (a >= 0) & (b >= 0); E.append(np.stack([a[m], b[m]], 1))
E = np.concatenate(E); X = P[ok]
L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / st
yy, xx = np.nonzero(ok); u0 = np.stack([xx, yy], 1).astype(np.float64)
dev = torch.device("cuda"); u = torch.tensor(u0, device=dev, requires_grad=True)
Et = torch.tensor(E, device=dev); Lt = torch.tensor(L, device=dev); opt = torch.optim.Adam([u], lr=0.05)
def err(q): return (torch.linalg.norm(q[Et[:, 0]] - q[Et[:, 1]], dim=1) - Lt) / Lt
with torch.no_grad(): e0 = err(u).abs()
for i in range(3000):
    opt.zero_grad(); loss = (err(u) ** 2).mean(); loss.backward(); opt.step()
un = u.detach().cpu().numpy(); un -= un.min(0)
gh, gw = int(np.ceil(un[:, 1].max())) + 1, int(np.ceil(un[:, 0].max())) + 1
gy, gx = np.mgrid[0:gh, 0:gw]
new = np.stack([griddata(un, X[:, k], (gx, gy), method="linear") for k in range(3)], -1)
bad = np.isnan(new).any(-1); new[bad] = -1
# check on the NEW grid: 3D length of each grid step / stride
def steps(A):
    d1 = np.linalg.norm(A[:, 1:] - A[:, :-1], axis=-1) / st; v1 = (A[:, 1:, 0] > 0) & (A[:, :-1, 0] > 0)
    d2 = np.linalg.norm(A[1:] - A[:-1], axis=-1) / st; v2 = (A[1:, :, 0] > 0) & (A[:-1, :, 0] > 0)
    return np.concatenate([d1[v1], d2[v2]])
s_old, s_new = steps(np.where(ok[..., None], P, -1)), steps(new)
for a, name in zip("xyz", range(3)): tifffile.imwrite(f"{mo}/{a}.tif", new[..., name].astype(np.float32))
meta = json.load(open(f"{md}/meta.json")); meta["uuid"] = meta.get("uuid", "surface") + "-flat"
meta["flattening"] = dict(method="isometric grid re-parametrisation (flatten_mesh.py)",
    step_len_over_stride_old_p5_p50_p95=[round(float(x), 3) for x in np.percentile(s_old, [5, 50, 95])],
    step_len_over_stride_new_p5_p50_p95=[round(float(x), 3) for x in np.percentile(s_new, [5, 50, 95])],
    edges_off_by_10pct_old=round(float((np.abs(s_old - 1) > 0.1).mean()), 4), edges_off_by_10pct_new=round(float((np.abs(s_new - 1) > 0.1).mean()), 4))
meta.pop("isometry", None); json.dump(meta, open(f"{mo}/meta.json", "w"), indent=1)
print(json.dumps(meta["flattening"]), "grid", (H, W), "->", (gh, gw))
