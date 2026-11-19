"""Render a surface volume from a tifxyz mesh and a public OME-Zarr CT volume: the mesh grid is upsampled to full
resolution (stride from meta.json), normals are taken from the mesh, and the CT is sampled (trilinear) at
p + k * n for k in -R..R. Output: (2R+1, H, W) uint8 .npy, layer order along +n.
usage: render_tifxyz.py MESH_DIR ZARR_URL OUT.npy [R=8]"""
import json, os, sys, time
import numpy as np, zarr
from scipy import ndimage

mdir, url, out = sys.argv[1:4]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 8
import tifffile
xyz = np.stack([tifffile.imread(f"{mdir}/{a}.tif").astype(np.float32) for a in "xyz"])
st = json.load(open(f"{mdir}/meta.json")).get("stride", 16)
LV = int(os.environ.get("LEVEL", "0"))           # read a coarser pyramid level: mesh coordinates and stride shrink by 2**LV
if LV: xyz = np.where(xyz > 0, xyz / 2 ** LV, xyz); st = max(1, st // 2 ** LV)
valid = xyz[0] > 0
# fill holes for interpolation, then upsample the grid to one point per voxel
filled = np.stack([np.where(valid, c, ndimage.generic_filter(np.where(valid, c, np.nan), np.nanmean, 3, mode="nearest")) for c in xyz])
filled = np.nan_to_num(filled, nan=-1)
H, W = xyz.shape[1] * st, xyz.shape[2] * st
up = np.stack([ndimage.zoom(c, st, order=1) for c in filled])[:, :H, :W]
vup = ndimage.zoom(valid.astype(np.float32), st, order=0)[:H, :W] > 0.5
gy = np.stack(np.gradient(up, axis=1)); gx = np.stack(np.gradient(up, axis=2))
n = np.cross(gx.reshape(3, -1).T, gy.reshape(3, -1).T).T.reshape(3, H, W)
n /= np.linalg.norm(n, axis=0, keepdims=True) + 1e-6
n = np.stack([ndimage.gaussian_filter(c, 8) for c in n]); n /= np.linalg.norm(n, axis=0, keepdims=True) + 1e-6
vol = zarr.open(url, mode="r")[str(LV)]  # (z, y, x)
print("volume", vol.shape, vol.chunks, flush=True)
res = np.zeros((2 * R + 1, H, W), np.uint8); T = 512; t0 = time.time()
for y0 in range(0, H, T):
    for x0 in range(0, W, T):
        sl = (slice(y0, y0 + T), slice(x0, x0 + T))
        if not vup[sl].any():
            continue
        P = up[(slice(None),) + sl]; N = n[(slice(None),) + sl]; ok = vup[sl]
        pts = P[:, ok]; lo = np.floor(pts.min(1) - R - 2).astype(int); hi = np.ceil(pts.max(1) + R + 2).astype(int)
        lo = np.maximum(lo, 0)
        for attempt in range(8):                                  # the CT host drops connections under load: retry
            try:
                box = np.asarray(vol[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]], np.float32)  # z, y, x
                break
            except Exception as e:
                print("retry", attempt, str(e)[:80], flush=True); time.sleep(5 * (attempt + 1))
        PG = float(os.environ.get("PAGANIN", "0"))                 # pi*lambda*D*(delta/beta)/pixel^2, e.g. 472 for 113 keV, 1.2 m, 9.362 um, d/b 1000
        if PG > 0:                                                    # Paganin on projections == isotropic 3D filter on a parallel-beam volume
            f = [np.fft.fftfreq(n) for n in box.shape]; F2 = f[0][:, None, None] ** 2 + f[1][None, :, None] ** 2 + f[2][None, None, :] ** 2
            box = np.real(np.fft.ifftn(np.fft.fftn(box - box.mean()) / (1 + PG * F2))) + box.mean()
            box = 5.0 * box - 4.0 * ndimage.gaussian_filter(box, 1.2)    # unsharp mask, c = 4, sigma = 1.2 px (as in arXiv 2606.29085)
            b1, b2 = np.percentile(box, 1), np.percentile(box, 99.9); box = (box - b1) / (b2 - b1 + 1e-6) * 255
        for k in range(-R, R + 1):
            q = P + k * N
            c = np.stack([q[2] - lo[2], q[1] - lo[1], q[0] - lo[0]])
            v = ndimage.map_coordinates(box, c.reshape(3, -1), order=1, mode="constant").reshape(ok.shape)
            res[(k + R,) + sl] = np.where(ok, np.clip(v, 0, 255), 0).astype(np.uint8)
        print(f"tile {y0},{x0} box {box.shape} {time.time() - t0:.0f}s", flush=True)
np.save(out, res); print("saved", res.shape, flush=True)
