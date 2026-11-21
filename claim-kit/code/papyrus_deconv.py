"""Deconvolution adapted to ink on papyrus (not stars):
- blur kernel = the ink model's own blur, measured on papyrus against expert letters (runs/argus_v3_psf.npy);
- total-variation regularised Richardson-Lucy (Dey et al. 2006): ink strokes are flat with sharp edges, so TV keeps
  them and stops the fibre grain from being amplified;
- adaptive strength: where the local signal is weak, the result is blended back towards the input.
usage: from papyrus_deconv import pdeconv"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
PSF = None
def _psf():
    global PSF
    if PSF is None: PSF = np.load("runs/argus_v3_psf.npy")
    return PSF
def tv_div(u, eps=1e-3):
    gy, gx = np.gradient(u); n = np.sqrt(gx ** 2 + gy ** 2 + eps ** 2)
    return np.gradient(gy / n, axis=0) + np.gradient(gx / n, axis=1)
def pdeconv(img, valid, it=10, lam=0.004, adapt=True):
    psf = _psf(); pf = psf[::-1, ::-1]
    x = np.clip(np.where(valid, img, 0), 1e-4, None); est = x.copy()
    for _ in range(it):
        est = est * fftconvolve(x / np.maximum(fftconvolve(est, psf, "same"), 1e-6), pf, "same") / np.maximum(1 - lam * tv_div(est), 0.2)
        est = np.clip(est, 0, 5)
    if adapt:
        loc = ndimage.gaussian_filter(np.where(valid, img, 0), 60) / np.maximum(ndimage.gaussian_filter(valid.astype(np.float32), 60), 1e-6)
        g = np.where(valid, img, 0); lo, hi = np.percentile(loc[valid], 50), np.percentile(loc[valid], 95)
        w = np.clip((loc - lo) / (hi - lo + 1e-6), 0, 1)                    # 0 where the area is faint, 1 where it is clearly inked
        sc = np.percentile(est[valid], 99.5) / max(np.percentile(g[valid], 99.5), 1e-6)
        est = w * est + (1 - w) * g * sc
    return np.where(valid, est, 0)
