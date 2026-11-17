"""false-ink rate at 80% 0841 ink recall on several blank PHerc0139 wraps. usage: blank_check.py name=path[:pub] ..."""
import sys, types, numpy as np, torch
import train as T
from ensemble import load
from eval_all import load as load_public
dev = torch.device("cuda")
args = types.SimpleNamespace(patch=256, depth=17, batch=8, amp="bf16", tta=2)
s41 = T.Segment("pherc0841-w00")
blanks = [T.Segment(n) for n in ("pherc0139-w053", "pherc0139-w056", "pherc0139-w057", "pherc0139-w058")]
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    pub = path.endswith(":pub"); path = path.removesuffix(":pub")
    m = (load_public(path)[0] if pub else load(path)).to(dev).eval()
    pr = T.predict(m, s41, args, dev, region=s41.val)
    t80 = float(np.quantile(pr[s41.val & s41.ink_eval], 0.2))
    out = [float((T.predict(m, b, args, dev, region=b.val)[b.val] >= t80).mean()) for b in blanks]
    print(f"{name:22s} " + "  ".join(f"{b.name[-4:]} {o*100:5.1f}%" for b, o in zip(blanks, out)), flush=True)
    del m; torch.cuda.empty_cache()
