"""Public demo of the Interprete on ALREADY PUBLISHED text: PHerc0306B line 1 (Prodigle team claim, CC BY-NC 4.0,
papyrologist reading TO DIAUTO AIRETON). Per letter: the ink-map crop, the Interprete top-3, and the 5 most similar
real papyrus letters (HomerComp). No image from any unpublished scroll."""
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import interprete as I
from letter_bench import bench
it = I.Interprete("runs/interprete_v1.pt"); fn = it.net.f[:-2]
def feats(X):
    with torch.no_grad():
        return torch.cat([torch.nn.functional.normalize(fn(torch.as_tensor(np.asarray(X[k:k+1024], np.float32)).to(it.dev)[:, None]).float(), dim=1).cpu() for k in range(0, len(X), 1024)])
L = np.load("hunt/letters_homercomp.npz"); LF = feats(L["shapes"].astype(np.float32))
items = bench()["PHerc0306B"]; P = it.predict(np.array([x for _, x in items]))
T = 110; f = ImageFont.truetype("arial.ttf", 22); fs = ImageFont.truetype("arial.ttf", 16)
g = Image.new("RGB", (T + 230 + 5 * (T + 4), 50 + len(items) * (T + 6)), "white"); d = ImageDraw.Draw(g)
d.text((6, 12), "PHerc. 0306B line 1 (published reading)  |  ink map  |  Interprete top-3  |  5 nearest real papyrus letters", fill=(0, 0, 0), font=fs)
ok = 0
for i, ((t, x), p) in enumerate(zip(items, P)):
    Y = 50 + i * (T + 6); top = np.argsort(-p[:-1])[:3]; ok += I.CLASSES[top[0]] == t
    g.paste(Image.fromarray((x * 255).astype(np.uint8)).resize((T, T)).convert("RGB"), (0, Y))
    d.text((T + 10, Y + 15), f"true: {t}", fill=(0, 0, 0), font=f)
    d.text((T + 10, Y + 55), "  ".join(f"{I.CLASSES[k]} {p[k]:.2f}" for k in top), fill=(0, 120, 0) if I.CLASSES[top[0]] == t else (180, 0, 0), font=fs)
    near = np.argsort(-(LF @ feats(x[None])[0]).numpy())[:5]
    for j, k in enumerate(near):
        g.paste(Image.fromarray(L["greys"][k]).resize((T, T)).convert("RGB"), (T + 230 + j * (T + 4), Y))
g.save("publish/interprete/pherc0306b_line1_interprete.png"); print("exact", ok, "/", len(items))
