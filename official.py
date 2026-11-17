"""Re-implementation of the official `vesuvius_unet_3d_stem_2d` ink model (scrollprize/ink_9um), written from the
checkpoint's tensor names and shapes only (no external code is run), so that:
  1. the published weights can be scored on exactly the same validation pixels as our models;
  2. the same architecture can be trained with our recipes (train.py --arch official).
Layout: depth fusion (two 3D convs, 16 ch, softmax attention over depth) -> 2D residual U-Net (nnU-Net ResEnc style,
BasicBlockD, 6 stages 32-64-128-256-320-320, blocks 1-3-4-6-6-6, one conv per decoder stage) -> 1x1 ink head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = (32, 64, 128, 256, 320, 320)
BLOCKS = (1, 3, 4, 6, 6, 6)


class ConvNormAct(nn.Module):
    def __init__(self, ci, co, k, stride=1, act=True, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, stride=stride, padding=k // 2, bias=bias)
        self.norm = nn.InstanceNorm2d(co, affine=True, eps=1e-5)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlockD(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.conv1 = ConvNormAct(ci, co, 3, stride, act=True)
        self.conv2 = ConvNormAct(co, co, 3, 1, act=False)
        if ci != co:                       # projection only when the channel count changes (index 1 = conv)
            self.skip = nn.Sequential(nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                                      ConvNormAct(ci, co, 1, 1, act=False, bias=False))
        elif stride > 1:
            self.skip = nn.AvgPool2d(stride)
        else:
            self.skip = nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class Stage(nn.Module):
    def __init__(self, ci, co, n, stride):
        super().__init__()
        self.blocks = nn.Sequential(*[BasicBlockD(ci if i == 0 else co, co, stride if i == 0 else 1) for i in range(n)])

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(self, cin, ch=CH, blocks=BLOCKS):
        super().__init__()
        self.stem = nn.Module()
        self.stem.convs = nn.ModuleList([ConvNormAct(cin, ch[0], 3)])
        self.stages = nn.ModuleList([Stage(ch[0] if i == 0 else ch[i - 1], ch[i], blocks[i], 1 if i == 0 else 2)
                                     for i in range(len(ch))])

    def forward(self, x):
        x = self.stem.convs[0](x)
        skips = []
        for s in self.stages:
            x = s(x)
            skips.append(x)
        return skips


class DecStage(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.convs = nn.ModuleList([ConvNormAct(ci, co, 3)])

    def forward(self, x):
        return self.convs[0](x)


class Decoder(nn.Module):
    def __init__(self, ch=CH):
        super().__init__()
        n = len(ch)
        self.transpconvs = nn.ModuleList([nn.ConvTranspose2d(ch[n - 1 - i], ch[n - 2 - i], 2, 2) for i in range(n - 1)])
        self.stages = nn.ModuleList([DecStage(2 * ch[n - 2 - i], ch[n - 2 - i]) for i in range(n - 1)])

    def forward(self, skips):
        x = skips[-1]
        for i, (t, s) in enumerate(zip(self.transpconvs, self.stages)):
            x = t(x)
            x = s(torch.cat([x, skips[-2 - i]], 1))
        return x


class DepthFusion(nn.Module):
    def __init__(self, c=16, combine="attn_max"):
        super().__init__()
        self.combine = combine
        self.features = nn.Sequential(
            nn.Conv3d(1, c, 3, padding=1, bias=False), nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, True),
            nn.Conv3d(c, c, 3, padding=1, bias=False), nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, True))
        self.attention_logits = nn.Conv3d(c, 1, 1)

    def forward(self, x):                                  # B,1,D,H,W -> B,2c,H,W
        f = self.features(x)
        w = torch.softmax(self.attention_logits(f), dim=2)
        att = (f * w).sum(2)
        other = {"max": f.amax(2), "mean": f.mean(2)}
        a, b = self.combine.split("_")
        parts = {"attn": att, **other}
        return torch.cat([parts[a], parts[b]], 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = Encoder(32)
        self.shared_decoder = Decoder()
        self.task_heads = nn.ModuleDict({"ink": nn.Conv2d(CH[0], 1, 1)})


class OfficialInkNet(nn.Module):
    def __init__(self, combine="attn_max"):
        super().__init__()
        self.depth_fusion = DepthFusion(16, combine)
        self.network = Net()

    def forward(self, x):
        f = self.depth_fusion(x)
        return self.network.task_heads["ink"](self.network.shared_decoder(self.network.shared_encoder(f)))


def load_official(path, combine="attn_max"):
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    sd = {k: v for k, v in sd.items() if ".all_modules." not in k}
    m = OfficialInkNet(combine)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    return m, missing, unexpected
