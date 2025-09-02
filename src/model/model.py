from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.utils.constants import MODEL_NAME, N_FEATS, N_RESBLOCKS, SCALE

# Minimal EDSR implementation (simplified)
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0,1.0,1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3,3,1,1)
        self.weight.data.div_(std.view(3,1,1,1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters(): p.requires_grad=False

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return x + res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if scale in (2, 4, 8):
            for _ in range(int(scale).bit_length()-1):
                m += [nn.Conv2d(n_feats, 4*n_feats, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        elif scale == 3:
            m += [nn.Conv2d(n_feats, 9*n_feats, 3, 1, 1), nn.PixelShuffle(3), nn.ReLU(inplace=True)]
        else:
            raise ValueError('Unsupported scale for EDSR: {}'.format(scale))
        super().__init__(*m)

class EDSR(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=64):
        super().__init__()
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)], nn.Conv2d(n_feats, n_feats, 3,1,1))
        self.upsample = Upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x) + x
        x = self.upsample(res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x

# Lightweight SRCNN baseline
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(True),
            nn.Conv2d(32, 3, 5, padding=2)
        )
    def forward(self, x):
        return self.net(x)

# ESRGAN/RRDBNet via BasicSR (if available)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except Exception:
    RRDBNet = None

@dataclass
class ModelConfig:
    name: str = MODEL_NAME  # 'edsr' | 'esrgan' | 'srcnn'
    scale: int = SCALE
    n_resblocks: int = N_RESBLOCKS
    n_feats: int = N_FEATS
    pretrained: Optional[str] = None  # path to .pth


def build_model(cfg: ModelConfig) -> nn.Module:
    name = cfg.name.lower()
    if name == 'edsr':
        model = EDSR(scale=cfg.scale, n_resblocks=cfg.n_resblocks, n_feats=cfg.n_feats)
    elif name == 'srcnn':
        model = SRCNN()
    elif name == 'esrgan':
        if RRDBNet is None:
            raise RuntimeError('BasicSR not installed or RRDBNet unavailable.')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=cfg.scale)
    else:
        raise ValueError(f'Unknown model {cfg.name}')

    if cfg.pretrained:
        state = torch.load(cfg.pretrained, map_location='cpu')
        # handle state dict wrapping
        if 'params_ema' in state:
            state = state['params_ema']
        elif 'state_dict' in state:
            state = {k.replace('module.', ''): v for k,v in state['state_dict'].items()}
        model.load_state_dict(state, strict=False)
        print(f'Successfully loaded pretrained weights from {cfg.pretrained}')
    return model
