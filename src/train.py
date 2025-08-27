import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from constants import *
from src.data.dataset import PairedImageDataset
from src.model.model import ModelConfig, build_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ds = PairedImageDataset(LR_DIR, HR_DIR, scale=SCALE, augment=True)
    n_val = max(1, int(0.1*len(ds)))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    cfg = ModelConfig(name=MODEL_NAME, scale=SCALE, n_resblocks=NUM_RESBLOCKS, n_feats=N_FEATS)
    model = build_model(cfg).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl_train, desc=f'Epoch {epoch}/{EPOCHS}')
        for batch in pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()

        if epoch % VAL_INTERVAL == 0:
            model.eval()
            import numpy as np
            from .utils import psnr as psnr_fn, ssim as ssim_fn
            psnrs, ssims = [], []
            with torch.no_grad():
                for batch in dl_val:
                    lr = batch['lr'].to(device)
                    hr = batch['hr'].to(device)
                    sr = model(lr)
                    sr_np = (sr.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).round().astype('uint8')
                    hr_np = (hr.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).round().astype('uint8')
                    psnrs.append(psnr_fn(sr_np, hr_np))
                    ssims.append(ssim_fn(sr_np, hr_np))
            mean_psnr = float(np.mean(psnrs))
            mean_ssim = float(np.mean(ssims))
            print(f'Val PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}')

            # Save checkpoint
            ckpt_path = CKPT_DIR / f'{MODEL_NAME}_x{SCALE}_e{epoch:03d}_psnr{mean_psnr:.2f}.pth'
            torch.save(model.state_dict(), ckpt_path)
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save(model.state_dict(), CKPT_DIR / f'best_{MODEL_NAME}_x{SCALE}.pth')
