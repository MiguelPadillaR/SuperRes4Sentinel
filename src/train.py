import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.utils.constants import *
from src.utils.utils import *
from src.data.dataset import PairedImageDataset
from src.model.model import ModelConfig, build_model

import random

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            psnrs, ssims = [], []
            with torch.no_grad():
                for batch in dl_val:
                    lr = batch['lr'].to(device)
                    hr = batch['hr'].to(device)
                    sr = model(lr)
                    sr_np = (sr.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).round().astype('uint8')
                    hr_np = (hr.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).round().astype('uint8')
                    psnrs.append(get_psnr(sr_np, hr_np))
                    ssims.append(get_ssim(sr_np, hr_np))
            mean_psnr = float(np.mean(psnrs))
            mean_ssim = float(np.mean(ssims))
            print(f'Val PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f}')

            # Set up checkpoint path
            ckpt_model_path = CKPT_DIR / MODEL_NAME
            ckpt_model_path.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_model_path / f'{MODEL_NAME}_x{SCALE}_e{epoch:03d}_psnr{mean_psnr:.2f}.pth'

            # Save checkpoint only 10 models and when best model appears
            offset = max(1, EPOCHS // 10)
            if epoch%offset == 0: 
                torch.save(model.state_dict(), ckpt_path)
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save(model.state_dict(), CKPT_DIR / f'best_{MODEL_NAME}_x{SCALE}.pth')

if __name__ == "__main__":
    start_time = time.time()
    train()
    finish_time = time.time()
    print(f"\nTotal time:\t{(finish_time - start_time)/60:.1f} minutes")