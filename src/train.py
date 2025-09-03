import argparse
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

def train(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LR_INIT, model_name=MODEL_NAME, scale=SCALE, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS,
          val_interval=VAL_INTERVAL, num_workers=N_WORKERS, lr_dir=LR_DIR, hr_dir=HR_DIR,
          ckpt_dir=CKPT_DIR, augment=True, device=None):
    device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ds = PairedImageDataset(lr_dir, hr_dir, scale=scale, augment=augment)
    n_val = max(1, int(0.1*len(ds)))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers)

    cfg = ModelConfig(name=model_name, scale=scale, n_resblocks=n_resblocks, n_feats=n_feats)
    model = build_model(cfg).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0

    # Setup model's checkpoints directory
    ckpt_model_path = ckpt_dir / "_".join([model_name, 'x'+str(scale)])
    ckpt_model_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs+1):
        crop_size = random.choice([96, 128, 256, 384, 512])
        ds.__set_crop_size__(crop_size)   # method you add in PairedImageDataset
        model.train()
        pbar = tqdm(dl_train, desc=f'Epoch {epoch}/{epochs}')
        for batch in pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()

        if epoch % val_interval == 0:
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
            print(f'Val PSNR: {mean_psnr:.2f} dB | Best PSNR: {best_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | Crop size: {crop_size} px')

            # Save checkpoint for only 10 models (regardless of epochs) or when best model appears
            offset = max(1, epochs // 10)
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                print(f'Better model found at epoch {epoch}, saving as best_{model_name}_x{scale}.pth')
                torch.save(model.state_dict(), ckpt_dir / f'best_{model_name}_x{scale}.pth')
            elif epoch%offset == 0: 
                ckpt_path = ckpt_model_path / f'{model_name}_x{scale}_e{epoch:03d}_psnr{mean_psnr:.2f}.pth'
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                }, ckpt_path)
            elif epoch == epochs:
                ckpt_path = ckpt_model_path / f'last_{model_name}_x{scale}.pth'
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                }, ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train super-resolution model on LoRes-HiRes paired image dataset. Image files must have the same name on both directories.")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=LR_INIT, help="Initial learning rate")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model architecture name")
    parser.add_argument("--scale", type=int, default=SCALE, help="Super-resolution scale factor")
    parser.add_argument("--n-resblocks", type=int, default=N_RESBLOCKS, help="Number of residual blocks (for EDSR model)")
    parser.add_argument("--n-feats", type=int, default=N_FEATS, help="Number of feature maps (for EDSR model)")
    parser.add_argument("--val-interval", type=int, default=VAL_INTERVAL, help="Validation interval (in epochs)")
    parser.add_argument("--num-workers", type=int, default=N_WORKERS ,help="Number of DataLoader workers")
    parser.add_argument("--lr-dir", type=str, default=LR_DIR, help="Directory with low-resolution images")
    parser.add_argument("--hr-dir", type=str, default=HR_DIR, help="Directory with high-resolution images")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR, help="Directory to save model checkpoints")
    parser.add_argument("--augment", action="store_true", default=True, help="Use data augmentation (random flips and rotations)")
    parser.add_argument("--device", type=str, choices=["cpu","cuda"], default=None, help="Device to use for training (default: auto-detect)")

    args = parser.parse_args()

    start_time = time.time()
    train(**vars(args))
    finish_time = time.time()
    print(f"\nTotal time:\t{(finish_time - start_time)/60:.1f} minutes")