import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from src.constants import CKPT_DIR

# ---------------- Dataset ---------------- #
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=64, scale=4):
        self.lr_files = sorted(glob(os.path.join(lr_dir, "*.png")))
        self.hr_files = sorted(glob(os.path.join(hr_dir, "*.png")))
        self.crop_size = crop_size
        self.scale = scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        hr = Image.open(self.hr_files[idx]).convert("RGB")

        # Random crop aligned between LR and HR
        w, h = lr.size
        x = torch.randint(0, w - self.crop_size, (1,)).item()
        y = torch.randint(0, h - self.crop_size, (1,)).item()

        lr_crop = lr.crop((x, y, x + self.crop_size, y + self.crop_size))
        hr_crop = hr.crop(
            (x * self.scale, y * self.scale,
             (x + self.crop_size) * self.scale,
             (y + self.crop_size) * self.scale)
        )

        return self.to_tensor(lr_crop), self.to_tensor(hr_crop)

# ---------------- Model (Tiny EDSR) ---------------- #
class EDSR(nn.Module):
    def __init__(self, num_channels=3, num_feats=64, num_blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(num_channels, num_feats, 3, 1, 1)
        body = []
        for _ in range(num_blocks):
            body += [nn.Conv2d(num_feats, num_feats, 3, 1, 1), nn.ReLU(True)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(num_feats, num_channels * (scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        x = self.pixel_shuffle(x)
        return x

# ---------------- Training ---------------- #
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SRDataset("data/LR", "data/HR", crop_size=64, scale=4)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = EDSR().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 21):  # 20 epochs
        total_loss = 0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), str(CKPT_DIR) + "/edsr_x4.pth")
    print("✅ Model saved to out/checkpoints/edsr_x4.pth")

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    train()
