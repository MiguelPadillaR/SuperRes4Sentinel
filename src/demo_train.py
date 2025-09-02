import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.constants import CKPT_DIR, HR_DIR, LR_DIR, EPOCHS
from src.data.dataset import PairedImageDataset
from src.model.model import build_model, ModelConfig

# ---------------- Training ---------------- #
def train(scale=4, n_epochcs=EPOCHS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = PairedImageDataset(LR_DIR, HR_DIR, scale=scale)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # model = ModelConfig().to(device)
    model_id = 'edsr'
    model_name = f"{model_id}_x{scale}"
    model = build_model(ModelConfig(name=model_id, scale=scale)).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, n_epochcs + 1):
        total_loss = 0
        for batch in loader:
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), str(CKPT_DIR) + f"/{model_name}.pth")
    print("✅ Model saved to out/checkpoints/edsr_x4.pth")

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    train()
