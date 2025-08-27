import torch
import os
from PIL import Image
import torchvision.transforms as T
from src.train_sr import EDSR
from src.utils.constants import LR_DIR, CKPT_DIR, RES_DIR

def super_resolve(img_path, out_path, model_path=str(CKPT_DIR) + "/edsr_x4.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EDSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(img_path).convert("RGB")
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(tensor).clamp(0, 1)

    sr_img = T.ToPILImage()(sr.squeeze().cpu())
    sr_img.save(out_path)
    print(f"Saved SR image to {out_path}")

if __name__ == "__main__":
    filename = os.listdir(LR_DIR)[0]
    super_resolve(LR_DIR / "36.31546_-6.02640_test.png", RES_DIR / "36.31546_-6.02640_test.png")
