import os
import random
import torch
import torchvision.transforms as T

from PIL import Image
from src.utils.constants import CKPT_DIR, LR_DIR, RES_DIR
from src.model.model import ModelConfig, build_model

def super_resolve(img_path, out_path, model_path= CKPT_DIR / "edsr_x4.pth" , scale=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = 'edsr'
    model = build_model(ModelConfig(name=model_id, scale=scale)).to(device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(tensor).clamp(0, 1)

    sr_img = T.ToPILImage()(sr.squeeze().cpu())
    sr_img.save(out_path)
    print(f"Saved SR image to {out_path}")

if __name__ == "__main__":
    filename = os.listdir(LR_DIR)[random.randint(0, len(os.listdir(LR_DIR))-1)]
    super_resolve(LR_DIR / filename, RES_DIR / filename)
