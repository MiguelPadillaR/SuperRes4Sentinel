import torch
from PIL import Image
import torchvision.transforms as T
from src.train_sr import EDSR
from src.constants import CKPT_DIR

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
    super_resolve("data/LR/36.627058_-6.051960.png", "out/res/36.627058_-6.051960.png")
