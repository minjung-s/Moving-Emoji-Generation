import os
import torch
from PIL import Image
from torchvision.utils import save_image
from model import Generator # refer to stylegan2-pytorch model.py


def generate_image(img_path, model_path):

    image = Image.open(img_path)
    width, height = image.size

    if (width != 1024) or (height != 1204):        
        img_resize_lanczos = image.resize((1024, 1024), Image.LANCZOS)
        img_resize_lanczos.save(img_path) 
    image.close()

    # refer to stylegan2-pytorch projector.py
    os.system(f"python projector.py --ckpt {model_path} --size 1024 {img_path}")
    file_name = img_path.split(".")[-2]
    latent = torch.load(f"{file_name}.pt")[file_name]['latent']
    ckpt = torch.load(model_path)

    g = Generator(1024, 512, 8).cuda()
    g.load_state_dict(ckpt['g_ema'], stric=False)

    trunc = g.mean_latent(4096)

    img, _ = g(
        [latent],
        truncation=1,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    save_image(img[0], img_path)
    




