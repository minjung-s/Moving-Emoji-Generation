import os
import torch
from torchvision.utils import save_image
from model import Generator


def generate_image(img_path, model_path):

    os.system(f"python projector.py --ckpt {model_path} --size 1024 {img_path}")
    file_name = img_path.split(".")[-2]
    latent = torch.load(f"{file_name}.pt")[file_name]['latent']
    ckpt = torch.load(model_path)

    g = Generator(1024, 512, 8, channel_multiplier=args.channel_multiplier).cuda()
    g.load_state_dict(ckpt['g_ema'], stric=False)

    # trunc = g.mean_latent(4096)

    img, _ = g(
        [latent],
        # truncation=1,
        # truncation_latent=trunc,
        input_is_latent=True,
    )

    save_image(img[0], img_path)
    




