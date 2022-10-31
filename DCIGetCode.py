import numpy as np
import glob
from scipy.io import loadmat
import torch
import pickle

from Deep3DFaceRecon.util import util
from PIL import Image

LATENTS_PATH = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/SG2_Skeleton/datasets/ffhq_coeffs_224_aligned/*"
OUTPUT_PATH = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/SG2_Skeleton/DCI_outputs"
MODEL_PATH = "./training-runs/oct_22_z_no_product/network-snapshot-005120.pkl"

device = torch.device("cuda", 0)

def lerp(a,b,t):
    return a + (b - a) * t


def GetCode(filepath):
    coeffs_list = list(glob.glob(filepath))
    n_coeffs = len(coeffs_list)
    dlatents=np.zeros((n_coeffs,257),dtype='float32')
    for index in range(n_coeffs):
        coeffs = np.hstack([
            loadmat(coeffs_list[index])["id"],
            loadmat(coeffs_list[index])["exp"],
            loadmat(coeffs_list[index])["tex"],
            loadmat(coeffs_list[index])["angle"],
            loadmat(coeffs_list[index])["gamma"],
            loadmat(coeffs_list[index])["trans"],
            ])
        dlatents[index] = coeffs
        if index % 500 == 0:
            print(index)

    return dlatents

def _scale_img(img):
    lo, hi = [-1, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img


def GetImg(dlatents, truncation_psi = 0.5):
    # Load the generator
    with open(MODEL_PATH, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()

    img_list = []
    for i,coeffs in enumerate(dlatents):
        coeffs = torch.Tensor(coeffs).to(device).unsqueeze(0)
        z = torch.randn(1, 64).to(device)
        image_npy = G(z, coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy() # shape (c, h, w)
        image_npy = _scale_img(image_npy)
        image_npy = image_npy.transpose((1, 2, 0)) # shape (h, w, c)
        img_list.append(np.expand_dims(image_npy, axis=0))
        print(i)

    all_images = np.concatenate(img_list) # shape (n, h, w, c)
    return all_images



if __name__ == "__main__":
    # Generate latent codes M and corresponding images
    # print("Generating latent codes...")
    # dlatents = GetCode(LATENTS_PATH)
    # np.save(OUTPUT_PATH + "/M.npy", dlatents)

    # dlatents = np.load(OUTPUT_PATH + "/M.npy")
    # print("Generating images...")
    # all_images = GetImg(dlatents)
    # np.save(OUTPUT_PATH + "/images.npy", all_images)

    images = np.load(OUTPUT_PATH + "/images.npy")
    new_images = []
    for i in range(30000):
        print(i)
        image = images[i]
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((256,256), resample=Image.Resampling.BICUBIC)
        new_image = np.array(pil_img)
        new_images.append(np.expand_dims(new_image, axis=0))
    new_images = np.concatenate(new_images)
    np.save(OUTPUT_PATH + "/new_images.npy", new_images)
    util.save_image(new_images[10], OUTPUT_PATH + "/test.png")