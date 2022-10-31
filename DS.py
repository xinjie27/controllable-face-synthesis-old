import numpy as np
import glob
from scipy.io import loadmat
import torch
import pickle
from PIL import Image
from random import randint

from Deep3DFaceRecon.util import util
from face_recon_util import preprocess, call_model

MODEL_PATH = "./training-runs/oct_22_z_no_product/network-snapshot-005120.pkl"
DPATH = "./datasets/DS"
# Deep3DFaceRecon Model config
MODEL_NAME = "test_model"
NUM_EPOCHS = 20

device = torch.device("cuda", 0)


def _get_mean_std():
    coeffs = np.load(DPATH + "/M.npy") # shape (n, 257)
    coeffs_mean = np.mean(coeffs, axis=0)
    coeffs_std = np.std(coeffs, axis=0)
    np.save(DPATH + "/mean.npy", coeffs_mean)
    np.save(DPATH + "/std.npy", coeffs_std)


def sample_coeffs(n_sets=1000, set_size=10):
    # load M
    M = np.load(DPATH + "/M.npy")
    N = M.shape[0]

    # identity 0:80 + texture 144:224
    id_coeffs = np.zeros((n_sets * set_size, 257))
    for i in range(0, n_sets):
        idx = randint(0, N - 1)
        id_set = M[idx]
        for j in range(0, set_size):
            id_idx = randint(0, N - 1)
            id_set[0:80] = M[id_idx, 0:80]
            id_set[144:224] = M[id_idx, 144:224]
            id_coeffs[i * set_size + j] = id_set
    # shape (10000, 257)
    print(id_coeffs.shape)
    np.save(DPATH + "/id_coeffs.npy", id_coeffs)

    # expression 80:144
    exp_coeffs = np.zeros((n_sets * set_size, 257))
    for i in range(0, n_sets):
        idx = randint(0, N - 1)
        exp_set = M[idx]
        for j in range(0, set_size):
            exp_idx = randint(0, N - 1)
            exp_set[80:144] = M[exp_idx, 80:144]
            exp_coeffs[i * set_size + j] = exp_set
    # shape (10000, 257)
    print(exp_coeffs.shape)
    np.save(DPATH + "/exp_coeffs.npy", exp_coeffs)

    # angle/pose 224:227
    pose_coeffs = np.zeros((n_sets * set_size, 257))
    for i in range(0, n_sets):
        idx = randint(0, N - 1)
        pose_set = M[idx]
        for j in range(0, set_size):
            pose_idx = randint(0, N - 1)
            pose_set[224:227] = M[pose_idx, 224:227]
            pose_coeffs[i * set_size + j] = pose_set
    # shape (10000, 257)
    print(pose_coeffs.shape)
    np.save(DPATH + "/pose_coeffs.npy", pose_coeffs)

    # gamma 227:254
    gamma_coeffs = np.zeros((n_sets * set_size, 257))
    for i in range(0, n_sets):
        idx = randint(0, N - 1)
        gamma_set = M[idx]
        for j in range(0, set_size):
            gamma_idx = randint(0, N - 1)
            gamma_set[227:254] = M[gamma_idx, 227:254]
            gamma_coeffs[i * set_size + j] = gamma_set
    # shape (10000, 257)
    print(gamma_coeffs.shape)
    np.save(DPATH + "/gamma_coeffs.npy", gamma_coeffs)


def _scale_img(img):
    lo, hi = [-1, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img


def generate_imgs(coeffs_list, truncation_psi = 0.5):
    # Load the generator
    with open(MODEL_PATH, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()

    for i, coeffs in enumerate(coeffs_list):
        print(i)
        coeffs = torch.Tensor(coeffs).to(device).unsqueeze(0)
        z = torch.randn(1, 64).to(device)
        image_npy = G(z, coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy() # shape (c, h, w)
        image_npy = _scale_img(image_npy)
        image_npy = image_npy.transpose((1, 2, 0)) # shape (h, w, c)

        util.save_image(image_npy, DPATH + f"/id_images/id_img_{i}.png")


def recon_coeffs(img_dir):
    # Get facial landmarks for all images in img_dir
    preprocess(img_dir)
    # Call model
    call_model(DPATH + "/id_coeffs_recon", img_dir)


def compute_DS(mode, coeffs_dir, n_sets=1000, set_size=10):
    coeffs_list = list(glob.glob(coeffs_dir))
    n_coeffs = len(coeffs_list)

    variances = []
    for i in range(0, n_coeffs, set_size):
        coeffs_set = np.zeros((10, 257), dtype='float32')
        for j in range(set_size):
            coeffs_set[j] = np.hstack([
                loadmat(coeffs_list[i + j])["id"],
                loadmat(coeffs_list[i + j])["exp"],
                loadmat(coeffs_list[i + j])["tex"],
                loadmat(coeffs_list[i + j])["angle"],
                loadmat(coeffs_list[i + j])["gamma"],
                loadmat(coeffs_list[i + j])["trans"],
                ])
        set_var = np.var(coeffs_set, axis=0) # shape (257,)
        variances.append(set_var)
    variances = np.array(variances)

    u = np.mean(variances, axis=0)

    u_id = np.concatenate((u[0: 80], u[144: 224]))
    u_exp = u[80: 144]
    u_pose = u[224: 227]
    u_gamma = u[227: 254]
    
    # TODO
    DS = None
    
    return DS

if __name__ == "__main__":
    # _get_mean_std()

    # sample_coeffs()
    id_coeffs = np.load(DPATH + "/id_coeffs.npy")
    generate_imgs(id_coeffs)