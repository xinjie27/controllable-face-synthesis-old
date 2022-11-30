from json import load
from random import randint
import torch
import glob
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import pickle
from scipy.io import loadmat

import dnnlib
from torch_utils import misc
from renderer import renderer128
from Deep3DFaceRecon.util import util

MODEL_PATH = "./training-runs/00000-ffhq-128x128-aligned-69885-gpus1-batch64-gamma1/network-snapshot-003942.pkl"
FILEPATH = "./datasets/ffhq_coeffs_224_aligned/*"


device = torch.device('cuda', 0)

# Load the generator
with open(MODEL_PATH, "rb") as f:
    G = pickle.load(f)["G_ema"].to(device)

# Structure to store z and coefficients
class Data():
    def __init__(self):
        self.z = None
        self.coeffs = None
data = Data()


def _scale_img(img):
    lo, hi = [-1, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    return img


def generate():
    try:
        index = int(entry_index.get())
    except ValueError:
        print("Please enter a valid integer.")

    coeffs_list = list(glob.glob(FILEPATH))
    coeffs = np.hstack([
            loadmat(coeffs_list[index])["id"],
            loadmat(coeffs_list[index])["exp"],
            loadmat(coeffs_list[index])["tex"],
            loadmat(coeffs_list[index])["angle"],
            loadmat(coeffs_list[index])["gamma"],
            loadmat(coeffs_list[index])["trans"],
            ])
    coeffs = torch.Tensor(coeffs).to(device)
    z = torch.randn(1, 64).to(device)
    data.z = z
    data.coeffs = coeffs
    # c = None
    image_npy = G(z, coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy()
    image_npy = image_npy.squeeze().transpose((1, 2, 0))
    image_npy = _scale_img(image_npy)
    image = ImageTk.PhotoImage(Image.fromarray(image_npy))
    label_img.configure(image=image)
    label_img.image = image
    img_rdr_npy = renderer128.render(data.coeffs)[0]
    img_rdr_npy = img_rdr_npy.cpu().numpy().squeeze().transpose((1, 2, 0))
    img_rdr_npy = _scale_img(img_rdr_npy)
    image_rdr = ImageTk.PhotoImage(Image.fromarray(img_rdr_npy))
    label_img_rdr.configure(image=image_rdr)
    label_img_rdr.image = image_rdr


# Set up the window
window = tk.Tk()
window.title("3DMMGAN Demo")
window.resizable(width=True, height=True)

# Generation section
frame_g = tk.Frame(master=window)
entry_index = tk.Entry(master=frame_g, width=10)
label_g = tk.Label(master=frame_g, text="FFHQ Image Index")

label_g.grid(row=0, column=0, sticky="e")
entry_index.grid(row=0, column=1, sticky="w")

button_g = tk.Button(master=window, text="Generate", command=generate)

frame_g.grid(row=0, column=0, padx=10)
button_g.grid(row=0, column=1, pady=10)

# Display section
frame_d = tk.Frame(master=window)
label_img = tk.Label(master=frame_d)
label_img.pack(side=tk.LEFT)
frame_d.grid(row=1, column=0)


# Perturbation section
scale_hi = 2
scale_lo = -1

# truncation factor
truncation_psi = 0.5


id_var = tk.DoubleVar()
id_slider = tk.Scale(window, variable=id_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
id_slider.set(1)
id_label = tk.Label(window, text="Identity")
id_label.grid(row=2, column=0)
id_slider.grid(row=2, column=1)

exp_var = tk.DoubleVar()
exp_slider = tk.Scale(window, variable=exp_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
exp_slider.set(1)
exp_label = tk.Label(window, text="Expression")
exp_label.grid(row=3, column=0)
exp_slider.grid(row=3, column=1)

tex_var = tk.DoubleVar()
tex_slider = tk.Scale(window, variable=tex_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
tex_slider.set(1)
tex_label = tk.Label(window, text="Texture")
tex_label.grid(row=4, column=0)
tex_slider.grid(row=4, column=1)

angle_var = tk.DoubleVar()
angle_slider = tk.Scale(window, variable=angle_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
angle_slider.set(1)
angle_label = tk.Label(window, text="Angle")
angle_label.grid(row=5, column=0)
angle_slider.grid(row=5, column=1)

gamma_var = tk.DoubleVar()
gamma_slider = tk.Scale(window, variable=gamma_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
gamma_slider.set(1)
gamma_label = tk.Label(window, text="Gamma")
gamma_label.grid(row=6, column=0)
gamma_slider.grid(row=6, column=1)

trans_var = tk.DoubleVar()
trans_slider = tk.Scale(window, variable=trans_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
trans_slider.set(1)
trans_label = tk.Label(window, text="Translation")
trans_label.grid(row=7, column=0)
trans_slider.grid(row=7, column=1)

z_var = tk.DoubleVar()
z_slider = tk.Scale(window, variable=z_var, from_=scale_lo, to=scale_hi, orient="horizontal", resolution=0.01)
z_slider.set(1)
z_label = tk.Label(window, text="z vector")
z_label.grid(row=8, column=0)
z_slider.grid(row=8, column=1)


frame_p = tk.Frame(master=window)
label_img_new = tk.Label(master=frame_p)
label_img_new.pack(side=tk.LEFT)
frame_p.grid(row=1, column=1)


frame_rdr = tk.Frame(master=window)
label_img_rdr = tk.Label(master=frame_rdr)
label_img_rdr.pack(side=tk.LEFT)
frame_rdr.grid(row=1, column=2)

def perturb():
    scale_factor = torch.Tensor(np.array([id_var.get()] * 80 + [exp_var.get()] * 64 + [tex_var.get()] * 80 + [angle_var.get()] * 3 + [gamma_var.get()] * 27 + [trans_var.get()] * 3)).to(device)
    z = data.z * z_var.get()
    # c = None
    img_new_npy = G(z, scale_factor * data.coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy()
    img_new_npy = img_new_npy.squeeze().transpose((1, 2, 0))
    img_new_npy = _scale_img(img_new_npy)
    image_new = ImageTk.PhotoImage(Image.fromarray(img_new_npy))
    label_img_new.configure(image=image_new)
    label_img_new.image = image_new
    img_rdr_npy = renderer128.render(scale_factor * data.coeffs)[0]
    img_rdr_npy = img_rdr_npy.cpu().numpy().squeeze().transpose((1, 2, 0))
    img_rdr_npy = _scale_img(img_rdr_npy)
    image_rdr = ImageTk.PhotoImage(Image.fromarray(img_rdr_npy))
    label_img_rdr.configure(image=image_rdr)
    label_img_rdr.image = image_rdr


button_p = tk.Button(master=window, text="Perturb", command=perturb)
button_p.grid(row=9, column=3, pady=10)


# Reset
def reset():
    id_slider.set(1)
    exp_slider.set(1)
    tex_slider.set(1)
    angle_slider.set(1)
    gamma_slider.set(1)
    trans_slider.set(1)
    z_slider.set(1)
    z = data.z
    coeffs = data.coeffs
    image_npy = G(z, coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy()
    image_npy = image_npy.squeeze().transpose((1, 2, 0))
    image_npy = _scale_img(image_npy)
    image = ImageTk.PhotoImage(Image.fromarray(image_npy))
    label_img.configure(image=image)
    label_img.image = image

# Resample z
def resample_z():
    z = torch.randn(1, 64).to(device)
    c = None
    image_npy = G(z, data.coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy()
    image_npy = image_npy.squeeze().transpose((1, 2, 0))
    image_npy = _scale_img(image_npy)
    image = ImageTk.PhotoImage(Image.fromarray(image_npy))
    label_img.configure(image=image)
    label_img.image = image
    img_rdr_npy = renderer128.render(data.coeffs)[0]
    img_rdr_npy = img_rdr_npy.cpu().numpy().squeeze().transpose((1, 2, 0))
    img_rdr_npy = _scale_img(img_rdr_npy)
    image_rdr = ImageTk.PhotoImage(Image.fromarray(img_rdr_npy))
    label_img_rdr.configure(image=image_rdr)
    label_img_rdr.image = image_rdr

def resample_m():
    coeffs_list = list(glob.glob(FILEPATH))
    index = randint(0, len(coeffs_list))
    coeffs = np.hstack([
            loadmat(coeffs_list[index])["id"],
            loadmat(coeffs_list[index])["exp"],
            loadmat(coeffs_list[index])["tex"],
            loadmat(coeffs_list[index])["angle"],
            loadmat(coeffs_list[index])["gamma"],
            loadmat(coeffs_list[index])["trans"],
            ])
    coeffs = torch.Tensor(coeffs).to(device)
    z = data.z
    image_npy = G(z, coeffs, noise_mode = "const", truncation_psi = truncation_psi)[0].detach().cpu().numpy()
    image_npy = image_npy.squeeze().transpose((1, 2, 0))
    image_npy = _scale_img(image_npy)
    image = ImageTk.PhotoImage(Image.fromarray(image_npy))
    label_img.configure(image=image)
    label_img.image = image
    img_rdr_npy = renderer128.render(data.coeffs)[0]
    img_rdr_npy = img_rdr_npy.cpu().numpy().squeeze().transpose((1, 2, 0))
    img_rdr_npy = _scale_img(img_rdr_npy)
    image_rdr = ImageTk.PhotoImage(Image.fromarray(img_rdr_npy))
    label_img_rdr.configure(image=image_rdr)
    label_img_rdr.image = image_rdr

# def smooth_z():
#     z1 = torch.randn(1, 512).to(device)
#     z2 = torch.randn(1, 512).to(device)
#     c = None
#     z1_image_npy = G(z1, data.coeffs, c).detach().cpu().numpy()
#     z1_image_npy = z1_image_npy.squeeze().transpose((1, 2, 0))
#     z1_image_npy = _scale_img(z1_image_npy)
#     util.save_image(z1_image_npy, 'test/z1.png')
#     z2_image_npy = G(z2, data.coeffs, c).detach().cpu().numpy()
#     z2_image_npy = z2_image_npy.squeeze().transpose((1, 2, 0))
#     z2_image_npy = _scale_img(z2_image_npy)
#     util.save_image(z2_image_npy, 'test/z2.png')

#     n = 100
#     for i in range(n):
#         z = z1 + (z2 - z1) / n * (i+1)
#         z_image_npy = G(z, data.coeffs, c).detach().cpu().numpy()
#         z_image_npy = z_image_npy.squeeze().transpose((1, 2, 0))
#         z_image_npy = _scale_img(z_image_npy)
#         util.save_image(z_image_npy, f'test/{i+1}.png')


# frame_r = tk.Frame(master=window)
button_r = tk.Button(master=window, text="Reset", command=reset)
button_r.grid(row=9, column=0, pady=10)

button_z = tk.Button(master=window, text="Resample z", command=resample_z)
button_z.grid(row=9, column=1, pady=10)

button_m = tk.Button(master=window, text="Resample m", command=resample_m)
button_m.grid(row=9, column=2, pady=10)


# Start the demo
window.mainloop()