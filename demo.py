import torch
import glob
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
import pickle

import dnnlib
from torch_utils import misc
from renderer import renderer128

MODEL_PATH = "./training-runs/naive_gbuffer/network-snapshot-002662.pkl"
FILEPATH = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq_coeffs_224/*"


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


def _save_img(img):
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
    coeffs = torch.Tensor(np.load(coeffs_list[index])).to(device)
    z = torch.randn(1, 512).to(device)
    data.z = z
    data.coeffs = coeffs
    c = None
    image_npy = G(z, coeffs, c).detach().cpu().numpy()
    image_npy = image_npy.squeeze().transpose((1, 2, 0))
    image_npy = _save_img(image_npy)
    image = ImageTk.PhotoImage(Image.fromarray(image_npy))
    label_img.configure(image=image)
    label_img.image = image


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


frame_p = tk.Frame(master=window)
label_img_new = tk.Label(master=frame_p)
label_img_new.pack(side=tk.LEFT)
frame_p.grid(row=1, column=1)


frame_r = tk.Frame(master=window)
label_img_rdr = tk.Label(master=frame_r)
label_img_rdr.pack(side=tk.LEFT)
frame_r.grid(row=1, column=2)

def perturb():
    scale_factor = torch.Tensor(np.array([id_var.get()] * 80 + [exp_var.get()] * 64 + [tex_var.get()] * 80 + [angle_var.get()] * 3 + [gamma_var.get()] * 27 + [trans_var.get()] * 3)).to(device)
    c = None
    img_new_npy = G(data.z, scale_factor * data.coeffs, c).detach().cpu().numpy()
    img_new_npy = img_new_npy.squeeze().transpose((1, 2, 0))
    img_new_npy = _save_img(img_new_npy)
    image_new = ImageTk.PhotoImage(Image.fromarray(img_new_npy))
    label_img_new.configure(image=image_new)
    label_img_new.image = image_new
    img_rdr_npy, _ = renderer128.render(scale_factor * data.coeffs)
    img_rdr_npy = img_rdr_npy.cpu().numpy().squeeze().transpose((1, 2, 0))
    img_rdr_npy = _save_img(img_rdr_npy)
    image_rdr = ImageTk.PhotoImage(Image.fromarray(img_rdr_npy))
    label_img_rdr.configure(image=image_rdr)
    label_img_rdr.image = image_rdr


button_p = tk.Button(master=window, text="Perturb", command=perturb)
button_p.grid(row=8, column=1, pady=10)


# Reset
def reset():
    id_slider.set(1)
    exp_slider.set(1)
    tex_slider.set(1)
    angle_slider.set(1)
    gamma_slider.set(1)
    trans_slider.set(1)
    generate()
    perturb()


frame_r = tk.Frame(master=window)
button_r = tk.Button(master=window, text="Reset", command=reset)
button_r.grid(row=8, column=0, pady=10)



# Start the demo
window.mainloop()