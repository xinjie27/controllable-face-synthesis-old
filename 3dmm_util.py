from genericpath import exists
import os
import glob
import argparse
from tkinter import COMMAND
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
from mtcnn import MTCNN
import cv2
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from renderer import Renderer
from Deep3DFaceRecon.util import util

from Deep3DFaceRecon.options.test_options import TestOptions
from Deep3DFaceRecon.util.load_mats import load_lm3d
from Deep3DFaceRecon.util.preprocess import align_img
from PIL import Image

'''
Author: socialvv
'''

# Model config
MODEL_NAME = "test_model"
NUM_EPOCHS = 20


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["id", "exp", "gamma"],
        # required=True,
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=False,
        default="/media/jayden27/d5a43ee1-58b7-4fc1-a084-7883ce143674/datasets",
        help="Input directory",
        dest="input_dir",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        # required=True,
        choices=["D","E"],
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory",
        dest="output_dir",
    )

    parser.add_argument("--log-dir", type=str, default="../log/")

    parser.add_argument("-r", "--num-reals", type=int, required=False, default=3)
    
    parser.add_argument("-f", "--num-fakes", type=int, required=False, default=1)

    parser.add_argument("--real-id", nargs="+", required=False, default=[150])

    parser.add_argument("--fake-id", nargs="+", required=False, default=[150])

    parser.add_argument("-t", "--threshold", type=float, required=False, default=0.5)

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config



def get_data_path(img_dir):

    # im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    # lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    # lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    #7/13
    img_path = glob.glob(f'{img_dir}/*/*.png')

    return img_path


### PREPROCESSING MODULE ###


def preprocess(img_dir):
    """
    Preprocesses the data and save 5 facial landmarks for each image.

    :param img_dir:
    :param output_dir:
    """
    detector = MTCNN()

    img_list = glob.glob(f'{img_dir}/*.png')
    # print(img_list)

    for img_file in img_list:
        
        # print(img_file)
        f_path = img_file.replace('png', 'txt')
        
        # print(f_path)

        # skip already preprocessed img
        if os.path.exists(f_path): continue

        img_path = img_file

        # # skip detection folder
        # if os.path.isdir(img_path): continue
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(img.shape)
        try:
            facial_landmarks = detector.detect_faces(img)[0]["keypoints"]
        except IndexError:
            # print("Error path:", img_path)
            continue

        assert len(facial_landmarks) == 5, "Incorrect number of facial landmarks"
        landmark_vals = list(facial_landmarks.values())

        with open(f_path, "w") as f:
            for x, y in landmark_vals:
                f.write(str(x) + " " + str(y) + "\n")


### END OF PREPROCESSING MODULE ###


### RUN DEEP3D_FACERECON MODEL ###

def call_model(config, output_dir, img_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else: return

    cmd = f"python ./test.py --name={MODEL_NAME}  --epoch={NUM_EPOCHS} --img_folder={img_dir}"
    os.system(cmd)

### END ###

### MASK OUT BACKGROUND ###

def preprocess_img(image_path, landmarks_path, lm3d_std, to_tensor = True):
    im = Image.open(image_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(landmarks_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, _, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im


def mask_background(img_paths, coeffs_dir, size = 224):
    renderer = Renderer()
    coeffs_paths = [f'{coeffs_dir}/{os.path.split(im)[1].replace("png","npy")}' for im in img_paths]
    lm_paths = [im.replace("png","txt") for im in img_paths]
    opt = TestOptions().parse()
    lm3d_std = load_lm3d(opt.bfm_folder)
    for img_path, coeffs_path, landmarks_path in list(zip(img_paths, coeffs_paths, lm_paths)):
        if not os.path.exists(coeffs_path):
            continue
        coeffs = get_coeff_npy(coeffs_path)
        mask = renderer.get_mask(coeffs, size = size)

        img_tensor = preprocess_img(img_path, landmarks_path, lm3d_std)
        
        output_vis = mask * img_tensor.to("cuda")
        output_vis_numpy = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        output = torch.tensor(output_vis_numpy / 255., dtype=torch.float32).permute(0, 3, 1, 2).squeeze()
        out_path = img_path.replace(f"ffhq-{size}x{size}", f"ffhq-{size}x{size}_masked")
        if not os.path.exists(os.path.split(out_path)[0]):
            os.mkdir(os.path.split(out_path)[0])
        util.save_image(util.tensor2im(output), out_path)
        print(out_path)

    return

def test_align(img_paths, coeffs_dir, size = 224):
    renderer = Renderer(resolution=size)
    coeffs_paths = [f'{coeffs_dir}/{os.path.split(im)[1].replace("png","npy")}' for im in img_paths]
    print(len(img_paths), len(coeffs_paths))    
    for img_path, coeffs_path in list(zip(img_paths, coeffs_paths)):
        if not os.path.exists(coeffs_path):
            continue
        coeffs = get_coeff_npy(coeffs_path)
        img_tensor = renderer.render(coeffs)[0]
        output_path = f'test/align/{os.path.basename(img_path)}'
        renderer.visualize(img_tensor, path=output_path)
        print(output_path)
    return

def align(img_paths, size = 224):
    lm_paths = [im.replace("png","txt") for im in img_paths]
    opt = TestOptions().parse()
    lm3d_std = load_lm3d(opt.bfm_folder)
    for img_path, landmarks_path in list(zip(img_paths, lm_paths)):
        if not os.path.exists(landmarks_path):
            continue
        out_path = img_path.replace(f"ffhq-{size}x{size}", f"ffhq-{size}x{size}_aligned")
        if os.path.exists(out_path):
            continue
        if not os.path.exists(os.path.split(out_path)[0]):
            os.mkdir(os.path.split(out_path)[0])
        img_tensor = preprocess_img(img_path, landmarks_path, lm3d_std)
        
        util.save_image(util.tensor2im(img_tensor.squeeze()), out_path)
        print(out_path)

    return

### END ###

### TEST MODULE ###

def get_coeff_npy(npy_path):
    coeffs = np.load(npy_path)
    return torch.Tensor(coeffs).to("cuda")

def get_coeff(mat_path):
    """
    :param filepath: image folder per camera source (e.g. /real/cam1)
    """
    if not mat_path.endswith(".mat"):
        return None

    # Read coefficient files
    coeff_dict = loadmat(mat_path)
    # coeff_dict["id"] = coeff_dict["id"].reshape(-1)
    # coeff_dict["exp"] = coeff_dict["exp"].reshape(-1)
    # coeff_dict["tex"] = coeff_dict["tex"].reshape(-1)
    # coeff_dict["trans"] = coeff_dict["trans"].reshape(-1)
    # coeff_dict["gamma"] = coeff_dict["gamma"].reshape(-1)
    coeffs = np.hstack([coeff_dict["id"], 
                            coeff_dict["exp"], 
                            coeff_dict["tex"],
                            coeff_dict["angle"],
                            coeff_dict["gamma"],
                            coeff_dict["trans"]
                            ])
    return torch.Tensor(coeffs).to("cuda") # coeff_dict


OUTPUT_DIR = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/3DMM-StyleGAN2/datasets/3DMM_stats"

def analyze(coeff_list):
    for i in range(len(coeff_list)):
        coeff_dict = coeff_list[i]
        coeff_list[i] = np.hstack([coeff_dict["id"], 
                        coeff_dict["exp"], 
                        coeff_dict["tex"],
                        coeff_dict["angle"],
                        coeff_dict["gamma"],
                        coeff_dict["trans"]
                        ]).reshape(-1)
    
    coeff_arr = np.array(coeff_list)
    mean = np.mean(coeff_arr, axis=0)
    stddev = np.std(coeff_arr, axis=0)

    # print(mean)
    # print(stddev)
    np.save(OUTPUT_DIR + "/3dmm_mean.npy", mean)
    np.save(OUTPUT_DIR + "/3dmm_std.npy", stddev)
        
    return mean, stddev

def plot_dist(coeff_list, idx=0, n_bins=50):
    id_list = []
    exp_list = []
    tex_list = []
    angle_list = []
    gamma_list = []
    trans_list = []

    for coeff_dict in coeff_list:
        id_list.append(coeff_dict["id"][idx])
        exp_list.append(coeff_dict["exp"][idx])
        tex_list.append(coeff_dict["tex"][idx])
        angle_list.append(coeff_dict["angle"][idx])
        gamma_list.append(coeff_dict["gamma"][idx])
        trans_list.append(coeff_dict["trans"][idx])
    
    plt.figure(1)
    plt.hist(id_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/id_dist")

    plt.figure(2)
    plt.hist(exp_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/exp_dist")

    plt.figure(3)
    plt.hist(tex_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/tex_dist")

    plt.figure(4)
    plt.hist(angle_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/angle_dist")

    plt.figure(5)
    plt.hist(gamma_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/gamma_dist")

    plt.figure(6)
    plt.hist(trans_list, bins=n_bins, histtype='step')
    plt.savefig(OUTPUT_DIR + "/trans_dist")



if __name__ == "__main__":
    config = get_config()
    
    # Step 1: preprocess ffhq data
    # print("=====preprocessing=====")
    # REAL_IMG_DIR = '/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq-224x224'
    # REAL_IMG_DIR = '/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/SG2_Skeleton/test'
    # preprocess(REAL_IMG_DIR)

    # Step 2: call Deep3dRecon model
    # print("Calling model...")
    # coeffs_dir = '/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq_coeffs_224'
    # call_model(config, coeffs_dir, REAL_IMG_DIR)

    # step 3: replace background with 0 in ffhq
    # img_path = get_data_path(REAL_IMG_DIR)
    # output_dir = '/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq-224x224_masked'
    # mask_background(img_path, coeffs_dir)

    img_paths = get_data_path('/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq-224x224')
    coeffs_dir = '/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/GAN/datasets/ffhq_coeffs_224'
    # test_align(img_paths, coeffs_dir)
    align(img_paths)

    # analyze data
    # coeff_list = []
    # for mat in os.listdir(output_dir):
    #     coeff_list.append(loadmat(join(output_dir,mat)))
    # # print(len(coeff_list))
    # print("Complete loading coeff list")

    # # analyze(coeff_list)
    # plot_dist(coeff_list, idx=0)


