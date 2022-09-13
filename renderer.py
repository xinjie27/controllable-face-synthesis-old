import argparse
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np

# Deep3DFaceRecon dependencies
from Deep3DFaceRecon.options.test_options import TestOptions
from Deep3DFaceRecon.models import create_model
from Deep3DFaceRecon.util.visualizer import MyVisualizer
from Deep3DFaceRecon.util import util

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Input mat file path",
        dest="input_path",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="outputs/",
        help="Output image path",
        dest="output_path",
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


class Renderer():
    def __init__(self, resolution, rank=0):
        opt = TestOptions().parse()
        self.opt = opt
        self.model = create_model(opt)
        device = torch.device(rank)
        self.model.setup(opt)
        self.model.device = device
        self.model.parallelize()
        self.model.eval()
        self.visualizer = MyVisualizer(opt)
        self.resolution = resolution

    def render(self, coeffs):
        pred_mask, pred_face_color, pred_face_vertex, pred_face_tex, pred_face_norm, pred_lm = self.model.forward2(coeffs)
        # img_tensor = self.model.render(pred_mask, pred_face)

        img_tensor = pred_mask * pred_face_color
        scaleF = self.resolution / 224
        img_tensor = F.interpolate(img_tensor, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        pred_mask = F.interpolate(pred_mask, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        pred_face_vertex = F.interpolate(pred_face_vertex, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        pred_face_tex = F.interpolate(pred_face_tex, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        pred_face_norm = F.interpolate(pred_face_norm, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        img_tensor = img_tensor * 2 - 1 # Change the range to [-1, 1]
        pred_face_tex = pred_face_tex * 2 - 1

        return img_tensor, pred_mask, pred_face_vertex, pred_face_tex, pred_face_norm, pred_lm

    def get_mask(self, coeffs):
        pred_mask, _ = self.model.forward2(coeffs)
        scaleF = self.resolution / 224
        pred_mask = F.interpolate(pred_mask, scale_factor=(scaleF, scaleF), mode='bilinear', align_corners=False, antialias=True)
        return pred_mask
    
    def visualize(self, img_tensor, path = None):
        # print(torch.sum(img_tensor))
        # print(img_tensor.shape)
        img = util.tensor2im((img_tensor.squeeze() + 1) / 2)
        util.save_image(img, path)

renderer128 = Renderer(128)

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


def get_coeff_npy(npy_path):
    coeffs = np.load(npy_path)
    return torch.Tensor(coeffs).to("cuda")


def test_single(mat_path, output_path):
    '''
    Tests the Renderer on a single image (.mat file).
    '''
    coeffs = get_coeff(mat_path)
    r = Renderer()
    img_t = (r.render(coeffs) + 1) / 2 
    r.visualize(img_t, output_path)

# def test_sample(output_path):
#     '''
#     Tests the Renderer on a single image (.mat file).
#     '''
#     sample = sampler.sample_m(1)
#     r = Renderer()
#     img_t = r.render(sample) 
#     r.visualize(img_t, output_path)

def test_mean(npy_path, output_path):
    coeffs = np.load(npy_path).reshape((1,-1))
    coeffs = torch.Tensor(coeffs).to("cuda")
    r = Renderer()
    img_t = r.render(coeffs)
    r.visualize(img_t, output_path)

def test_batch(img_dir):
    pass


if __name__ == "__main__":
#    config = get_config()
#    mat_path = config["input_path"]
#    output_path = config["output_path"]

    mat_path = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/outputs/id/ID150/real/cam3/cropped_frames0057.mat"
    output_path = "./renderer_out/test.png"
#    test_single(mat_path, output_path)
    # test_sample(output_path)

#    npy_path = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/3DMM-StyleGAN2/datasets/3DMM_stats/3dmm_mean.npy"
#    mean_output_path = "./renderer_out/test_mean.png"
#    test_mean(npy_path, mean_output_path)