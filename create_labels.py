import json
import glob
from scipy.io import loadmat
import numpy as np
import os
import os.path as path

dataset_json_file = "./datasets/ffhq-256x256_aligned/dataset.json"
coeffs_dir = "./datasets/ffhq_coeffs_256_aligned"
img_paths = glob.glob(f"./datasets/ffhq-256x256_aligned/*/*.png")

pair_list = []
anomaly = 0
for img_path in img_paths:
    img_path_short = img_path.split("/",3)[-1]
    img_path_tail = path.split(img_path)[1]
    coeff_path = path.join(coeffs_dir, img_path_tail).replace("png", "npy")
    if not path.exists(coeff_path):
        anomaly += 1
        print(f"anomaly {anomaly}", img_path, coeff_path)
        os.remove(img_path)
        continue
    # coeff_dict = loadmat(coeff_path)
    # id = coeff_dict["id"]
    # exp = coeff_dict["exp"]
    # tex = coeff_dict["tex"]
    # angle = coeff_dict["angle"]
    # gamma = coeff_dict["gamma"]
    # trans = coeff_dict["trans"]
    # label = list(np.concatenate((id, exp, tex, angle, gamma, trans), axis = 1).squeeze().astype(float))
    label = list(np.load(coeff_path).squeeze().astype(float))
    pair_list.append([img_path_short, label])

print(anomaly)
print(len(pair_list))
dataset_json = {}
dataset_json['labels'] = pair_list

with open(dataset_json_file, 'w') as f:
    json.dump(dataset_json, f)

# img_list = []
# for img_path in img_paths:
#     img_paths_tail = path.split(img_path)[1]
#     img_list.append(img_paths_tail.split(".")[0])

# for coeffs_path in glob.glob(coeffs_dir+"/*"):
#     assert "mat" in coeffs_path
#     if not "flipped" in coeffs_path:
#         img_name = path.split(coeffs_path)[1].split(".")[0]
#         if not img_name in img_list:
#             print(img_name)