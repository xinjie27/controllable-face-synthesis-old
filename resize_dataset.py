import glob
from PIL import Image
import os

dataset_dir = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/controllable-face-synthesis/test/align/ffhq-256x256/00000"
output_dir = "/media/socialvv/d5a43ee1-58b7-4fc1-a084-7883ce143674/controllable-face-synthesis/test/align/ffhq-512x512/00000"

# images_path = glob.glob(f"{dataset_dir}/*/*.png")
# for i,image_path in enumerate(images_path):
#     folder_name, image_name = image_path.split("/")[-2], image_path.split("/")[-1]
#     new_image_dir = f"{output_dir}/{folder_name}"
#     if not os.path.exists(new_image_dir):
#         os.makedirs(new_image_dir)
#     output_path = f"{new_image_dir}/{image_name}"
#     if os.path.exists(output_path): continue
    
#     image = Image.open(image_path)
#     new_image = image.resize((224,224))
#     new_image.save(output_path)
#     print("resized image", i)

images_path = glob.glob(f"{dataset_dir}/*.png")
for i,image_path in enumerate(images_path):
    image_name = image_path.split("/")[-1]
    new_image_dir = output_dir
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    output_path = f"{new_image_dir}/{image_name}"
    if os.path.exists(output_path): continue
    
    image = Image.open(image_path)
    new_image = image.resize((512,512))
    new_image.save(output_path)
    print("resized image", i)