python train.py --outdir=./training-runs --data=./datasets/ffhq-masked-128x128.zip --gpus=1 --batch=64 --gamma=1 --cbase=16384 --glr=0.0025 --dlr=0.0025 --mbstd-group=8 --mirror=0 --aug=noaug --tick=1 --fp32=True --batch-gpu=16 --metrics=none