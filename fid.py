import os

a = os.listdir('./training-runs/00000-stylegan2-ffhq-256x256-gpus1-batch64-gamma1')
b = [x for x in a if x.endswith('.pkl')]
b.sort()

for x in b:
    os.system('python calc_metrics.py --network=./training-runs/00000-stylegan2-ffhq-256x256-gpus1-batch64-gamma1/' + x)
