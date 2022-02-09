import argparse
import random
from pathlib import Path
import os

parser = argparse.ArgumentParser("Train gooddetector")

# basic config
parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
parser.add_argument("--num-classes", type=int, default=6, help="number of classes in dataset")

config = parser.parse_args()
if config.dataset == "MNIST":
    random_seed = 42
elif config.dataset == "CIFAR10":
    random_seed = 24
elif config.dataset == "TinyImageNet":
    random_seed = 533
else:
    random_seed = 335

random.seed(random_seed)
seeds = random.sample(range(1000), 5)
commands = []
experiments_dir = '/home/cc/osr/experiments/save'#specify the root dir

for seed in seeds:
    path = Path("./")
    for dir in os.listdir(experiments_dir):
        exp_name, dataset, model_arch, _, _, _, num_classes, random_seed, _, _ = dir.split("_")
        if exp_name == 'osrclassifier' and dataset == config.dataset and config.num_classes == int(num_classes[2:]) and seed == int(random_seed[2:]):
            ckpt_path = os.path.join(experiments_dir, dir, "checkpoints", "ckpt_epoch_current.pth")
            cline = "python train_detector.py --exp-name osrdetector --n-gpu 4 --tensorboard --image-size 224 \
                      --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset {} --num-classes {} --checkpoint-path {} --random-seed {}".format(config.dataset, config.num_classes, ckpt_path, seed)
            if os.system(cline):
                raise RuntimeError('program {} failed!'.format(cline))