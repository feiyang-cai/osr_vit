import argparse
import random
from pathlib import Path
import os

parser = argparse.ArgumentParser("Train classifier")

# basic config
parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
parser.add_argument("--num-classes", type=int, default=6, help="number of classes in dataset")

config = parser.parse_args()
if config.dataset == "MNIST":
    random_seed = 42
elif config.dataset == "SVHN":
    random_seed = 7
elif config.dataset == "CIFAR10":
    random_seed = 24
elif config.dataset == "TinyImageNet":
    random_seed = 533
else:
    random_seed = 335

random.seed(random_seed)
seeds = random.sample(range(1000), 5)
commands = []
for seed in seeds:
    path = Path("./")
    cline = "python train_classifier.py --exp-name osrclassifier --n-gpu 4 --tensorboard --image-size 224 \
              --batch-size 256 --num-workers 16 --train-steps 4590 --lr 0.01 --wd 1e-5 --dataset {} --num-classes {} --random-seed {}".format(config.dataset, config.num_classes, seed)
    #if config.dataset != "MNIST":
    cline += " --checkpoint-path {}".format(config.checkpoint_path)
    if os.system(cline):
        raise RuntimeError('program {} failed!'.format(cline))