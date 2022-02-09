import argparse
import os
import json
import numpy as np

def parse_option():

    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')

    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
    parser.add_argument('--in-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--in-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--out-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--out-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[128, 160, 224, 384])

    opt = parser.parse_args()

    return opt

def main(opt):
    experiments_dir = '/home/cc/osr/experiments/save'#specify the root dir
    best_acc_list = []
    cur_acc_list = []
    best_auroc_list = []
    cur_auroc_list = []

    for dir in os.listdir(experiments_dir):
        exp_name, dataset, model_arch, _, _, _, num_classes, random_seed, _, _ = dir.split("_")

        if opt.exp_name == exp_name and opt.in_dataset == dataset and opt.in_num_classes == int(num_classes[2:]):
            json_dir = os.path.join(experiments_dir, dir, "results")
            for json_file in os.listdir(json_dir):
                _, ood_dataset, num_ood_classes = json_file.split("_")
                if ood_dataset[3:] == opt.out_dataset and int(num_ood_classes[4:-5]) == opt.out_num_classes:
                    with open(os.path.join(json_dir, json_file)) as f:
                        data = json.load(f)
                        if json_file.startswith("best"):
                            best_acc_list.append(data['in_acc'])
                            best_auroc_list.append(data['auroc'])
                        else:
                            cur_acc_list.append(data['in_acc'])
                            cur_auroc_list.append(data['auroc'])
    
    best_acc = np.mean(best_acc_list)
    best_auroc = np.mean(best_auroc_list)
    cur_acc = np.mean(cur_acc_list)
    cur_auroc = np.mean(cur_auroc_list)
    print(best_acc, best_auroc, cur_acc, cur_auroc)

if __name__ == '__main__':
    opt = parse_option()
    main(opt)




