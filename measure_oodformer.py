import argparse
import os
import torch
from src.config import *
from src.model import VisionTransformer as ViT
import random
from torch.utils.data import DataLoader
from src.dataset import *
from torch.nn import functional as F
import sklearn.metrics as skm
from src.utils import write_json

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
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[128, 160, 224, 384, 448])

    opt = parser.parse_args()

    return opt

def run_model(model, loader, softmax=False):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    for images, target in loader:
        total += images.size(0)
        images = images.cuda()
        output, classifier = model(images,feat_cls=True)

        out_list.append(output.data.cpu())
        cls_list.append(F.softmax(classifier, dim=1).data.cpu())
        tgt_list.append(target)

    return  torch.cat(out_list), torch.cat(tgt_list), torch.cat(cls_list)

def euclidean_dist(x, support_mean):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = support_mean.size(0)
    d = x.size(1)
    if d != support_mean.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    support_mean = support_mean.unsqueeze(0).expand(n, m, d)

    #return torch.pow(x - support_mean, 2).sum(2)
    return ((x - support_mean)*(x-support_mean)).sum(2)

def get_distances(in_list, out_list, classes_mean):

    print('Compute euclidean distance for in and out distribution data')
    test_dists = euclidean_dist(in_list, classes_mean)
    out_dists = euclidean_dist(out_list, classes_mean)

    return test_dists, out_dists

def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc

def main(opt, model):
    ckpt = torch.load(opt.ckpt_file, map_location=torch.device("cpu"))
    # load networks
    #model = opt.model
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.cuda()
    model.eval() 
    print('load model: ' + opt.ckpt_file)

    # load ID dataset
    print('load in target data: ', opt.in_dataset)
    if opt.in_dataset == "CUB":
        import pickle
        if opt.in_dataset == "CUB":
            splits_file_path = "src/cub_osr_splits.pkl"

        with open(splits_file_path, "rb") as f:
            splits = pickle.load(f)

            known_classes = splits['known_classes']
            train_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='train', data_path=opt.data_dir, known_classes=known_classes)
            in_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=known_classes)

            unknown_classes = splits['unknown_classes'][opt.mode]
            out_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=unknown_classes)
            
            
    else:
        random.seed(opt.random_seed)
        if opt.in_dataset == "MNIST" or opt.in_dataset == "SVHN" or opt.in_dataset == "CIFAR10":
            total_classes = 10
        elif opt.in_dataset == "CIFAR100":
            total_classes = 100
        elif opt.in_dataset == "TinyImageNet":
            total_classes = 200

        known_classes = random.sample(range(0, total_classes), opt.in_num_classes)

        train_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='train', data_path=opt.data_dir, known_classes=known_classes)
        in_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=known_classes)

        # load OOD dataset
        print('load out target data: ', opt.out_dataset)
        if opt.in_dataset == opt.out_dataset:
            unknown_classes =  list(set(range(total_classes)) -  set(known_classes))
            out_dataset = eval("get{}Dataset".format(opt.in_dataset))(image_size=opt.image_size, split='out_test', data_path=opt.data_dir, known_classes=known_classes)
        else:
            random.seed(opt.random_seed)
            if opt.out_dataset == "MNIST" or opt.out_dataset == "CIFAR10":
                out_total_classes = 10
            elif opt.out_dataset == "CIFAR100" :
                out_total_classes = 100
            elif opt.out_dataset == "TinyImageNet":
                out_total_classes = 200
            unknown_classes = random.sample(range(0, out_total_classes), opt.out_num_classes)
            out_dataset = eval("get{}Dataset".format(opt.out_dataset))(image_size=opt.image_size, split='in_test', data_path=opt.data_dir, known_classes=unknown_classes)

    test_data_len = min(len(in_dataset), len(out_dataset))
    random.seed(opt.random_seed)
    in_index = random.sample(range(len(in_dataset)), test_data_len)
    in_dataset = torch.utils.data.Subset(in_dataset, in_index)
    random.seed(opt.random_seed)
    out_index = random.sample(range(len(out_dataset)), test_data_len)
    out_dataset = torch.utils.data.Subset(out_dataset, out_index)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    in_dataloader = DataLoader(in_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    out_dataloader = DataLoader(out_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    print('Compute sample mean for training data....')

    train_emb, train_targets, train_sfmx = run_model(model,train_dataloader)
    train_acc = float(torch.sum(torch.argmax(train_sfmx, dim=1) == train_targets)) / len(train_sfmx)

    in_classes = torch.unique(train_targets)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_emb[idx] for idx in class_idx]
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats],dim=0)

    in_emb, in_targets, in_sfmx = run_model(model,in_dataloader)
    in_acc = float(torch.sum(torch.argmax(in_sfmx, dim=1) == in_targets)) / len(in_sfmx)

    out_emb, out_targets, out_sfmx = run_model(model,out_dataloader)
    in_dists, out_dists = get_distances(in_emb, out_emb, classes_mean)

    in_dist_lbl = torch.argmax(in_sfmx, dim=1).cpu()
    in_score = [dist[in_dist_lbl[i]].cpu() for i, dist in enumerate(in_dists)]

    ood_lbl = torch.argmax(out_sfmx, dim=1).cpu()
    ood_score = [dist[ood_lbl[i]].cpu() for i, dist in enumerate(out_dists)]

    auroc = get_roc_sklearn(in_score,ood_score)
    print("SSD AUROC {0}".format(auroc))
    
    return {'train_acc': train_acc, 'in_acc': in_acc, 'auroc': auroc, 'known_classes': sorted(known_classes), 'unknown_classes': sorted(unknown_classes)}

    


    

def run_ood_distance(opt):
    experiments_dir = os.path.join(os.getcwd(), 'experiments/save')#specify the root dir
    for dir in os.listdir(experiments_dir):
        exp_name, dataset, model_arch, _, _, _, num_classes, random_seed, _, _ = dir.split("_")
        opt = eval("get_{}_config".format(model_arch))(opt)
        model = ViT(
                 image_size=(opt.image_size, opt.image_size),
                 patch_size=(opt.patch_size, opt.patch_size),
                 emb_dim=opt.emb_dim,
                 mlp_dim=opt.mlp_dim,
                 num_heads=opt.num_heads,
                 num_layers=opt.num_layers,
                 num_classes=opt.in_num_classes,
                 attn_dropout_rate=opt.attn_dropout_rate,
                 dropout_rate=opt.dropout_rate,
                 )
        
        if opt.exp_name == exp_name and opt.in_dataset == dataset and opt.in_num_classes == int(num_classes[2:]):
            ckpt_dir = os.path.join(experiments_dir, dir, "checkpoints")
            for ckpt_file in os.listdir(ckpt_dir):
                if ckpt_file.endswith("best.pth") or ckpt_file.endswith("current.pth"):
                    ckpt_file = os.path.join(ckpt_dir, ckpt_file)
                    opt.ckpt_file = ckpt_file
                    opt.random_seed = int(random_seed[2:])

                    if dataset == "CUB":
                        result = dict()
                        for mode in ["Easy", "Medium", "Hard"]:
                            opt.mode = mode
                            sub_result = main(opt, model)
                            result[mode] = sub_result

                    else:
                        result = main(opt, model)
                    result_path = os.path.join(experiments_dir, dir, "results", "best_ood{}_nood{}.json".format(opt.out_dataset, opt.out_num_classes) if "best" in ckpt_file else "current_ood{}_nood{}.json".format(opt.out_dataset, opt.out_num_classes))
                    write_json(result, result_path)




if __name__ == '__main__':
    #parse argument
    opt = parse_option()
    run_ood_distance(opt)