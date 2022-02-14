from src.model import OODTransformer, VisionTransformer as ViT
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from src.config import get_train_config
from src.utils import setup_device, TensorboardWriter, MetricTracker, load_checkpoint, write_json, accuracy
import random
from src.dataset import *
import os

def run_model(model, loader):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    for images, target in loader:
        total += images.size(0)
        images = images.cuda()
        with torch.no_grad():
            output = model(images)

        out_list.append(output.data)
        tgt_list.append(target)

    return  torch.cat(out_list), torch.cat(tgt_list)

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, classes_mean, device=torch.device('cpu')):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        mean = classes_mean[batch_target]

        loss = criterion(batch_pred, mean)

        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        #torch.cuda.empty_cache()
        if metrics.writer is not None:
            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())

    return metrics.result()

def valid_epoch(epoch, model, data_loader, criterion, metrics, classes_mean, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            mean = classes_mean[batch_target]
            loss = criterion(batch_pred, mean)
            losses.append(loss.item())
                

    loss = np.mean(losses)
    if metrics.writer is not None:
        metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    return metrics.result()

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, classes_mean, device_ids, best=False, save_freq=100):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
        'classes_mean': classes_mean.cpu()
    }
    filename = str(save_dir + 'ckpt_epoch_current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'ckpt_epoch_best.pth')
        torch.save(state, filename)
    elif epoch%save_freq==0:
        filename = str(save_dir + 'ckpt_epoch_' + str(epoch) + '.pth')
        print('Saving file : ',filename)
        torch.save(state, filename)

def main(config, device, device_ids):
    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss']
        
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model = OODTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate,
             )

    # load checkpoint
    if config.checkpoint_path:
        state_dict = torch.load(config.checkpoint_path, map_location=torch.device("cpu"))['state_dict']
        print("Loading pretrained weights from {}".format(config.checkpoint_path))
        if not config.eval and config.num_classes != state_dict['classifier.weight'].size(0)  :#not
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print("re-initialize fc layer")
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys from checkpoint ",missing_keys.missing_keys)
        print("Unexpected keys in network : ",missing_keys.unexpected_keys)

    for param in model.transformer.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = False

    for param in model.embedding.parameters():
        param.requires_grad = False

    if config.dataset == "MNIST" or config.dataset == "SVHN" or config.dataset == "CIFAR10":
        total_classes = 10
    elif config.dataset == "TinyImageNet":
        total_classes = 200

    random.seed(config.random_seed)
    known_classes = random.sample(range(0, total_classes), config.num_classes)
    train_dataset = eval("get{}Dataset".format(config.dataset))(image_size=config.image_size, split='train', data_path=config.data_dir, known_classes=known_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_dataset = eval("get{}Dataset".format(config.dataset))(image_size=config.image_size, split='in_test', data_path=config.data_dir, known_classes=known_classes)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    # create dataloader
    config.model = 'vit'

    train_emb, train_targets = run_model(model, train_dataloader)
    in_classes = torch.unique(train_targets)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_emb[idx] for idx in class_idx]
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats],dim=0)

    # training criterion
    print("create criterion and optimizer")
    criterion = torch.nn.MSELoss().to(device)

    # create optimizers and learning rate scheduler
    if config.opt =="AdamW":
        print("Using AdmW optimizer")
        optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.lr,weight_decay=config.wd)
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
            momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)


    # start training
    print("start training")
    best_loss = float("inf")
    best_epoch = 0
    config.epochs = config.train_steps // len(train_dataloader)
    print("length of train loader : ",len(train_dataloader),' and total epoch ',config.epochs)
    for epoch in range(1, config.epochs + 1):
        for param_group in optimizer.param_groups:
            print("learning rate at {0} epoch is {1}".format(epoch, param_group['lr']))

        log = {'epoch': epoch}

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, classes_mean, device)
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, classes_mean, device)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        if log['loss'] < best_loss:
            best_loss = log['loss']
            best_epoch = epoch
            best = True
        else:
            best = False

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, classes_mean, device_ids, best, config.save_freq)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

    print("Best loss : ",best_loss, ' for ',best_epoch)# saving class mean
    best_curr_loss = {'best_loss':best_loss,'best_epoch':best_epoch,
                     'curr_loss':log['loss'],'curr_epoch':epoch}
    write_json(best_curr_loss,os.path.join(config.checkpoint_dir,'loss.json'))


if __name__ == '__main__':
    config = get_train_config()
    # device
    device, device_ids = setup_device(config.n_gpu)

    main(config, device, device_ids)