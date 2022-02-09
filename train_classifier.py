from src.model import VisionTransformer as ViT
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from src.config import get_train_config
from src.utils import setup_device, TensorboardWriter, MetricTracker, load_checkpoint, write_json, accuracy
import random
from src.dataset import *
import os

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)

        loss = criterion(batch_pred, batch_target)

        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        #torch.cuda.empty_cache()
        if metrics.writer is not None:
            metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())

        if  batch_idx % 100 == 10:
            if config.num_classes >= 5:
                acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
                metrics.update('acc1', acc1.item())
                metrics.update('acc5', acc5.item())        
            else:
                acc1 = accuracy(batch_pred, batch_target, topk=(1,))
                metrics.update('acc1', acc1[0].item())


    return metrics.result()

def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            losses.append(loss.item())
            if config.num_classes >= 5:
                acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
                acc1s.append(acc1.item())
                acc5s.append(acc5.item())
            else:
                acc1 = accuracy(batch_pred, batch_target, topk=(1,))
                acc1s.append(acc1[0].item())
                

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    if config.num_classes >= 5:
        acc5 = np.mean(acc5s)
    if metrics.writer is not None:
        metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    if config.num_classes >= 5:
        metrics.update('acc5', acc5)
    return metrics.result()

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False, save_freq=100):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
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
    if config.num_classes >= 5:
        metric_names = ['loss', 'acc1', 'acc5']
    else:
        metric_names = ['loss', 'acc1']
        
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model = ViT(
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
        state_dict = load_checkpoint(config.checkpoint_path, new_img=config.image_size, emb_dim=config.emb_dim,
                                     layers= config.num_layers,patch=config.patch_size)
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

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    config.model = 'vit'
    random.seed(config.random_seed)
    if config.dataset == "MNIST" or config.dataset == "CIFAR10":
        total_classes = 10
    elif config.dataset == "TinyImageNet":
        total_classes = 200
    known_classes = random.sample(range(0, total_classes), config.num_classes)
    train_dataset = eval("get{}Dataset".format(config.dataset))(image_size=config.image_size, split='train', data_path=config.data_dir, known_classes=known_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_dataset = eval("get{}Dataset".format(config.dataset))(image_size=config.image_size, split='in_test', data_path=config.data_dir, known_classes=known_classes)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # training criterion
    print("create criterion and optimizer")
    criterion = torch.nn.CrossEntropyLoss().to(device)

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
    best_acc = 0.0
    best_epoch = 0
    config.epochs = config.train_steps // len(train_dataloader)
    print("length of train loader : ",len(train_dataloader),' and total epoch ',config.epochs)
    for epoch in range(1, config.epochs + 1):
        for param_group in optimizer.param_groups:
            print("learning rate at {0} epoch is {1}".format(epoch, param_group['lr']))

        log = {'epoch': epoch}

        # train the model
        model.train()
        result = train_epoch(epoch, model, train_dataloader, criterion, optimizer, lr_scheduler, train_metrics, device)
        log.update(result)

        # validate the model
        model.eval()
        result = valid_epoch(epoch, model, valid_dataloader, criterion, valid_metrics, device)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best_epoch = epoch
            best = True
        else:
            best = False

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best, config.save_freq)

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))

    print("Best accuracy : ",best_acc, ' for ',best_epoch)# saving class mean
    best_curr_acc = {'best_acc':best_acc,'best_epoch':best_epoch,
                     'curr_acc':log['val_acc1'],'curr_epoch':epoch}
    write_json(best_curr_acc,os.path.join(config.checkpoint_dir,'acc.json'))


if __name__ == '__main__':
    config = get_train_config()
    # device
    device, device_ids = setup_device(config.n_gpu)

    main(config, device, device_ids)