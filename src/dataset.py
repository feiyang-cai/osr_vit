from torchvision import datasets, transforms
import torch
import numpy as np
import os


def getMNISTDataset(data_path='./data', **args):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)    
        
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']

    data_split = True if split=='train' else False
    dataset = datasets.MNIST(data_path, download=True, train=data_split, transform=transform)

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(10)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split=='train' or split=='in_test' else unknown_classes
        mapping = known_mapping if split=='train' or split=='in_test' else unknown_mapping

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.targets==i)
            else:
                idx |= (dataset.targets==i)

        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.targets):
            dataset.targets[idx] = torch.tensor(mapping[val.item()])
        
    return dataset

def getCIFAR10Dataset(data_path='./data', **args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
        
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']

    data_split = True if split=='train' else False
    dataset = datasets.CIFAR10(data_path, download=True, train=data_split, transform=transform)

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(10)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split=='train' or split=='in_test' else unknown_classes
        mapping = known_mapping if split=='train' or split=='in_test' else unknown_mapping

        dataset.targets = np.array(dataset.targets)

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.targets==i)
            else:
                idx |= (dataset.targets==i)

        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.targets):
            dataset.targets[idx] = torch.tensor(mapping[val.item()])
        
    return dataset

def getCIFAR100Dataset(data_path='./data', **args):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
        
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']

    data_split = True if split=='train' else False
    dataset = datasets.CIFAR100(data_path, download=True, train=data_split, transform=transform)

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(100)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split=='train' or split=='in_test' else unknown_classes
        mapping = known_mapping if split=='train' or split=='in_test' else unknown_mapping

        dataset.targets = np.array(dataset.targets)

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.targets==i)
            else:
                idx |= (dataset.targets==i)

        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.targets):
            dataset.targets[idx] = torch.tensor(mapping[val.item()])
        
    return dataset

class Tiny_ImageNet_Filter(datasets.ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

def getTinyImageNetDataset(data_path='./data', **args):
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)
        
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']

    data_split = 'train' if split=='train' else 'val'
    dataset = Tiny_ImageNet_Filter(os.path.join(data_path, 'tiny-imagenet-200', data_split), transform)

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(200)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        if split == 'in_test' or split == 'train':
            dataset.__Filter__(known=known_classes)
        else:
            dataset.__Filter__(known=unknown_classes)
            
    else:
        dataset.__Filter__(known=[i for i in range(200)])
        
    return dataset

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = getTinyImageNetDataset(image_size=32, split='out_test', known_classes=[1,2,6,8,21,182])
    print(dataset[-1][1])
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    #mean, std = get_mean_and_std(loader)
    #print(mean, std)
    data, target = next(iter(loader))
    print(target)
    #print(data[0][2][14][14])


