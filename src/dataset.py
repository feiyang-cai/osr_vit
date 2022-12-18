from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader
import pickle


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

def getSVHNDataset(data_path='./data', **args):
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
        
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']

    data_split = 'train' if split=='train' else 'test'
    dataset = datasets.SVHN(data_path, download=True, split=data_split, transform=transform)

    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(10)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)
        unknown_mapping = {val: idx+len(known_classes) for idx, val in enumerate(unknown_classes)}

        classes = known_classes if split=='train' or split=='in_test' else unknown_classes
        mapping = known_mapping if split=='train' or split=='in_test' else unknown_mapping

        dataset.labels = np.array(dataset.labels)

        idx = None
        for i in classes:
            if idx is None:
                idx = (dataset.labels==i)
            else:
                idx |= (dataset.labels==i)

        dataset.labels = dataset.labels[idx]
        dataset.data = dataset.data[idx]
        for idx, val in enumerate(dataset.labels):
            dataset.labels[idx] = torch.tensor(mapping[val.item()])
        
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

class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target#, self.uq_idxs[idx]

def subsample_dataset_cub(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes_cub(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset_cub(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def getCUBDataset(data_path='./data', **args):
    mean = (0.4856, 0.4994, 0.4325)
    std = (0.2264, 0.2218, 0.2606)
        
    transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']
    data_split = True if split=='train' else False

    dataset = CustomCub2011(root=data_path, train=data_split, transform=transform)
    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(1, 200+1)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)

        if split == 'train' or split == 'in_test':
            dataset = subsample_classes_cub(dataset, include_classes=known_classes)
        else:
            dataset = subsample_classes_cub(dataset, include_classes=unknown_classes)
    
    return dataset

def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'fgvc-aircraft-2013b', 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in dataloader:
        data = data[0]
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
    import pickle
    with open("src/aircraft_osr_splits.pkl", 'rb') as f:
        splits = pickle.load(f)
        print(len(splits['known_classes']))
    
    dataset = getFGVCDataset(image_size=448, split='train')#, known_classes=splits['known_classes'])
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    mean, std = get_mean_and_std(loader)
    print(mean, std)
    #data = next(iter(loader))
    #print(data)
    #print(data.shape)
    #print(data[0][2][14][14])


