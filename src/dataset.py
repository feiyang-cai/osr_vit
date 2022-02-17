from torchvision import datasets, transforms
import torch
import numpy as np
import os
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
from torchvision.datasets.folder import default_loader
import pickle
from src.randaugment import RandAugment


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
            tar.extractall(path=self.root)

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
    
class FGVCAircraft(Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.
    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=True):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        import tarfile
        import requests

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(self.root)
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = requests.get(self.url, allow_redirects=True)


        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.content)

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')

def subsample_dataset_fgvc(dataset, idxs):

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes_fgvc(dataset, include_classes=range(60)):

    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset_fgvc(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def getFGVCDataset(data_path='./data', **args):
    mean = (0.4814, 0.5123, 0.5356)
    std = (0.2230, 0.2162, 0.2477)

    train_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.RandomCrop(args['image_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_transform.transforms.insert(0, RandAugment(2, 15))
        
    test_transform = transforms.Compose([
        transforms.Resize((args['image_size'], args['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    split = args['split']
    data_split = 'train' if split=='train' else 'test'
    transform = train_transform if split=='train' else test_transform

    dataset = FGVCAircraft(root=data_path, split=data_split, transform=transform)
    if 'known_classes' in args:
        known_classes = sorted(args['known_classes'])
        known_mapping = {val:idx for idx, val in enumerate(known_classes)}
        unknown_classes =  list(set(range(100)) -  set(known_classes))
        unknown_classes = sorted(unknown_classes)

        if split == 'train' or split == 'in_test':
            dataset = subsample_classes_fgvc(dataset, include_classes=known_classes)
        else:
            dataset = subsample_classes_fgvc(dataset, include_classes=unknown_classes)
    
    return dataset
    

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


