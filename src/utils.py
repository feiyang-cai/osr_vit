import json
from pathlib import Path
from datetime import datetime
import os
import torch
import importlib
import pandas as pd
import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_{}_bs{}_lr{}_wd{}_nc{}_rs{}'.format(config.dataset,config.model_arch, config.batch_size, config.lr, config.wd, config.num_classes, config.random_seed)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr

def load_checkpoint(path, new_img=384, patch=16, emb_dim=768,layers=12):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('npz'):
        keys, values = load_jax(path)
        state_dict = convert_jax_pytorch(keys, values)
    elif path.endswith('pth'):
        if 'deit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))['model']
        elif 'jx' in path or 'vit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))['state_dict']
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))

    if 'jx' in path or any(x in  os.path.basename(path) for x in ['vit','deit']): # for converting rightmann weight
            old_img = (24,24) #TODO: check if not needed
            # num_layers_model = layers  #
            # num_layers_state_dict = int((len(state_dict) - 8) / 12)
            # if num_layers_model != num_layers_state_dict:
            #     raise ValueError(
            #         f'Pretrained model has different number of layers: {num_layers_state_dict} than defined models layers: {num_layers_model}')
            #state_dict['class_token'] = state_dict.pop('cls_token')
            if 'distilled' in path:
                state_dict['distilled_token'] = state_dict.pop('dist_token')
            state_dict['transformer.pos_embedding.pos_embedding'] = state_dict.pop('pos_embed')
            state_dict['embedding.weight'] = state_dict.pop('patch_embed.proj.weight')
            state_dict['embedding.bias'] = state_dict.pop('patch_embed.proj.bias')
            if os.path.basename(path) == 'vit_small_p16_224-15ec54c9.pth' : # hack for vit small
                state_dict['embedding.weight'] = state_dict['embedding.weight'].reshape(768,3, 16,16)
            state_dict['classifier.weight'] = state_dict.pop('head.weight')
            state_dict['classifier.bias'] = state_dict.pop('head.bias')
            state_dict['transformer.norm.weight'] = state_dict.pop('norm.weight')
            state_dict['transformer.norm.bias'] = state_dict.pop('norm.bias')
            posemb = state_dict['transformer.pos_embedding.pos_embedding']
            for i, block_name in enumerate(list(state_dict.keys()).copy()):
                if 'blocks' in block_name:
                    new_block = "transformer.encoder_layers."+block_name.split('.',1)[1]
                    state_dict[new_block]=state_dict.pop(block_name)

    else:
        # resize positional embedding in case of diff image or grid size
        posemb = state_dict['transformer.pos_embedding.pos_embedding']
    # Deal with class token
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    model_grid_seq = new_img//patch
    ckpt_grid_seq = int(np.sqrt(posemb_grid.shape[0]))

    if model_grid_seq!=ckpt_grid_seq:
        # Get old and new grid sizes
        posemb_grid = posemb_grid.reshape(ckpt_grid_seq, ckpt_grid_seq, -1)

        posemb_grid = torch.unsqueeze(posemb_grid.permute(2, 0, 1), dim=0)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(model_grid_seq, model_grid_seq), mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)

        # Deal with class token and return
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        # if 'jx' in path:
        #     state_dict['pos_embed'] = posemb
        # else:
        state_dict['transformer.pos_embedding.pos_embedding'] = posemb
        print('Resized positional embedding from (%d,%d) to (%d,%d)'%(ckpt_grid_seq,ckpt_grid_seq,model_grid_seq,model_grid_seq))
    return state_dict

def load_jax(path):
    """ Loads params from a npz checkpoint previously stored with `save()` in jax implemetation """
    ckpt_dict = np.load(path, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    # with gfile.GFile(path, 'rb') as f:
    #     ckpt_dict = np.load(f, allow_pickle=False)
    #     keys, values = zip(*list(ckpt_dict.items()))
    return keys, values

def convert_jax_pytorch(keys, values):
    """ Convert jax model parameters with pytorch model parameters """
    state_dict = {}
    for key, value in zip(keys, values):

        # convert name to torch names
        names = key.split('/')
        torch_names = replace_names(names)
        torch_key = '.'.join(w for w in torch_names)

        # convert values to tensor and check shapes
        tensor_value = torch.tensor(value, dtype=torch.float)
        # check shape
        num_dim = len(tensor_value.shape)

        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            # for normal weight, transpose it
            tensor_value = tensor_value.T
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['query', 'key', 'value']:
            feat_dim, num_heads, head_dim = tensor_value.shape
            # for multi head attention q/k/v weight
            tensor_value = tensor_value
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['query', 'key', 'value']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            tensor_value = tensor_value
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    return state_dict

def replace_names(names):
    """ Replace jax model names with pytorch model names """
    new_names = []
    for name in names:
        if name == 'Transformer':
            new_names.append('transformer')
        elif name == 'encoder_norm':
            new_names.append('norm')
        elif 'encoderblock' in name:
            num = name.split('_')[-1]
            new_names.append('encoder_layers')
            new_names.append(num)
        elif 'LayerNorm' in name:
            num = name.split('_')[-1]
            if num == '0':
                new_names.append('norm{}'.format(1))
            elif num == '2':
                new_names.append('norm{}'.format(2))
        elif 'MlpBlock' in name:
            new_names.append('mlp')
        elif 'Dense' in name:
            num = name.split('_')[-1]
            new_names.append('fc{}'.format(int(num) + 1))
        elif 'MultiHeadDotProductAttention' in name:
            new_names.append('attn')
        elif name == 'kernel' or name == 'scale':
            new_names.append('weight')
        elif name == 'bias':
            new_names.append(name)
        elif name == 'posembed_input':
            new_names.append('pos_embedding')
        elif name == 'pos_embedding':
            new_names.append('pos_embedding')
        elif name == 'embedding':
            new_names.append('embedding')
        elif name == 'head':
            new_names.append('classifier')
        elif name == 'cls':
            new_names.append('cls_token')
        else:
            new_names.append(name)
    return new_names

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)
#
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
#
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res