import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats


def get_backbone(name, state_dict=None):

    if name == 'conv4':
        from backbones import conv4
        backbone = conv4()
    elif name == 'resnet12':
        from backbones import resnet12
        backbone = resnet12()
    else:
        raise ValueError('Non-supported Backbone.')

    if state_dict is not None:
        backbone.load_state_dict(state_dict)

    return backbone
    

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Averager_with_interval():

    def __init__(self, confidence=0.95):
        self.list = []
        self.confidence = confidence
        self.n = 0

    def add(self, x):
        self.list.append(x)
        self.n += 1

    def item(self, return_str=False):
        mean, standard_error = np.mean(self.list), scipy.stats.sem(self.list)
        h = standard_error * scipy.stats.t._ppf((1 + self.confidence) / 2, self.n - 1)
        if return_str:
            return '{0:.2f}; {1:.2f}'.format(mean * 100, h * 100)
        else:
            return mean


def count_acc(logits, labels):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(labels).float())


def set_reproducibility(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_outputs_c_h(backbone, image_len):
    c_dict = {
        'conv4': 64,
        'resnet12': 512
    }
    c = c_dict[backbone]

    h_devisor_dict = {
        'conv4': 16,
        'resnet12': 16,
    }

    h = image_len // h_devisor_dict[backbone]

    return c, h


def get_semantic_size(args):

    semantic_size_list = []

    for semantic_type in args.semantic_type:

        if semantic_type == 'class_attributes':
            if args.train_data == 'cub':
                semantic_size_list.append(312)
        elif semantic_type == 'image_attributes':
            if args.train_data == 'sun':
                semantic_size_list.append(102)
        
    if not len(semantic_size_list) == len(args.semantic_type):
        raise ValueError('Non-supported Semantic Type to the Dataset.')
    if len(semantic_size_list) == 1:
        return semantic_size_list[0]
    else:
        return semantic_size_list


def get_inputs_and_outputs(args, batch):

    semantic_type_limitation = [
        'class_attributes',
        'image_attributes',
    ]

    for semantic_type in args.semantic_type:
        if semantic_type not in semantic_type_limitation:
            raise ValueError('Non-supported Semantic Type.')

    return_list = ['images', 'targets'] + args.semantic_type

    if args.use_cuda:
        return [batch[return_type].cuda(non_blocking=True) for return_type in return_list]
    else:
        return [batch[return_type] for return_type in return_list]

    