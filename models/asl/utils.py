import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import sys 
sys.path.append("..") 
sys.path.append("../..") 

from torchmeta.transforms import Categorical, ClassSplitter

def get_dataset(args, dataset_name, phase):
    if dataset_name == 'cub':
        from torchmeta.datasets import CUBMM as dataset_class
        image_size = 84
        padding_len = 8
    elif dataset_name == 'sun':
        from torchmeta.datasets import SUNMM as dataset_class
        image_size = 84
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    # augmentations
    # reference: https://github.com/Sha-Lab/FEAT
    if args.augment and phase == 'train':
        transforms_list = [
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transforms_list = [
            transforms.Resize((image_size+padding_len, image_size+padding_len)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]


    # pre-processing 
    if args.backbone == 'resnet12':
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    else:
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])


    # get datasets
    dataset = dataset_class(
        root=args.data_folder,
        num_classes_per_task=args.num_ways,
        meta_split=phase,
        transform=transforms_list,
        target_transform=Categorical(num_classes=args.num_ways),
        download=args.download
    )

    dataset = ClassSplitter(dataset, 
        shuffle=(phase == 'train'),
        num_train_per_class=args.num_shots, 
        num_test_per_class=args.test_shots
    )

    return dataset


def get_proto_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())

